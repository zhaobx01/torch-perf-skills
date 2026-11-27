#!/usr/bin/env python3
"""Self-test for opt-runtime-env/scripts/scan_runtime_env.py.

Pure Python standard library (Python 3.6+ compatible).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_runtime_env.py")


def _run(args, expect_code=0):
    proc = subprocess.run(
        [sys.executable, SCRIPT] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    assert proc.returncode == expect_code, "Expected {}, got {}: {}\n{}".format(
        expect_code, proc.returncode, proc.stdout, proc.stderr
    )
    return proc


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # 1. YAML with all 5 categories, block scalar false-positive guard, and /dev/shm warning
        yaml_sample = Path(tmp) / "runtime.yaml"
        yaml_sample.write_text(
            "notes: |-\n"
            "  HIPBLASLT_ALLOW_TF32: '1'\n"
            "  NCCL_RINGS: 'fake'\n"
            "\n"
            "environment:\n"
            "  set:\n"
            "    # MIOPEN_FIND_MODE: '1'\n"
            "    HIPBLASLT_ALLOW_TF32: '0'\n"
            "    MIOPEN_FIND_MODE: '1'\n"
            "    PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True'\n"
            "    NCCL_ALGO: Ring\n"
            "    NCCL_RINGS: 'N0 0 1 N0'\n"
            "    TORCHINDUCTOR_CACHE_DIR: '/dev/shm/cache'\n"
            "    TORCHINDUCTOR_ALIGNMENT_ASSERTS: '0'\n",
            encoding="utf-8",
        )

        # 2. Shell launch script with export, inline prefix, comment, heredoc, and override conflict
        sh_sample = Path(tmp) / "launch.sh"
        sh_sample.write_text(
            "#!/usr/bin/env bash\n"
            "# export HIPBLASLT_ALLOW_TF32=1\n"
            "export ROCBLAS_MATH_MODE=1\n"
            "cat <<'EOF' > /tmp/readme.txt\n"
            "NCCL_RINGS=ignored_inside_heredoc\n"
            "EOF\n"
            "MIOPEN_FIND_MODE=1 TOKENIZERS_PARALLELISM=false torchrun train.py\n"
            "export ROCBLAS_MATH_MODE=0  # conflicting re-assignment\n",
            encoding="utf-8",
        )

        # 3. Python launcher with os alias, setdefault, update, and non-tracked env
        py_sample = Path(tmp) / "launcher.py"
        py_sample.write_text(
            "import os as _os\n"
            "from os import environ as env\n"
            "_os.environ['GPU_MAX_HW_QUEUES'] = '16'\n"
            "env.setdefault('TRITON_F32_DEFAULT', 'tf32')\n"
            "env.update({'DISABLE_ADDMM_CUDA_LT': '1', 'UNRELATED_KEY': '1'})\n",
            encoding="utf-8",
        )

        res = _run(["--json", str(yaml_sample), str(sh_sample), str(py_sample)])
        assert "STATIC findings only" in res.stderr, res.stderr
        payload = json.loads(res.stdout)
        assert payload["files_scanned"] == 3 and payload["files_skipped"] == 0, payload

        yaml_hits = [(f["variable"], f["pattern"], f["severity"]) for f in payload["findings"] if f["file"] == str(yaml_sample)]
        assert yaml_hits == [
            ("HIPBLASLT_ALLOW_TF32", "precision_env_guard", "medium"),
            ("MIOPEN_FIND_MODE", "layout_miopen_env", "info"),
            ("PYTORCH_CUDA_ALLOC_CONF", "memory_hw_env", "info"),
            ("NCCL_ALGO", "comm_thread_env", "info"),
            ("NCCL_RINGS", "topology_bound_env", "high"),
            ("TORCHINDUCTOR_CACHE_DIR", "compile_cache_env", "medium"),
            ("TORCHINDUCTOR_ALIGNMENT_ASSERTS", "compile_cache_env", "medium"),
        ], yaml_hits

        sh_hits = [(f["variable"], f["pattern"]) for f in payload["findings"] if f["file"] == str(sh_sample)]
        assert sh_hits == [
            ("ROCBLAS_MATH_MODE", "precision_env_config"),
            ("TOKENIZERS_PARALLELISM", "comm_thread_env"),
            ("MIOPEN_FIND_MODE", "layout_miopen_env"),
            ("ROCBLAS_MATH_MODE", "env_override_conflict"),
            ("ROCBLAS_MATH_MODE", "precision_env_config"),
        ], sh_hits

        py_hits = [(f["variable"], f["pattern"]) for f in payload["findings"] if f["file"] == str(py_sample)]
        assert py_hits == [
            ("GPU_MAX_HW_QUEUES", "memory_hw_env"),
            ("TRITON_F32_DEFAULT", "precision_env_config"),
            ("DISABLE_ADDMM_CUDA_LT", "precision_env_guard"),
        ], py_hits

        # Invalid path / unsupported extension -> exit 2
        _run(["--json", os.path.join(tmp, "missing.yaml")], expect_code=2)
        bad_ext = Path(tmp) / "notes.txt"
        bad_ext.write_text("HIPBLASLT_ALLOW_TF32=1\n", encoding="utf-8")
        _run([str(bad_ext)], expect_code=2)

        # Empty dir -> exit 1
        empty = Path(tmp) / "empty"
        empty.mkdir()
        _run(["--json", str(empty)], expect_code=1)

        # Oversized YAML, binary shell, broken Python -> exit 1 with skipped_files
        large = Path(tmp) / "large.yaml"
        large.write_text("MIOPEN_FIND_MODE: '1'\n# " + "x" * (2 * 1024 * 1024), encoding="utf-8")
        binary = Path(tmp) / "binary.sh"
        binary.write_bytes(b"export MIOPEN_FIND_MODE=1\n\xff")
        broken = Path(tmp) / "broken.py"
        broken.write_text("def bad(\n", encoding="utf-8")
        for sample in (large, binary, broken):
            proc = _run(["--json", str(yaml_sample), str(sample)], expect_code=1)
            out = json.loads(proc.stdout)
            assert out["files_scanned"] == 1 and out["files_skipped"] == 1, out
            assert out["skipped_files"][0]["file"] == str(sample), out

        # --help works
        assert "5 categories" in _run(["--help"]).stdout

        cases = {
            "scoped.yaml": (
                "metadata:\n  MIOPEN_FIND_MODE: 'ignored'\n"
                "environment:\n  set:\n    HIPBLASLT_ALLOW_TF32: '1'\n"
                "    TORCH_COMPILE_CACHE_KEY_TAG: 'ab#cd' # comment\n"
                "    TORCHINDUCTOR_EMULATE_PRECISION_CASTS: '0'\n"
                "  unset: [NCCL_RINGS]\n  accuracy:\n"
                "    unset:\n      - MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32\n"
                "    set:\n      HIPBLASLT_ALLOW_TF32: '0'\n",
                [("HIPBLASLT_ALLOW_TF32", "1", "precision_env_config"),
                 ("TORCH_COMPILE_CACHE_KEY_TAG", "ab#cd", "compile_cache_env"),
                 ("TORCHINDUCTOR_EMULATE_PRECISION_CASTS", "0", "precision_env_config"),
                 ("NCCL_RINGS", "<unset>", "env_unset"),
                 ("MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32", "<unset>", "env_unset"),
                 ("HIPBLASLT_ALLOW_TF32", "0", "precision_env_guard")],
            ),
            "commands.sh": (
                'echo "flag HIPBLASLT_ALLOW_TF32=1"\necho MIOPEN_FIND_MODE=2\n'
                'echo "see <<EOF"\nexport HIPBLASLT_ALLOW_TF32=\n'
                'MIOPEN_FIND_MODE=1 python a.py\nMIOPEN_FIND_MODE=2 python b.py\n'
                'export ROCBLAS_MATH_MODE=1| cat\nexport MIOPEN_FIND_MODE=$(get_mode)\n',
                [("HIPBLASLT_ALLOW_TF32", "", "precision_env_config"),
                 ("MIOPEN_FIND_MODE", "1", "layout_miopen_env"),
                 ("MIOPEN_FIND_MODE", "2", "layout_miopen_env"),
                 ("ROCBLAS_MATH_MODE", "1", "precision_env_config"),
                 ("MIOPEN_FIND_MODE", "<dynamic>", "layout_miopen_env")],
            ),
            "continuation.sh": (
                'PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \\\n'
                '  MIOPEN_FIND_MODE=1 torchrun train.py\n',
                [("MIOPEN_FIND_MODE", "1", "layout_miopen_env"),
                 ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True", "memory_hw_env")],
            ),
            "scopes.py": (
                "import os\nimport os as real_os\n"
                "os.environ['ROCBLAS_MATH_MODE'] = '1'\n"
                "def f(os):\n    os.environ['HIPBLASLT_ALLOW_TF32'] = '1'\n"
                "def setup():\n    from os import environ as env\n"
                "    env['GPU_MAX_HW_QUEUES']='16'\n"
                "def unrelated(env):\n    env['GPU_MAX_HW_QUEUES']='4'\n"
                "if False:\n    real_os.environ['HIPBLASLT_ALLOW_TF32'] = '1'\n"
                "real_os.environ['MIOPEN_FIND_MODE'] = '2'\n"
                "real_os.environ.setdefault('MIOPEN_FIND_MODE', '1')\n",
                [("ROCBLAS_MATH_MODE", "1", "precision_env_config"),
                 ("GPU_MAX_HW_QUEUES", "16", "memory_hw_env"),
                 ("MIOPEN_FIND_MODE", "2", "layout_miopen_env"),
                 ("MIOPEN_FIND_MODE", "1", "layout_miopen_env")],
            ),
            "shell_literals.sh": (
                "'MIOPEN_FIND_MODE=1'\nMIOPEN_FIND_MODE\\=2\n"
                "printf '%s\\n' 'first\nMIOPEN_FIND_MODE=1\nlast'\n"
                "cat <<'END-OF-DOC'\nMIOPEN_FIND_MODE=1\nEND-OF-DOC\n"
                "export MIOPEN_FIND_MODE=2\n",
                [("MIOPEN_FIND_MODE", "2", "layout_miopen_env")],
            ),
            "fast.env": (
                "# MIOPEN_FIND_MODE=2\nPYTORCH_ALLOC_CONF=expandable_segments:True\n"
                "TORCH_COMPILE_CACHE_KEY_TAG='ab#cd'\nPYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n",
                [("PYTORCH_ALLOC_CONF", "expandable_segments:True", "memory_hw_env"),
                 ("TORCH_COMPILE_CACHE_KEY_TAG", "ab#cd", "compile_cache_env"),
                 ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True", "memory_hw_env")],
            ),
            "generic_env.yaml": (
                "metadata:\n  HIPBLASLT_ALLOW_TF32: 'ignored'\n"
                "env:\n  HIPBLASLT_ALLOW_TF32: '1'\n"
                "environment:\n  MIOPEN_FIND_MODE: '1'\n",
                [("HIPBLASLT_ALLOW_TF32", "1", "precision_env_config"),
                 ("MIOPEN_FIND_MODE", "1", "layout_miopen_env")],
            ),
            "shell_env_unset.sh": (
                "env -i -u HIPBLASLT_ALLOW_TF32 MIOPEN_FIND_MODE=1 python train.py\n"
                "env --unset=ROCBLAS_MATH_MODE torchrun train.py\n"
                "unset NCCL_RINGS\n",
                [("HIPBLASLT_ALLOW_TF32", "<unset>", "env_unset"),
                 ("MIOPEN_FIND_MODE", "1", "layout_miopen_env"),
                 ("ROCBLAS_MATH_MODE", "<unset>", "env_unset"),
                 ("NCCL_RINGS", "<unset>", "env_unset")],
            ),
            "py_unset_and_dead_else.py": (
                "import os\n"
                "if True:\n    os.environ['HIPBLASLT_WORKSPACE_SIZE'] = '32768'\n"
                "else:\n    os.environ['HIPBLASLT_ALLOW_TF32'] = '1'\n"
                "os.environ.pop('MIOPEN_FIND_MODE', None)\n"
                "del os.environ['ROCBLAS_MATH_MODE']\n"
                "os.unsetenv('NCCL_ALGO')\n",
                [("HIPBLASLT_WORKSPACE_SIZE", "32768", "memory_hw_env"),
                 ("MIOPEN_FIND_MODE", "<unset>", "env_unset"),
                 ("ROCBLAS_MATH_MODE", "<unset>", "env_unset"),
                 ("NCCL_ALGO", "<unset>", "env_unset")],
            ),
        }
        failures = []
        for name, (source, expected) in cases.items():
            sample = Path(tmp) / name
            sample.write_text(source, encoding="utf-8")
            try:
                out = json.loads(_run(["--json", str(sample)]).stdout)
                assert set(out) == {"disclaimer", "files_scanned", "files_skipped", "skipped_files", "findings"}
                got = [(row["variable"], row["value"], row["pattern"]) for row in out["findings"]]
                assert got == expected, "expected {}, got {}".format(expected, got)
                if name == "scoped.yaml":
                    assert "environment.accuracy.unset" in out["findings"][4]["note"]
            except AssertionError as exc:
                failures.append("{}: {}".format(name, exc))
        assert not failures, "\n".join(failures)

    print("PASS: scan_runtime_env.py self-tests succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
