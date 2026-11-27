#!/usr/bin/env python3
"""Self-test for opt-precision/scripts/scan_precision_candidates.py.

Pure Python standard library (Python 3.6+ compatible).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_precision_candidates.py")


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
        py_sample = os.path.join(tmp, "sample.py")
        with open(py_sample, "w", encoding="utf-8") as fh:
            fh.write(
                "import os\n"
                "import torch\n"
                "\n"
                "def bad_bf16_check():\n"
                "    return torch.version.cuda is not None and torch.cuda.is_bf16_supported()\n"
                "\n"
                "def good_bf16_check():\n"
                "    return (torch.version.cuda or torch.version.hip) and torch.cuda.is_bf16_supported()\n"
                "\n"
                "torch.backends.cuda.matmul.allow_tf32 = False\n"
                "torch.set_float32_matmul_precision('highest')\n"
                "os.environ['HIPBLASLT_ALLOW_TF32'] = '1'\n"
                "os.environ['HIPBLASLT_ALLOW_TF32'] = '0'\n"
                "\n"
                "with torch.autocast('cuda', enabled=False):\n"
                "    pass\n"
                "with torch.autocast('cuda', dtype=torch.float16):\n"
                "    pass\n"
                "\n"
                "def get_float32_precision():\n"
                "    return \"'tf32'\"\n"
            )

        yaml_sample = os.path.join(tmp, "runtime.yaml")
        with open(yaml_sample, "w", encoding="utf-8") as fh:
            fh.write(
                "environment:\n"
                "  set:\n"
                "    # HIPBLASLT_ALLOW_TF32: '1'\n"
                "    HIPBLASLT_ALLOW_TF32: '0'\n"
                "    MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32: '1'\n"
            )

        res = _run(["--json", py_sample, yaml_sample])
        assert "STATIC candidates only" in res.stderr, res.stderr
        payload = json.loads(res.stdout)
        assert payload["files_scanned"] == 2 and payload["files_skipped"] == 0, payload
        assert "STATIC candidates only" in payload["disclaimer"], payload

        py_pats = [f["pattern"] for f in payload["findings"] if f["file"] == py_sample]
        assert py_pats == [
            "cuda_only_bf16_check",
            "tf32_disabled_explicitly",
            "tf32_disabled_explicitly",
            "tf32_env_config",
            "tf32_disabled_explicitly",
            "fp32_autocast_disabled",
            "fp16_autocast_cue",
            "tf32_hook_missing_cache_tag",
        ], py_pats

        yaml_pats = [(f["line"], f["pattern"]) for f in payload["findings"] if f["file"] == yaml_sample]
        assert yaml_pats == [(4, "tf32_disabled_explicitly"), (5, "tf32_env_config")], yaml_pats

        # Invalid path -> exit 2
        _run(["--json", os.path.join(tmp, "missing.py")], expect_code=2)

        # Empty dir -> exit 1
        empty = os.path.join(tmp, "empty")
        os.mkdir(empty)
        _run(["--json", empty], expect_code=1)

        # Syntax error -> exit 1 with skipped_files
        broken = os.path.join(tmp, "broken.py")
        with open(broken, "w", encoding="utf-8") as fh:
            fh.write("def bad(\n")
        res_bad = _run(["--json", py_sample, broken], expect_code=1)
        bad_payload = json.loads(res_bad.stdout)
        assert bad_payload["files_scanned"] == 1 and bad_payload["files_skipped"] == 1, bad_payload

        cases = [
            ("ordinary_precision_helper", "def get_float32_precision(): return 'ieee'\n", []),
            ("conditional_precision_helper", "def get_float32_precision():\n"
             "    if torch.version.hip: return 'ieee'\n"
             "    return 'tf32'\n", []),
            ("comment_is_not_cache_tag", "# cache_key_tag\n"
             "def get_float32_precision(): return 'tf32'\n", ["tf32_hook_missing_cache_tag"]),
            ("real_cache_tag", "import torch.compiler.config as compiler_config\n"
             "compiler_config.cache_key_tag = 'flex-tf32'\n"
             "def get_float32_precision(): return 'tf32'\n", []),
            ("env_cache_tag", "os.environ['TORCH_COMPILE_CACHE_KEY_TAG'] = 'flex-tf32'\n"
             "def get_float32_precision(): return 'tf32'\n", []),
            ("no_effective_tag", "torch.compiler.config.cache_key_tag = None\n"
             "if False: torch.compiler.config.cache_key_tag = 'dead-branch'\n"
             "def get_float32_precision(): return 'tf32'\n", ["tf32_hook_missing_cache_tag"]),
            ("constant_tf32_return", "_PRECISION = 'tf32'\n"
             "def get_float32_precision(): return f\"'{_PRECISION}'\"\n", ["tf32_hook_missing_cache_tag"]),
            ("unrelated_precision_api", "client.allow_tf32 = False\n"
             "client.set_float32_matmul_precision('highest')\n"
             "client.autocast(dtype=torch.bfloat16)\n"
             "factory().torch.amp.autocast('cuda', torch.bfloat16)\n", []),
            ("shadowed_autocast", "from torch import autocast as cast\n"
             "def f(cast): return cast('cuda', torch.bfloat16)\n", []),
            ("dynamic_autocast_positions", "torch.cuda.amp.autocast(*[False], torch.bfloat16)\n", []),
            ("independent_cuda_and_bf16", "def f():\n"
             "    if torch.version.cuda: pass\n"
             "    return torch.cuda.is_bf16_supported()\n", []),
            ("cuda_guarded_bf16", "def f():\n"
             "    if torch.version.cuda:\n"
             "        return torch.cuda.is_bf16_supported()\n"
             "    return False\n", ["cuda_only_bf16_check"]),
            ("backend_precision_settings", "torch.backends.cuda.matmul.fp32_precision = 'ieee'\n"
             "torch.backends.cudnn.conv.fp32_precision = 'tf32'\n"
             "client.fp32_precision = 'ieee'\n", ["tf32_disabled_explicitly", "fp32_precision_config"]),
            ("explicit_autocast", "from torch.cuda.amp import autocast as legacy_autocast\n"
             "with torch.autocast('cuda', None, False): pass\n"
             "with torch.amp.autocast('cuda', torch.bfloat16): pass\n"
             "with legacy_autocast(False, torch.float16): pass\n"
             "with torch.cuda.amp.autocast(True, torch.bfloat16): pass\n",
             ["fp32_autocast_disabled", "bf16_autocast_cue", "fp32_autocast_disabled", "bf16_autocast_cue"]),
            ("torch_and_api_aliases", "import torch as th\n"
             "from torch import set_float32_matmul_precision as set_prec\n"
             "th.backends.cuda.matmul.allow_tf32 = False\n"
             "th.backends.cuda.matmul.fp32_precision = 'ieee'\n"
             "th.set_float32_matmul_precision('highest')\n"
             "set_prec('highest')\n"
             "with th.autocast('cuda', dtype=th.bfloat16): pass\n"
             "with th.amp.autocast('cuda', enabled=False): pass\n",
             [
                 "tf32_disabled_explicitly",
                 "tf32_disabled_explicitly",
                 "tf32_disabled_explicitly",
                 "tf32_disabled_explicitly",
                 "bf16_autocast_cue",
                 "fp32_autocast_disabled",
             ]),
            ("reversed_cuda_gate_and_alias", "import torch as th\n"
             "def pick():\n"
             "    return None is not th.version.cuda and th.cuda.is_bf16_supported()\n",
             ["cuda_only_bf16_check"]),
            ("annotated_const_multiline_precision_helper", "DEFAULT_PREC: str = 'tf32'\n"
             "def get_float32_precision():\n"
             "    print('using tf32')\n"
             "    return DEFAULT_PREC\n",
             ["tf32_hook_missing_cache_tag"]),
            ("partial_return_precision_helper", "def get_float32_precision(flag):\n"
             "    if flag:\n"
             "        return 'tf32'\n",
             []),
        ]
        failures = []
        for name, source, expected in cases:
            sample = Path(tmp) / (name + ".py")
            sample.write_text(source, encoding="utf-8")
            output = json.loads(_run(["--json", str(sample)]).stdout)
            actual = [item["pattern"] for item in output["findings"]]
            if actual != expected:
                failures.append("{}: expected {}, got {}".format(name, expected, actual))

        yaml_block = Path(tmp) / "block_scalar.yaml"
        yaml_block.write_text(
            "notes: |-\n"
            "  HIPBLASLT_ALLOW_TF32: '0'\n"
            "  ROCBLAS_MATH_MODE: '1'\n"
            "\n"
            "  MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32: '1'\n"
            "environment:\n"
            "  set:\n"
            "    HIPBLASLT_ALLOW_TF32: '1'\n",
            encoding="utf-8",
        )
        block_out = json.loads(_run(["--json", str(yaml_block)]).stdout)
        block_pats = [(item["line"], item["pattern"]) for item in block_out["findings"]]
        if block_pats != [(8, "tf32_env_config")]:
            failures.append("yaml_block_scalar: expected [(8, 'tf32_env_config')], got {}".format(block_pats))

        help_text = _run(["--help"]).stdout
        if "YAML/JSON config" in help_text:
            failures.append("Help advertises JSON input but .json input is unsupported")
        json_sample = Path(tmp) / "sample.json"
        json_sample.write_text('{"HIPBLASLT_ALLOW_TF32":"0"}', encoding="utf-8")
        _run([str(json_sample)], expect_code=2)

        large = Path(tmp) / "large.yaml"
        large.write_text("HIPBLASLT_ALLOW_TF32: '0'\n# " + "x" * (2 * 1024 * 1024), encoding="utf-8")
        malformed = Path(tmp) / "binary.yaml"
        malformed.write_bytes(b"HIPBLASLT_ALLOW_TF32: '0'\n\xff")
        deep_source = Path(tmp) / "deep.py"
        deep_source.write_text("value = root" + ".child" * 6000 + "\n", encoding="utf-8")
        for sample in (large, malformed, deep_source):
            proc = subprocess.run([sys.executable, SCRIPT, "--json", py_sample, str(sample)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if not proc.stdout:
                failures.append("{}: partial scan returned no JSON envelope".format(sample.name))
                continue
            output = json.loads(proc.stdout)
            if (proc.returncode != 1 or output["files_scanned"] != 1 or output["files_skipped"] != 1
                    or output["skipped_files"][0]["file"] != str(sample)):
                failures.append("{}: expected a reported skip and exit 1, got exit {} with {} scanned".format(
                    sample.name, proc.returncode, output["files_scanned"]))

        assert not failures, "\n".join(failures)

    print("PASS: scan_precision_candidates.py self-tests succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
