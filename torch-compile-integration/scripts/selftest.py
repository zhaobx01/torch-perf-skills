#!/usr/bin/env python3
"""Self-test for scan_compile_candidates.py.

Verifies all 7 static compile candidate/risk detectors, import alias & scope
resolution, async function decorators, common false-positive boundaries,
JSON envelope metadata, encoding/NUL-byte error handling, and CLI exit codes.
Pure Python standard library (Python 3.6+ compatible).
"""

import json
import os
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_compile_candidates.py")

SYNTHETIC_SAMPLE = """
import re
import torch
import torch.nn as nn
import torch._dynamo.config
from torch._inductor.inductor_prims import force_stride_order
from torch._inductor.runtime import triton_heuristics as th
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# False positive: re.compile must NOT trigger existing_compile_entry
PAT = re.compile(r"blocks\\.\\d+")

# Positive compile entries
compiled_fn = torch.compile(fn, mode="default")
model.compile(backend="inductor")

# Disable boundary (sync + async)
@torch.compiler.disable
def eager_island(x):
    return x

@torch.compiler.disable
async def async_eager_island(x):
    return x

# Checkpoint & distributed wrapping cues (direct + aliased FSDP)
wrapped = checkpoint_wrapper(layer)
ddp_model = DistributedDataParallel(model)
fsdp_model = FSDP(model)

# Global config mutation (positive) vs scoped config.patch (negative)
torch._dynamo.config.optimize_ddp = False
with torch._dynamo.config.patch(optimize_ddp=False):
    pass

# Multiple ModuleLists in one file collapse to 1 summary
layers_a = nn.ModuleList([nn.Linear(4, 4)])
layers_b = nn.ModuleList([nn.Linear(4, 4)])
"""

EXPECTED_COUNTS = {
    "existing_compile_entry": 2,
    "compile_disable_boundary": 2,
    "checkpoint_wrap_cue": 1,
    "distributed_wrap_cue": 2,
    "global_compiler_config": 1,
    "private_inductor_api": 2,
    "repeated_module_list": 1,
}


def _run(args, expect_ok=True, expect_code=None):
    proc = subprocess.run(
        [sys.executable, SCRIPT] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if expect_code is not None:
        if proc.returncode != expect_code:
            raise AssertionError("Expected exit {}, got {}: {}\n{}".format(
                expect_code, proc.returncode, proc.stdout, proc.stderr
            ))
    elif expect_ok and proc.returncode != 0:
        raise AssertionError("CLI failed ({}): {}\n{}".format(proc.returncode, proc.stdout, proc.stderr))
    elif not expect_ok and proc.returncode == 0:
        raise AssertionError("Expected non-zero exit for args {}".format(args))
    return proc


def test_aliases_and_false_positives(directory):
    path = os.path.join(directory, "aliases.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "import torch as th\n"
            "from torch import compile as tc\n"
            "from my_pkg import compile as custom_compile\n"
            "from other_pkg import FullyShardedDataParallel as FakeFSDP\n"
            "import torch.utils.checkpoint as cp\n"
            "\n"
            "@custom_compile\n"
            "def unrelated(x):\n"
            "    return x\n"
            "\n"
            "@tc\n"
            "def compiled_via_alias(x):\n"
            "    return x\n"
            "\n"
            "y = th.compile(unrelated)\n"
            "z = FakeFSDP(model)\n"
            "m = cnn.ModuleList([a])\n"
            "w1 = ptd_checkpoint_wrapper(block)\n"
            "w2 = cp.checkpoint_sequential(blocks, 2, x)\n"
            "model.set_gradient_checkpointing(True)\n"
        )
    res = _run(["--json", path])
    payload = json.loads(res.stdout)
    items = [(f["line"], f["pattern"]) for f in payload["findings"]]
    assert items == [
        (11, "existing_compile_entry"),
        (15, "existing_compile_entry"),
        (18, "checkpoint_wrap_cue"),
        (19, "checkpoint_wrap_cue"),
        (20, "checkpoint_wrap_cue"),
    ], items


def test_cli_status_and_malformed_inputs(directory):
    # Invalid / missing path -> exit 2
    res_missing = _run(["--json", os.path.join(directory, "missing.py")], expect_code=2)
    assert not res_missing.stdout and "missing.py" in res_missing.stderr, res_missing

    # Empty directory -> exit 1 with JSON envelope
    empty = os.path.join(directory, "empty")
    os.mkdir(empty)
    res_empty = _run(["--json", empty], expect_code=1)
    payload = json.loads(res_empty.stdout)
    assert payload["files_scanned"] == 0 and payload["files_skipped"] == 0, payload
    assert "STATIC candidates only" in payload["disclaimer"], payload
    assert "STATIC candidates only" in res_empty.stderr, res_empty.stderr

    # Partial scan: cp1252 valid file + syntax error + NUL byte -> exit 1, no Traceback
    encoded = os.path.join(directory, "encoded.py")
    broken = os.path.join(directory, "broken.py")
    nul_file = os.path.join(directory, "nul.py")
    with open(encoded, "wb") as handle:
        handle.write(b"# coding: cp1252\ncaf\xe9 = torch.compile(fn)\n")
    with open(broken, "w", encoding="utf-8") as handle:
        handle.write("def invalid(\n")
    with open(nul_file, "wb") as handle:
        handle.write(b"x = 1\x00\n")

    res_partial = _run(["--json", encoded, broken, nul_file], expect_code=1)
    assert "Traceback" not in res_partial.stderr, res_partial.stderr
    assert "STATIC candidates only" in res_partial.stderr, res_partial.stderr
    payload = json.loads(res_partial.stdout)
    assert payload["files_scanned"] == 1 and payload["files_skipped"] == 2, payload
    skipped_paths = {item["file"] for item in payload["skipped_files"]}
    assert skipped_paths == {broken, nul_file}, payload
    assert [f["snippet"] for f in payload["findings"]] == ["caf\u00e9 = torch.compile(fn)"], payload


def main():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(SYNTHETIC_SAMPLE)
        sample_path = tf.name

    try:
        res_json = _run(["--json", sample_path])
        assert "STATIC candidates only" in res_json.stderr, res_json.stderr
        payload = json.loads(res_json.stdout)
        assert "STATIC candidates only" in payload["disclaimer"], payload
        assert payload["files_scanned"] == 1 and payload["files_skipped"] == 0, payload
        findings = payload["findings"]
        counts = {}
        for item in findings:
            p = item["pattern"]
            counts[p] = counts.get(p, 0) + 1
        assert counts == EXPECTED_COUNTS, "Expected {}, got {}".format(EXPECTED_COUNTS, counts)

        ml = [x for x in findings if x["pattern"] == "repeated_module_list"][0]
        assert "2 nn.ModuleList site(s)" in ml["snippet"], ml

        res_text = _run(["--patterns", "existing_compile_entry,global_compiler_config", sample_path])
        assert "3 static candidate(s): 2 existing_compile_entry, 1 global_compiler_config" in res_text.stdout, res_text.stdout
        assert "STATIC candidates only" in res_text.stderr, res_text.stderr

        _run(["--patterns", "unknown_pat", sample_path], expect_code=2)

        with tempfile.TemporaryDirectory() as directory:
            test_aliases_and_false_positives(directory)
            test_cli_status_and_malformed_inputs(directory)

        print("selftest PASSED: {} patterns verified, alias/async/false-positive boundaries, JSON envelope & CLI status OK.".format(
            len(EXPECTED_COUNTS)
        ))
    finally:
        os.unlink(sample_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
