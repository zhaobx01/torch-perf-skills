#!/usr/bin/env python3
"""Self-test for scan_channels_last_candidates.py.

Verifies all 7 static candidate detectors, per-file Conv2d/Conv3d summarization,
common false-positive boundaries (e.g. non-NHWC permute, non-5D/4D rearrange,
state_dict outside save path), and CLI flags (--json, --patterns, invalid pattern).
Pure Python standard library (Python 3.6+ compatible).
"""

import json
import os
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_channels_last_candidates.py")

SYNTHETIC_SAMPLE = """
import os
import torch
import torch.nn as nn
from einops import rearrange

# Multiple Conv2d / Conv3d sites (including Conv3d subclass) collapse to 1 summary each
class CausalConv3d(nn.Conv3d):
    pass

c2a = nn.Conv2d(3, 16, 3)
c2b = nn.ConvTranspose2d(16, 3, 3)
c3a = nn.Conv3d(3, 16, 3)
c3b = nn.ConvTranspose3d(16, 3, 3)

# 5D <-> 4D rearrange (positive) vs 3D rearrange (negative)
y4 = rearrange(x5, "b c t h w -> (b t) c h w")
y5 = rearrange(y4, "(b t) c h w -> b c t h w", t=4)
y3 = rearrange(tokens, "b n d -> b d n")

# Logical NHWC permute (positive) vs attention head permute (negative)
nhwc = x.permute(0, 2, 3, 1)
attn = q.permute(0, 2, 1, 3)

# Existing channels_last & MIOpen env var
m = c2a.to(memory_format=torch.channels_last)
os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NHWC", "1")

# Optimizer checkpoint save (positive) vs non-save state_dict (negative)
def save_checkpoint(optimizer, path):
    state = optimizer.state_dict()
    return state

def inspect_state(optimizer):
    return optimizer.state_dict()
"""

EXPECTED_COUNTS = {
    "conv2d_subgraph": 1,
    "conv3d_subgraph": 1,
    "rearrange_5d_4d": 2,
    "permute_nhwc_trap": 1,
    "existing_channels_last": 1,
    "miopen_suggest_env": 1,
    "optimizer_state_save": 1,
}


def _run(args, expect_ok=True):
    proc = subprocess.run(
        [sys.executable, SCRIPT] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if expect_ok and proc.returncode != 0:
        raise AssertionError("CLI failed ({}): {}\n{}".format(proc.returncode, proc.stdout, proc.stderr))
    if not expect_ok and proc.returncode == 0:
        raise AssertionError("Expected non-zero exit for args {}".format(args))
    return proc


def main():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(SYNTHETIC_SAMPLE)
        sample_path = tf.name

    try:
        # 1. JSON mode: all 7 patterns + per-file collapse + false-positive boundaries
        res_json = _run(["--json", sample_path])
        findings = json.loads(res_json.stdout)
        counts = {}
        for item in findings:
            p = item["pattern"]
            counts[p] = counts.get(p, 0) + 1
        assert counts == EXPECTED_COUNTS, "Expected {}, got {}".format(EXPECTED_COUNTS, counts)

        c2 = [x for x in findings if x["pattern"] == "conv2d_subgraph"][0]
        assert "2 Conv2d/ConvTranspose2d site(s)" in c2["snippet"], c2

        # 2. Text mode + --patterns filter + static disclaimer on stderr
        res_text = _run(["--patterns", "rearrange_5d_4d,permute_nhwc_trap", sample_path])
        assert "3 static candidate(s): 1 permute_nhwc_trap, 2 rearrange_5d_4d" in res_text.stdout, res_text.stdout
        assert "STATIC candidates only" in res_text.stderr, res_text.stderr

        # 3. Unknown pattern must exit non-zero
        _run(["--patterns", "nonexistent_pattern", sample_path], expect_ok=False)

        print("selftest PASSED: {} patterns verified, per-file summary & false-positive boundaries OK.".format(
            len(EXPECTED_COUNTS)
        ))
    finally:
        os.unlink(sample_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
