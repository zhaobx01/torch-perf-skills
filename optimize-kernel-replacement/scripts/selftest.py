#!/usr/bin/env python3
"""Self-test for scan_kernel_candidates.py.

Verifies all 13 static candidate detectors, common false-positive boundaries
(e.g. AdamW(fused=True), average_attn_weights=False, interpolate(mode="nearest"),
per-custom_op register_fake matching), and CLI flags (--json, --patterns, invalid
pattern rejection). Pure Python standard library (Python 3.6+ compatible).
"""

import json
import os
import subprocess
import sys
import tempfile

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_kernel_candidates.py")

SYNTHETIC_SAMPLE = """
import torch
import torch.nn as nn
import torch.nn.functional as F

# Positive & negative AdamW
opt_unfused = torch.optim.AdamW(params, lr=1e-3)
opt_fused = torch.optim.AdamW(params, lr=1e-3, fused=True)
opt_foreach = torch.optim.AdamW(params, lr=1e-3, foreach=True)

# Non-AdamW matrix optimizer
muon_opt = DistributedMuon(params)

# FlexAttention / BlockMask
y = flex_attention(q, k, v)
bm = create_block_mask(mask_mod, 1, 1, 128, 128)

# FA2 varlen vs dense
out_varlen = flash_attn_varlen_func(q, k, v, cu_q, cu_k, 128, 128)
out_dense = flash_attn_func(q, k, v)

# SDPA & MHA (positive vs negative average_attn_weights / need_weights)
s = F.scaled_dot_product_attention(q, k, v)
mha = nn.MultiheadAttention(256, 8)
a_default = F.multi_head_attention_forward(q, k, v)
a_no_avg = F.multi_head_attention_forward(q, k, v, average_attn_weights=False)
a_no_weights = F.multi_head_attention_forward(q, k, v, need_weights=False)

# Manual attention
attn = F.softmax(q @ k.transpose(-2, -1), dim=-1)

# Complex / polar RoPE
freqs_cis = torch.polar(ones, freqs)

# Norm & norm_then_add
ln = nn.LayerNorm(64)
z = self.norm1(x) + x

# Bilinear upsample/interpolate (nearest must NOT trigger)
u1 = nn.Upsample(scale_factor=2, mode="bilinear")
u2 = F.interpolate(x, scale_factor=2, mode="bilinear")
u3 = F.interpolate(x, scale_factor=2, mode="nearest")

# Custom autograd.Function
class CustomFn(torch.autograd.Function):
    pass

# Two custom_ops in one file: first has register_fake, second is missing it
@torch.library.custom_op("ns::op_with_fake", mutates_args=())
def op_with_fake(x):
    return x

@op_with_fake.register_fake
def op_with_fake_meta(x):
    return x

@torch.library.custom_op("ns::op_without_fake", mutates_args=())
def op_without_fake(x):
    return x
"""

EXPECTED_COUNTS = {
    "adamw_not_fused": 1,
    "non_adamw_optimizer": 1,
    "flex_attention_call": 2,
    "fa2_varlen_call": 1,
    "sdpa_call": 1,
    "mha_default_avg_weights": 2,
    "manual_attention": 1,
    "complex_rope": 1,
    "layernorm_module": 1,
    "norm_then_add": 1,
    "upsample_bilinear": 2,
    "custom_autograd": 1,
    "custom_op_missing_fake": 1,
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
        # 1. JSON mode: all 13 patterns + false-positive boundaries
        res_json = _run(["--json", sample_path])
        findings = json.loads(res_json.stdout)
        counts = {}
        for item in findings:
            p = item["pattern"]
            counts[p] = counts.get(p, 0) + 1
        assert counts == EXPECTED_COUNTS, "Expected {}, got {}".format(EXPECTED_COUNTS, counts)

        missing = [x for x in findings if x["pattern"] == "custom_op_missing_fake"]
        assert len(missing) == 1 and "ns::op_without_fake" in missing[0]["snippet"], missing

        # 2. Text mode + --patterns filter + static disclaimer on stderr
        res_text = _run(["--patterns", "complex_rope,upsample_bilinear", sample_path])
        assert "3 static candidate(s): 1 complex_rope, 2 upsample_bilinear" in res_text.stdout, res_text.stdout
        assert "STATIC candidates only" in res_text.stderr, res_text.stderr

        # 3. Unknown pattern must exit non-zero
        _run(["--patterns", "does_not_exist", sample_path], expect_ok=False)

        print("selftest PASSED: {} patterns verified, false-positive boundaries & CLI OK.".format(
            len(EXPECTED_COUNTS)
        ))
    finally:
        os.unlink(sample_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
