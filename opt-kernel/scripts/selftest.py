#!/usr/bin/env python3
"""Self-test for scan_kernel_candidates.py.

Verifies all 13 detectors, false-positive boundaries, import alias & scope-aware
fake registration matching, JSON metadata envelope, and CLI behavior on complete
and incomplete scans. Uses only the Python standard library (Python 3.6+ compatible).
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

# Norm & add_then_norm (norm(x + residual), NOT norm(x) + x)
ln = nn.LayerNorm(64)
z = self.norm1(x + residual)
z_non_fused_order = self.norm1(x) + residual

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
    "add_then_norm": 1,
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


def test_fake_registration(directory):
    path = os.path.join(directory, "fake_registration.py")
    with open(path, "w") as sample:
        sample.write('''
import torch.library as tl
from torch.library import custom_op as reg_op

@torch.library.custom_op("ns::registered", mutates_args=())
def registered(x): return x
@torch.library.custom_op("ns::missing", mutates_args=())
def missing(x): return x
@torch.library.register_fake("ns::registered")
def registered_fake(x): return x

op = torch.library.custom_op("ns::assigned", fn, mutates_args=())
op.register_fake(fake)

@torch.library.custom_op("ns::object_argument", mutates_args=())
def object_argument(x): return x
@torch.library.register_fake(object_argument)
def object_argument_fake(x): return x

@tl.custom_op("ns::aliased_ok", mutates_args=())
def aliased_ok(x): return x
@tl.register_fake("ns::aliased_ok")
def aliased_ok_fake(x): return x

@reg_op("ns::aliased_missing", mutates_args=())
def aliased_missing(x): return x

def scope_a():
    @tl.custom_op("ns::scoped_ok", mutates_args=())
    def local_op(x): return x
    @local_op.register_fake
    def local_op_fake(x): return x

def scope_b():
    @tl.custom_op("ns::scoped_missing", mutates_args=())
    def local_op(x): return x
''')
    result = _run(["--json", "--patterns", "custom_op_missing_fake", path])
    payload = json.loads(result.stdout)
    lines = [finding["snippet"] for finding in payload["findings"]]
    assert lines == [
        '@torch.library.custom_op("ns::missing", mutates_args=())',
        '@reg_op("ns::aliased_missing", mutates_args=())',
        '@tl.custom_op("ns::scoped_missing", mutates_args=())',
    ], lines


def test_api_boundaries(directory):
    path = os.path.join(directory, "api_boundaries.py")
    with open(path, "w") as sample:
        sample.write('''
from other_pkg import AdamW as ForeignAdamW
from torch.optim import AdamW as TorchAdamW

a = image.create_mask(x)
b = metric.normalize(x + residual)
c = torch.view_as_real(fft_x)
d = torch.view_as_real(y * freqs)
e = torch.optim.AdamW(params, **{"fused": True})
e_foreign = ForeignAdamW(params)
e_torch = TorchAdamW(params)
f = flash_attn_varlen_with_mask_func(q, k, v, cu_q, cu_k)
g = F.interpolate(x, None, 2, "bilinear")
h = F.multi_head_attention_forward(q, k, v, d, h, w, b, bk, bv,
                                   False, 0, ow, ob, True, None, False)
i = nn.MultiheadAttention(32, 4)
i_bad_norm = cnn.LayerNorm(64)
j = self.layernorm1(x + residual)
k = self.rmsnorm(x + residual)
m = self.norm1(x) + residual
''')
    result = _run(["--json", path])
    payload = json.loads(result.stdout)
    findings = payload["findings"]
    assert {item["line"]: item["pattern"] for item in findings} == {
        8: "complex_rope",
        11: "adamw_not_fused",
        12: "fa2_varlen_call",
        13: "upsample_bilinear",
        16: "mha_default_avg_weights",
        18: "add_then_norm",
        19: "add_then_norm",
    }, findings
    mha = next(item for item in findings if item["pattern"] == "mha_default_avg_weights")
    assert "call site" in mha["note"], mha


def test_scan_status(directory):
    valid = os.path.join(directory, "valid.py")
    invalid = os.path.join(directory, "broken.py")
    with open(valid, "w") as sample:
        sample.write("x = 1\n")
    with open(invalid, "w") as sample:
        sample.write("def invalid(\n")

    result = _run(["--json", valid])
    payload = json.loads(result.stdout)
    assert payload["findings"] == [], payload
    assert payload["files_scanned"] == 1 and payload["files_skipped"] == 0, payload
    assert payload["skipped_files"] == [], payload
    assert "STATIC candidates only" in payload["disclaimer"], payload
    assert "STATIC candidates only" in result.stderr, result.stderr

    result = _run(["--json", valid, invalid], expect_ok=False)
    payload = json.loads(result.stdout)
    assert payload["findings"] == [], payload
    assert payload["files_scanned"] == 1 and payload["files_skipped"] == 1, payload
    assert len(payload["skipped_files"]) == 1 and payload["skipped_files"][0]["file"] == invalid, payload
    assert "STATIC candidates only" in payload["disclaimer"], payload
    assert "1 scanned" in result.stderr and "1 skipped" in result.stderr, result.stderr
    assert "STATIC candidates only" in result.stderr, result.stderr

    result = _run([invalid], expect_ok=False)
    assert "No Python files scanned" in result.stdout, result.stdout

    empty = os.path.join(directory, "empty")
    os.mkdir(empty)
    result = _run([empty], expect_ok=False)
    assert "No Python files scanned" in result.stdout, result.stdout
    _run([os.path.join(directory, "missing.py")], expect_ok=False)


def test_source_encoding_and_parse_error(directory):
    encoded = os.path.join(directory, "encoded.py")
    malformed = os.path.join(directory, "nul.py")
    with open(encoded, "wb") as sample:
        sample.write(b"# coding: cp1252\ncaf\xe9 = torch.optim.AdamW(params)\n")
    with open(malformed, "wb") as sample:
        sample.write(b"x = 1\x00\n")

    result = _run(["--json", encoded, malformed], expect_ok=False)
    assert "1 scanned" in result.stderr and "1 skipped" in result.stderr, result.stderr
    payload = json.loads(result.stdout)
    assert payload["files_scanned"] == 1 and payload["files_skipped"] == 1, payload
    findings = payload["findings"]
    assert [item["snippet"] for item in findings] == ["caf\u00e9 = torch.optim.AdamW(params)"], findings


def main():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(SYNTHETIC_SAMPLE)
        sample_path = tf.name

    try:
        # 1. JSON mode: all 13 patterns + false-positive boundaries + envelope metadata
        res_json = _run(["--json", sample_path])
        payload = json.loads(res_json.stdout)
        assert "STATIC candidates only" in payload["disclaimer"], payload
        assert payload["files_scanned"] == 1 and payload["files_skipped"] == 0, payload
        findings = payload["findings"]
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

        with tempfile.TemporaryDirectory() as directory:
            test_fake_registration(directory)
            test_api_boundaries(directory)
            test_scan_status(directory)
            test_source_encoding_and_parse_error(directory)

        print("selftest PASSED: {} detectors plus alias/scoped fake registration, API boundaries, JSON envelope and CLI status.".format(
            len(EXPECTED_COUNTS)
        ))
    finally:
        os.unlink(sample_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
