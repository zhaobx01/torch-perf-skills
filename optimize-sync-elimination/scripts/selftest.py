#!/usr/bin/env python3
"""Self-test for optimize-sync-elimination scripts.

Verifies:
1. scan_sync_points.py:
   - True-positive detection for .item(), .cpu(), .to("cpu"), .unique(),
     torch.argwhere(), dist.barrier(), single-arg torch.where(cond), and
     in-loop tagging.
   - False-positive boundaries: 3-arg torch.where(cond, a, b) is ignored;
     chained .to("cpu").tolist() / .cpu().numpy() downgrades to info;
     literal int(42) is ignored.
   - CLI flags (--json, --min-severity) and stderr static disclaimer.
2. analyze_trace_sync.py:
   - Synthetic Chrome trace JSON and .json.gz parsing.
   - Exact category counts, total_dur_us sums, duration sorting, stack
     extraction, metadata ("M") event filtering, and --list-names / --json.

Usage:
    python3 scripts/selftest.py
"""

import gzip
import json
import os
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_SCRIPT = os.path.join(SCRIPT_DIR, "scan_sync_points.py")
TRACE_SCRIPT = os.path.join(SCRIPT_DIR, "analyze_trace_sync.py")


def _run(cmd):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    assert proc.returncode == 0, "Command failed (%s): %s\nSTDERR:\n%s" % (
        proc.returncode, cmd, proc.stderr
    )
    return proc.stdout, proc.stderr


def test_static_scanner(tmpdir):
    sample_py = os.path.join(tmpdir, "sample_train.py")
    with open(sample_py, "w", encoding="utf-8") as handle:
        handle.write(
            "from __future__ import annotations\n"
            "import torch\n"
            "import torch.distributed as dist\n"
            "\n"
            "def train_step(x, mask, losses):\n"
            "    # True positives\n"
            "    scalar = x.item()\n"
            "    idx1 = torch.where(mask)\n"
            "    idx2 = torch.argwhere(mask)\n"
            "    uniq = idx1[0].unique()\n"
            "    for loss in losses:\n"
            "        _ = loss.item()\n"
            "    dist.barrier()\n"
            "    torch.cuda.synchronize()\n"
            "\n"
            "    # False-positive boundaries\n"
            "    selected = torch.where(mask, x, x)  # 3-arg elementwise, NOT sync\n"
            "    rows = x.detach().to('cpu').tolist()  # .to('cpu') is high; chained .tolist() is info\n"
            "    arr = x.cpu().numpy()  # .cpu() is high; chained .numpy() is info\n"
            "    lit = int(42)  # literal cast, NOT sync\n"
            "    return scalar, idx2, uniq, selected, rows, arr, lit\n"
        )

    out, err = _run([sys.executable, SCAN_SCRIPT, "--json", sample_py])
    assert "STATIC candidates only" in err, "Missing static disclaimer in stderr"
    findings = json.loads(out)

    by_pattern = {}
    for item in findings:
        by_pattern.setdefault(item["pattern"], []).append(item)

    # Check true positives
    assert ".item()" in by_pattern and len(by_pattern[".item()"]) == 2, by_pattern
    in_loop_items = [f for f in by_pattern[".item()"] if f["in_loop"]]
    assert len(in_loop_items) == 1, "Expected exactly 1 in-loop .item()"
    assert "torch.where(cond)" in by_pattern, "Expected single-arg torch.where(cond)"
    assert "torch.argwhere" in by_pattern, "Expected torch.argwhere"
    assert ".unique()" in by_pattern, "Expected .unique()"
    assert "dist.barrier" in by_pattern, "Expected dist.barrier"
    assert "torch.cuda.synchronize" in by_pattern, "Expected torch.cuda.synchronize"
    assert '.to("cpu")' in by_pattern and by_pattern['.to("cpu")'][0]["severity"] == "high"
    assert ".cpu()" in by_pattern and by_pattern[".cpu()"][0]["severity"] == "high"

    # Check false-positive boundaries
    assert len(by_pattern["torch.where(cond)"]) == 1, "3-arg torch.where must not fire"
    assert by_pattern[".tolist()"][0]["severity"] == "info", "Chained .to('cpu').tolist() must downgrade to info"
    assert by_pattern[".numpy()"][0]["severity"] == "info", "Chained .cpu().numpy() must downgrade to info"
    assert "int(...)" not in by_pattern, "Literal int(42) must not fire"

    # Check --min-severity high filters out medium/info
    out_high, _ = _run([sys.executable, SCAN_SCRIPT, "--json", "--min-severity", "high", sample_py])
    high_findings = json.loads(out_high)
    assert all(f["severity"] == "high" for f in high_findings), high_findings
    assert not any(f["pattern"] in (".tolist()", ".numpy()", ".unique()") for f in high_findings)


def test_trace_analyzer(tmpdir):
    trace_data = {
        "traceEvents": [
            {"name": "process_name", "ph": "M", "dur": 99999},
            {"name": "aten::item", "ph": "X", "dur": 400.0, "args": {"Call stack": "train.py(12): step"}},
            {"name": "aten::_local_scalar_dense", "ph": "X", "dur": 100.0, "args": {"Call stack": "train.py(12): step"}},
            {"name": "hipStreamSynchronize", "ph": "X", "dur": 1200.5},
            {"name": "cudaDeviceSynchronize", "ph": "X", "dur": 299.5},
            {"name": "Memcpy DtoH (Device -> Pinned)", "ph": "X", "dur": 250.0},
            {"name": "aten::_unique2", "ph": "X", "dur": 80.0},
            {"name": "aten::nonzero", "ph": "X", "dur": 20.0},
            {"name": "aten::mm", "ph": "X", "dur": 5000.0},
        ]
    }
    json_path = os.path.join(tmpdir, "trace.json")
    gz_path = os.path.join(tmpdir, "trace.json.gz")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(trace_data, handle)
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        json.dump(trace_data, handle)

    for path in (json_path, gz_path):
        out, err = _run([sys.executable, TRACE_SCRIPT, "--json", path])
        assert "trace-bottleneck-report" in err, "Missing GPU starvation note in stderr"
        rows = json.loads(out)
        labels = [r["label"] for r in rows]
        assert labels == [
            "explicit_synchronize",
            "item_or_local_scalar",
            "memcpy_d2h_or_h2d",
            "nonzero_or_dynamic_shape",
        ], labels

        by_label = {r["label"]: r for r in rows}
        assert by_label["explicit_synchronize"]["count"] == 2
        assert abs(by_label["explicit_synchronize"]["total_dur_us"] - 1500.0) < 1e-6
        assert by_label["item_or_local_scalar"]["count"] == 2
        assert abs(by_label["item_or_local_scalar"]["total_dur_us"] - 500.0) < 1e-6
        assert by_label["item_or_local_scalar"]["sample_stacks"] == ["train.py(12): step"]
        assert by_label["memcpy_d2h_or_h2d"]["count"] == 1
        assert abs(by_label["memcpy_d2h_or_h2d"]["total_dur_us"] - 250.0) < 1e-6
        assert by_label["nonzero_or_dynamic_shape"]["count"] == 2
        assert abs(by_label["nonzero_or_dynamic_shape"]["total_dur_us"] - 100.0) < 1e-6

    # Check --list-names skips metadata "M" events and ranks aten::mm first
    out_names, _ = _run([sys.executable, TRACE_SCRIPT, "--list-names", "--json", json_path])
    name_rows = json.loads(out_names)
    names = [r["name"] for r in name_rows]
    assert "process_name" not in names, "Metadata 'M' event must be excluded from --list-names"
    assert names[0] == "aten::mm" and abs(name_rows[0]["total_dur_us"] - 5000.0) < 1e-6


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_static_scanner(tmpdir)
        test_trace_analyzer(tmpdir)
    print("PASS: scan_sync_points.py and analyze_trace_sync.py self-tests succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
