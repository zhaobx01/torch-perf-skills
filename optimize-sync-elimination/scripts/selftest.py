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
   - Control-flow soundness: with-blocks preserve bindings, if-conditions use
     pre-branch facts, and for-loop iterables are not marked in-loop.
   - Directory exclusions (.venv), JSON envelope metadata, partial/empty scan
     non-zero exit codes, and CLI flags (--json, --min-severity).
2. analyze_trace_sync.py:
   - Synthetic Chrome trace JSON and .json.gz parsing.
   - Exact category counts, total_dur_us sums, duration sorting, stack
     extraction, metadata ("M") event filtering, --list-names / --json,
     hipMemcpyWithStream / synchronous cudaMemcpy/hipMemcpy, and corrupt .gz
     error handling without traceback.

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


def _invoke(cmd):
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def _run(cmd):
    proc = _invoke(cmd)
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
    payload = json.loads(out)
    assert "STATIC candidates only" in payload["disclaimer"], payload
    assert payload["files_scanned"] == 1 and payload["files_skipped"] == 0, payload
    assert payload["skipped_files"] == [], payload
    findings = payload["findings"]

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
    high_findings = json.loads(out_high)["findings"]
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


def _scan_source(tmpdir, source):
    path = os.path.join(tmpdir, "candidate.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    out, _ = _run([sys.executable, SCAN_SCRIPT, "--json", path])
    return json.loads(out)["findings"]


def test_scanner_requires_mask_evidence(tmpdir):
    rows = _scan_source(tmpdir, """\
def step(x, ranks: torch.Tensor, indices, condition):
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = ranks[1:] != ranks[:-1]
    selected = x[kept]
    x[kept] -= 1
    x[:, ranks[1:] != ranks[:-1]] = 0
    assert x[indices].shape
    if condition:
        kept = indices
    selected = x[kept]
def other(x, kept):
    return x[kept]
""")
    hits = [row["line"] for row in rows if row["pattern"] == "boolean indexing"]
    assert hits == [4, 5, 6], hits


def test_scanner_respects_device_and_size_arguments(tmpdir):
    rows = _scan_source(tmpdir, """\
def step(x, repeats, condition):
    result = x.cpu().item()
    host = x.to(device=torch.device('cpu:0'))
    result = host.reshape(-1)[0].item()
    result = torch.zeros((), device='cpu').item()
    result = x.to('cpu', non_blocking=True).tolist()
    invalid = x.to('cputypo')
    known = torch.repeat_interleave(x, repeats, output_size=6)
    unknown = x.repeat_interleave(repeats, output_size=None)
    hist = x.bincount(minlength=3)
    if condition:
        host = x
    result = host.item()
    fixed_method = x.repeat_interleave(2)
    fixed_function = torch.repeat_interleave(x, repeats=2)
    repeats_only = torch.repeat_interleave(repeats)
""")
    hits = {(row["line"], row["pattern"]): row for row in rows}
    assert [hits[n, ".item()"]["severity"] for n in (2, 4, 5, 13)] == ["info", "info", "info", "high"]
    assert hits[3, '.to("cpu")']["severity"] == "high"
    assert "wait" in hits[6, ".tolist()"]["note"].lower()
    assert {row["line"] for row in rows if row["pattern"] == '.to("cpu")'} == {3, 6}
    assert {line for line, pattern in hits if pattern == ".repeat_interleave()"} == {9, 16}
    assert "lower bound" in hits[10, ".bincount()"]["note"]


def test_scanner_control_flow_and_with_blocks(tmpdir):
    rows = _scan_source(tmpdir, """\
import torch
def step(x, y):
    host = x.cpu()
    if host.item() > 0:
        host = y
    post_if = host.item()
    with torch.no_grad():
        m = torch.ones(4, dtype=torch.bool)
    z = x[m]
    for i in x.tolist():
        v = y.item()
    return post_if, z, v
""")
    hits = {(row["line"], row["pattern"]): row for row in rows}
    assert hits[4, ".item()"]["severity"] == "info", hits[4, ".item()"]
    assert hits[6, ".item()"]["severity"] == "high", hits[6, ".item()"]
    assert (9, "boolean indexing") in hits, hits
    assert hits[10, ".tolist()"]["in_loop"] is False, hits[10, ".tolist()"]
    assert hits[11, ".item()"]["in_loop"] is True, hits[11, ".item()"]


def test_scanner_status_skip_dirs_and_invalid_paths(tmpdir):
    missing = os.path.join(tmpdir, "missing.py")
    result = _invoke([sys.executable, SCAN_SCRIPT, "--json", missing])
    assert result.returncode == 2 and missing in result.stderr and not result.stdout, result
    assert _scan_source(tmpdir, "x = 1\n") == []

    # Empty directory -> exit 1
    empty_dir = os.path.join(tmpdir, "empty_dir")
    os.mkdir(empty_dir)
    result = _invoke([sys.executable, SCAN_SCRIPT, "--json", empty_dir])
    assert result.returncode == 1, result
    payload = json.loads(result.stdout)
    assert payload["files_scanned"] == 0 and payload["files_skipped"] == 0, payload

    # Partial scan (valid + syntax error) -> exit 1 with skipped_files populated
    proj = os.path.join(tmpdir, "proj")
    venv_dir = os.path.join(proj, ".venv")
    os.makedirs(venv_dir)
    valid = os.path.join(proj, "valid.py")
    broken = os.path.join(proj, "broken.py")
    venv_file = os.path.join(venv_dir, "ignored.py")
    with open(valid, "w", encoding="utf-8") as handle:
        handle.write("a = x.item()\n")
    with open(broken, "w", encoding="utf-8") as handle:
        handle.write("def bad(\n")
    with open(venv_file, "w", encoding="utf-8") as handle:
        handle.write("b = y.item()\n")

    result = _invoke([sys.executable, SCAN_SCRIPT, "--json", proj])
    assert result.returncode == 1, result
    payload = json.loads(result.stdout)
    assert payload["files_scanned"] == 1 and payload["files_skipped"] == 1, payload
    assert payload["skipped_files"][0]["file"] == broken, payload
    assert [f["file"] for f in payload["findings"]] == [valid], payload


def test_scanner_recognizes_explicit_mask_producers(tmpdir):
    rows = _scan_source(tmpdir, """\
def step(x, flags, repeats, size=None):
    mask = torch.ones_like(x, dtype=torch.bool)
    selected = x[mask & torch.isfinite(x)]
    x[~mask] = 0
    selected = x[torch.logical_and(mask, flags)]
    selected = x[flags & flags]
    result = x.repeat_interleave(repeats, output_size=size)
""")
    assert [row["line"] for row in rows if row["pattern"] == "boolean indexing"] == [3, 4, 5]
    assert [row["line"] for row in rows if row["pattern"] == ".repeat_interleave()"] == [7]


def test_trace_boundaries_and_directions(tmpdir):
    path = os.path.join(tmpdir, "directions.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([
            {"name": "Memcpy DtoH (Device -> Pinned)", "ph": "X", "dur": 20},
            {"name": "Memcpy H2D", "dur": 10},
            {"name": "hipMemcpyAsync", "ph": "X", "dur": 4},
            {"name": "hipMemcpyWithStream", "ph": "X", "dur": 5},
            {"name": "cudaMemcpy", "ph": "X", "dur": 12},
            {"name": "hipMemcpy", "ph": "X", "dur": 8},
            {"name": "aten::item", "ph": "X", "dur": 100, "ts": 0},
            {"name": "aten::_local_scalar_dense", "ph": "X", "dur": 80, "ts": 10},
            {"name": "aten::item", "ph": "X"},
            {"name": "aten::item", "ph": "X", "dur": True},
            {"name": "aten::item", "ph": "X", "dur": -1},
            {"name": "aten::item", "ph": "X", "dur": float("nan")},
            {"name": "aten::itemized_custom", "ph": "X", "dur": 9},
            {"name": "my::aten::item", "ph": "X", "dur": 11},
            {"name": "aten::nonzero_static", "ph": "X", "dur": 7},
            {"name": "unrelated aten::nonzero called", "ph": "X", "dur": 12},
            {"name": "aten::unique_dim_consecutive", "ph": "X", "dur": 6},
            {"name": "aten::index_put_", "ph": "X", "dur": 8},
            {"name": "aten::index_select.out", "ph": "X", "dur": 3},
        ], handle)
    out, err = _run([sys.executable, TRACE_SCRIPT, "--json", path])
    rows = {row["label"]: row for row in json.loads(out)}
    assert rows["item_or_local_scalar"]["count"] == 2, rows
    assert rows["item_or_local_scalar"]["total_dur_us"] == 180
    assert rows["nonzero_or_dynamic_shape"]["sample_names"] == ["aten::unique_dim_consecutive"]
    assert rows["index_ops"]["sample_names"] == ["aten::index_put_", "aten::index_select.out"]
    copy_row = rows["memcpy_d2h_or_h2d"]
    assert (copy_row["count"], copy_row["total_dur_us"]) == (2, 30)
    assert copy_row["direction_breakdown"] == {
        "D2H": {"count": 1, "total_dur_us": 20},
        "H2D": {"count": 1, "total_dur_us": 10},
    }
    assert rows["memcpy_async_api"]["count"] == 2 and rows["memcpy_async_api"]["total_dur_us"] == 9
    assert rows["memcpy_sync_api"]["count"] == 2 and rows["memcpy_sync_api"]["total_dur_us"] == 20
    assert "overlap" in err
    names, _ = _run([sys.executable, TRACE_SCRIPT, "--json", "--list-names", path])
    item = next(row for row in json.loads(names) if row["name"] == "aten::item")
    assert (item["count"], item["total_dur_us"]) == (1, 100)
    summary, _ = _run([sys.executable, TRACE_SCRIPT, path])
    assert all(label in summary for label in ("D2H", "H2D", "direction unknown", "memcpy_sync_api")), summary


def test_trace_rejects_invalid_shape_and_corrupt_gz(tmpdir):
    path = os.path.join(tmpdir, "bad.json")
    for payload in ({"traceEvents": None}, {"traceEvents": {}}, {"unrelated": 0}, 17):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        result = _invoke([sys.executable, TRACE_SCRIPT, "--json", path])
        assert result.returncode != 0 and "traceEvents" in result.stderr, result
        assert "Traceback" not in result.stderr and not result.stdout, result

    corrupt_gz = os.path.join(tmpdir, "corrupt.json.gz")
    with open(corrupt_gz, "wb") as handle:
        handle.write(b"\x1f\x8b\x08\x00corrupt_zlib_stream_data_1234567890")
    result = _invoke([sys.executable, TRACE_SCRIPT, "--json", corrupt_gz])
    assert result.returncode != 0 and "could not read trace" in result.stderr, result
    assert "Traceback" not in result.stderr and not result.stdout, result

    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"traceEvents": []}, handle)
    out, _ = _run([sys.executable, TRACE_SCRIPT, "--json", path])
    assert json.loads(out) == []


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_static_scanner(tmpdir)
        test_trace_analyzer(tmpdir)
        test_scanner_requires_mask_evidence(tmpdir)
        test_scanner_respects_device_and_size_arguments(tmpdir)
        test_scanner_control_flow_and_with_blocks(tmpdir)
        test_scanner_status_skip_dirs_and_invalid_paths(tmpdir)
        test_scanner_recognizes_explicit_mask_producers(tmpdir)
        test_trace_boundaries_and_directions(tmpdir)
        test_trace_rejects_invalid_shape_and_corrupt_gz(tmpdir)
    print("PASS: scan_sync_points.py and analyze_trace_sync.py self-tests succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
