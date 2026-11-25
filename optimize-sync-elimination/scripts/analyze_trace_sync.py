#!/usr/bin/env python3
"""Find host-device synchronization evidence in an already-captured trace.

This script only PARSES an existing Chrome-trace JSON file produced by
``torch.profiler`` (``prof.export_chrome_trace(path)``). It does not run
your model, does not attach a profiler, and does not capture anything.

Note on scope:
- It matches event names and sums their CPU/event durations; it does NOT
  cross-check whether the GPU track was actually idle during each call.
  When you have full multi-rank traces, prefer ``trace-bottleneck-report``
  (§4 GPU starvation attribution) to judge real sync cost.
- HIP/DTK event naming varies by build and is NOT verified across all DTK
  releases -- run ``--list-names`` on your trace first to inspect actual
  event names.

Usage:
    python3 analyze_trace_sync.py trace.json
    python3 analyze_trace_sync.py trace.json.gz --list-names
    python3 analyze_trace_sync.py trace.json --json

Only the Python standard library is used (compatible with Python 3.6+).
"""

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# (label, compiled regex). Matched against the raw event "name" field.
_SYNC_PATTERNS = [
    ("explicit_synchronize", re.compile(r"(cuda|hip)(Device|Stream|Event)?Synchronize", re.IGNORECASE)),
    ("memcpy_d2h_or_h2d", re.compile(r"Memcpy\s*(DtoH|HtoD|D2H|H2D)", re.IGNORECASE)),
    ("memcpy_async_api", re.compile(r"(cuda|hip)MemcpyAsync", re.IGNORECASE)),
    ("item_or_local_scalar", re.compile(r"aten::(item|_local_scalar_dense)")),
    ("nonzero_or_dynamic_shape", re.compile(r"aten::(nonzero|argwhere|masked_select|_unique2|unique_dim|unique_consecutive)")),
    ("index_ops", re.compile(r"aten::(index|index_select|index_put)\b")),
]


class Aggregate(object):
    def __init__(self, label: str) -> None:
        self.label = label
        self.count = 0
        self.total_dur_us = 0.0
        self.sample_names: Set[str] = set()
        self.sample_stacks: List[str] = []

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "count": self.count,
            "total_dur_us": self.total_dur_us,
            "sample_names": sorted(self.sample_names),
            "sample_stacks": self.sample_stacks,
        }


def _load_events(path: str) -> List[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        events = data.get("traceEvents", [])
    elif isinstance(data, list):
        events = data
    else:
        events = []
    return [e for e in events if isinstance(e, dict)]


def _extract_stack(event: dict) -> Optional[str]:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    for key in ("Call stack", "stack", "Python stack", "python_stack"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def analyze(events: List[dict]) -> Dict[str, Aggregate]:
    aggregates: Dict[str, Aggregate] = {}
    for event in events:
        name = event.get("name")
        if not isinstance(name, str):
            continue
        if event.get("ph") not in (None, "X"):
            # "X" = complete event with duration; some traces omit ph.
            continue
        for label, pattern in _SYNC_PATTERNS:
            if pattern.search(name):
                agg = aggregates.setdefault(label, Aggregate(label=label))
                agg.count += 1
                dur = event.get("dur")
                if isinstance(dur, (int, float)):
                    agg.total_dur_us += float(dur)
                agg.sample_names.add(name)
                stack = _extract_stack(event)
                if stack and stack not in agg.sample_stacks and len(agg.sample_stacks) < 5:
                    agg.sample_stacks.append(stack)
                break
    return aggregates


def list_all_names(events: List[dict]) -> List[Tuple[str, int, float]]:
    totals: Dict[str, List[float]] = defaultdict(list)
    for event in events:
        name = event.get("name")
        if not isinstance(name, str):
            continue
        if event.get("ph") not in (None, "X"):
            continue
        dur = event.get("dur")
        totals[name].append(float(dur) if isinstance(dur, (int, float)) else 0.0)
    rows = [(name, len(durs), sum(durs)) for name, durs in totals.items()]
    rows.sort(key=lambda row: (-row[2], row[0]))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", help="Path to a Chrome-trace JSON file (optionally .gz).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report.")
    parser.add_argument(
        "--list-names",
        action="store_true",
        help="Dump every distinct duration event name with count and total duration, sorted by duration. "
        "Use this first on a real DTK/HIP trace to verify actual event names.",
    )
    args = parser.parse_args(argv)

    try:
        events = _load_events(args.trace)
    except (OSError, ValueError) as exc:
        print("error: could not read trace: %s" % exc, file=sys.stderr)
        return 1

    if args.list_names:
        rows = list_all_names(events)
        if args.json:
            print(json.dumps(
                [{"name": n, "count": c, "total_dur_us": d} for n, c, d in rows],
                ensure_ascii=False, indent=2,
            ))
        else:
            print("%d distinct event name(s), sorted by total duration (us):" % len(rows))
            for name, count, dur in rows[:200]:
                print("  %12.1f us  x%-6d  %s" % (dur, count, name))
        return 0

    aggregates = analyze(events)
    if not aggregates:
        if args.json:
            print("[]")
        print(
            "No events matched the known CUDA/HIP sync patterns. This may mean there are "
            "none in this trace, or that this trace uses different event names (common on "
            "HIP/DTK). Run with --list-names to see what names actually appear.",
            file=sys.stderr,
        )
        return 0

    ordered = sorted(aggregates.values(), key=lambda a: (-a.total_dur_us, a.label))
    if args.json:
        print(json.dumps([a.to_dict() for a in ordered], ensure_ascii=False, indent=2))
    else:
        print("Sync-related events found (sorted by total duration):\n")
        for agg in ordered:
            print("[%s] count=%d total_dur=%.1fus" % (agg.label, agg.count, agg.total_dur_us))
            print("    matched names: %s" % ", ".join(sorted(agg.sample_names)))
            for stack in agg.sample_stacks:
                first_line = stack.splitlines()[0] if stack else ""
                print("    stack (first line): %s" % first_line)
            print()

    print(
        "Note: this matches event NAMES and sums event durations without checking whether the GPU "
        "track was idle. Use trace-bottleneck-report for GPU starvation attribution, and run "
        "--list-names on unfamiliar HIP/DTK traces to verify actual event names.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
