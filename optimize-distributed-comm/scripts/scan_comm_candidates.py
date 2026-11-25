#!/usr/bin/env python3
"""Static scanner for distributed communication candidates in PyTorch projects.

This is a STATIC scanner. It scans Python source files (AST) and DeepSpeed
JSON/YAML config files for conservative DDP/FSDP/ZeRO-2 settings, explicit
collective calls (including loop-internal sync collectives and custom-stream
RCCL ordering risks), and metadata gather_object calls.

It does NOT prove that communication is a bottleneck, distinguish true
exposed communication from slow-rank skew, or guarantee that enabling a
flag (such as static_graph=True or reshard_after_forward=False) is safe for
the target model's control flow or memory budget. Always confirm with
trace-bottleneck-report (T5) before changing code.

Usage:
    python3 scan_comm_candidates.py <file-or-dir> [<file-or-dir> ...]
    python3 scan_comm_candidates.py --json <file-or-dir> ...
    python3 scan_comm_candidates.py --min-severity medium <file-or-dir> ...
    python3 scan_comm_candidates.py --no-configs <file-or-dir> ...

Only the Python standard library is used (compatible with Python 3.6+).
"""

import argparse
import ast
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple


_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules", ".venv", "venv"}
_MAX_CONFIG_BYTES = 2 * 1024 * 1024

_DDP_NAMES = {"DistributedDataParallel", "MMDistributedDataParallel", "DDP"}
_DDP_KWARGS_NAMES = {"DistributedDataParallelKwargs"}
_FSDP1_NAMES = {"FullyShardedDataParallel", "FSDP"}
_COLLECTIVE_NAMES = {
    "all_gather",
    "all_gather_into_tensor",
    "_all_gather_base",
    "all_reduce",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "_reduce_scatter_base",
    "all_to_all",
    "all_to_all_single",
    "broadcast",
    "reduce",
}
_OBJECT_COLLECTIVE_NAMES = {"gather_object", "all_gather_object", "broadcast_object_list"}


class Finding(object):
    def __init__(
        self,
        file: str,
        line: int,
        col: int,
        pattern: str,
        severity: str,
        snippet: str,
        note: str,
    ) -> None:
        self.file = file
        self.line = line
        self.col = col
        self.pattern = pattern
        self.severity = severity
        self.snippet = snippet
        self.note = note

    def to_dict(self) -> Dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "col": self.col,
            "pattern": self.pattern,
            "severity": self.severity,
            "snippet": self.snippet,
            "note": self.note,
        }


def _dotted_name(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _call_short_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _kwarg_map(node: ast.Call) -> Dict[str, ast.AST]:
    return {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}


def _const_bool(node: Optional[ast.AST]) -> Optional[bool]:
    if node is None:
        return None
    if isinstance(node, ast.NameConstant) and isinstance(node.value, bool):
        return node.value
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _is_stream_context(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Call):
        dotted = _dotted_name(expr.func)
        short = _call_short_name(expr)
        if short == "stream" and ("cuda" in dotted or "stream" in dotted):
            return True
    return False


class _PyVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, source_lines: List[str]) -> None:
        self.filename = filename
        self.source_lines = source_lines
        self.findings: List[Finding] = []
        self._loop_depth = 0
        self._stream_depth = 0

    def _snippet(self, node: ast.AST) -> str:
        line_no = getattr(node, "lineno", None)
        if line_no is None or line_no > len(self.source_lines):
            return ""
        return self.source_lines[line_no - 1].strip()

    def _emit(self, node: ast.AST, pattern: str, severity: str, note: str) -> None:
        self.findings.append(
            Finding(
                file=self.filename,
                line=getattr(node, "lineno", -1),
                col=getattr(node, "col_offset", -1),
                pattern=pattern,
                severity=severity,
                snippet=self._snippet(node),
                note=note,
            )
        )

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_With(self, node: ast.With) -> None:
        entered = any(_is_stream_context(item.context_expr) for item in node.items)
        if entered:
            self._stream_depth += 1
        self.generic_visit(node)
        if entered:
            self._stream_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        short = _call_short_name(node)
        dotted = _dotted_name(node.func)
        kwargs = _kwarg_map(node)

        # 1. DDP constructor / Accelerate DistributedDataParallelKwargs
        if short in _DDP_NAMES or short in _DDP_KWARGS_NAMES:
            find_unused = _const_bool(kwargs.get("find_unused_parameters"))
            if find_unused is True:
                self._emit(
                    node,
                    "ddp_find_unused_true",
                    "high",
                    "DDP find_unused_parameters=True traverses the autograd graph every step; verify if all trainable params participate and consider False (C1).",
                )
            if short in _DDP_NAMES:
                missing = []
                if "static_graph" not in kwargs:
                    missing.append("static_graph")
                if "gradient_as_bucket_view" not in kwargs:
                    missing.append("gradient_as_bucket_view")
                if missing:
                    self._emit(
                        node,
                        "ddp_conservative_defaults",
                        "medium",
                        "DDP constructor omits %s; if graph topology is static, evaluate static_graph=True and gradient_as_bucket_view=True (C1)." % ", ".join(missing),
                    )

        # 2. FSDP1 constructor
        elif short in _FSDP1_NAMES and (node.args or node.keywords):
            limit_ag = _const_bool(kwargs.get("limit_all_gathers"))
            if limit_ag is True:
                self._emit(
                    node,
                    "fsdp1_limit_all_gathers_true",
                    "high",
                    "FSDP1 limit_all_gathers=True throttles CPU AllGather launch; if memory headroom allows, evaluate False with prefetch (C2).",
                )
            missing_pf = [k for k in ("forward_prefetch", "backward_prefetch") if k not in kwargs]
            if missing_pf:
                self._emit(
                    node,
                    "fsdp1_missing_prefetch",
                    "medium",
                    "FSDP1 constructor omits %s; evaluate enabling prefetch when layer order is fixed and memory permits (C2)." % ", ".join(missing_pf),
                )

        # 3. FSDP2 fully_shard(...)
        elif short == "fully_shard" and (node.args or node.keywords):
            reshard = _const_bool(kwargs.get("reshard_after_forward"))
            if reshard is True:
                self._emit(
                    node,
                    "fsdp2_reshard_after_forward_true",
                    "medium",
                    "FSDP2 reshard_after_forward=True re-gathers weights in backward; if memory headroom permits, evaluate False (C3).",
                )
            elif "reshard_after_forward" not in kwargs:
                has_star_kwargs = any(kw.arg is None for kw in node.keywords)
                if not has_star_kwargs:
                    self._emit(
                        node,
                        "fsdp2_implicit_reshard",
                        "info",
                        "FSDP2 fully_shard() relies on default reshard_after_forward=True; verify sharding granularity and residency trade-off (C3).",
                    )

        # 4. Explicit torch.distributed collectives
        elif isinstance(node.func, ast.Attribute) and short in _COLLECTIVE_NAMES and (
            dotted.startswith("dist.") or dotted.startswith("torch.distributed.") or ".distributed." in dotted
        ):
            async_flag = _const_bool(kwargs.get("async_op"))
            if self._stream_depth > 0:
                self._emit(
                    node,
                    "collective_in_custom_stream",
                    "high",
                    "Collective %s invoked inside a custom CUDA/HIP stream; on RCCL verify canonical per-group ordering to prevent deadlock (R1) and call record_stream()." % dotted,
                )
            elif self._loop_depth > 0 and async_flag is not True:
                self._emit(
                    node,
                    "collective_in_loop_sync",
                    "medium",
                    "Synchronous collective %s inside a loop (async_op!=True); check if collectives can be coalesced or replaced by owner-sharded all_to_all (C5)." % dotted,
                )
            else:
                self._emit(
                    node,
                    "collective_call",
                    "info",
                    "Explicit collective %s; inspect overlap and message size in trace-bottleneck-report T5." % dotted,
                )

        # 5. Distributed object collectives (RCCL gather_object caution)
        elif isinstance(node.func, ast.Attribute) and short in _OBJECT_COLLECTIVE_NAMES and (
            dotted.startswith("dist.") or dotted.startswith("torch.distributed.") or ".distributed." in dotted
        ):
            self._emit(
                node,
                "dist_gather_object",
                "info",
                "Object collective %s; on HIP/RCCL verify stability and consider a dedicated CPU/Gloo group for checkpoint metadata (R2)." % dotted,
            )

        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, val in zip(node.keys, node.values):
            if key is None:
                continue
            kstr = None
            if isinstance(key, ast.Str):
                kstr = key.s
            elif hasattr(ast, "Constant") and isinstance(key, ast.Constant) and isinstance(key.value, str):
                kstr = key.value
            if kstr == "reshard_after_forward" and _const_bool(val) is True:
                self._emit(
                    val,
                    "fsdp2_reshard_after_forward_true",
                    "medium",
                    "Dict config sets reshard_after_forward=True (FSDP2 re-gathers weights in backward); if memory headroom permits, evaluate False (C3).",
                )
        self.generic_visit(node)


_REGEX_FALLBACK_RULES = [
    (
        "ddp_find_unused_true",
        "high",
        re.compile(r"\bfind_unused_parameters\s*=\s*True\b"),
        "[regex-fallback] DDP find_unused_parameters=True traverses the autograd graph every step; verify if all trainable params participate and consider False (C1).",
    ),
    (
        "fsdp1_limit_all_gathers_true",
        "high",
        re.compile(r"\blimit_all_gathers\s*=\s*True\b"),
        "[regex-fallback] FSDP1 limit_all_gathers=True throttles CPU AllGather launch; if memory headroom allows, evaluate False with prefetch (C2).",
    ),
    (
        "fsdp2_reshard_after_forward_true",
        "medium",
        re.compile(r"""(?:["']reshard_after_forward["']\s*:|\breshard_after_forward\s*=)\s*True\b"""),
        "[regex-fallback] FSDP2 reshard_after_forward=True re-gathers weights in backward; if memory headroom permits, evaluate False (C3).",
    ),
]


def _scan_python_regex_fallback(source: str, path: str) -> List[Finding]:
    findings: List[Finding] = []
    for idx, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for pattern, severity, regex, note in _REGEX_FALLBACK_RULES:
            m = regex.search(line)
            if m:
                findings.append(Finding(path, idx, m.start(), pattern, severity, stripped, note))
    return findings


def _parse_python(source: str, path: str) -> Optional[ast.AST]:
    try:
        return ast.parse(source, filename=path)
    except SyntaxError as exc:
        if "future feature annotations is not defined" in str(exc):
            stripped = "\n".join(
                "" if line.strip() == "from __future__ import annotations" else line
                for line in source.splitlines()
            )
            try:
                return ast.parse(stripped, filename=path)
            except SyntaxError:
                return None
        return None


def scan_python_file(path: str) -> List[Finding]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read()
    except OSError as exc:
        print("warning: could not read %s: %s" % (path, exc), file=sys.stderr)
        return []
    tree = _parse_python(source, path)
    if tree is None:
        print(
            "warning: AST parse unavailable for %s under current Python (%d.%d); using regex fallback"
            % (path, sys.version_info[0], sys.version_info[1]),
            file=sys.stderr,
        )
        return _scan_python_regex_fallback(source, path)
    visitor = _PyVisitor(path, source.splitlines())
    visitor.visit(tree)
    return visitor.findings


def _find_line(lines: List[str], key: str) -> Tuple[int, str]:
    for idx, line in enumerate(lines, 1):
        if key in line:
            return idx, line.strip()
    return 1, lines[0].strip() if lines else ""


def scan_json_config(path: str) -> List[Finding]:
    try:
        if os.path.getsize(path) > _MAX_CONFIG_BYTES:
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            raw = handle.read()
        data = json.loads(raw)
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict) or not isinstance(data.get("zero_optimization"), dict):
        return []

    lines = raw.splitlines()
    zero = data["zero_optimization"]
    stage = zero.get("stage")
    findings: List[Finding] = []

    if stage in (1, 2):
        if zero.get("overlap_comm") is False:
            ln, snip = _find_line(lines, "overlap_comm")
            findings.append(Finding(
                path, ln, 0, "ds_overlap_comm_disabled", "high", snip,
                "DeepSpeed ZeRO-%s config sets overlap_comm=false; evaluate enabling overlap_comm=true (C4)." % stage,
            ))
        if zero.get("contiguous_gradients") is False:
            ln, snip = _find_line(lines, "contiguous_gradients")
            findings.append(Finding(
                path, ln, 0, "ds_contiguous_gradients_disabled", "high", snip,
                "DeepSpeed ZeRO-%s config sets contiguous_gradients=false; evaluate enabling contiguous_gradients=true (C4)." % stage,
            ))
        if stage == 2 and zero.get("overlap_comm") is True:
            ln, snip = _find_line(lines, "zero_optimization")
            findings.append(Finding(
                path, ln, 0, "ds_zero2_config", "info", snip,
                "DeepSpeed ZeRO-2 with overlap_comm=true; if T5 still shows low backward overlap, check DeepSpeed version for double wait_stream (C6).",
            ))
    return findings


def scan_yaml_config(path: str) -> List[Finding]:
    try:
        if os.path.getsize(path) > _MAX_CONFIG_BYTES:
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            raw = handle.read()
    except OSError:
        return []
    if "zero_optimization" not in raw and "overlap_comm" not in raw and "contiguous_gradients" not in raw:
        return []

    findings: List[Finding] = []
    for idx, line in enumerate(raw.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.match(r"^overlap_comm\s*:\s*false\b", stripped, re.IGNORECASE):
            findings.append(Finding(
                path, idx, 0, "ds_overlap_comm_disabled", "high", stripped,
                "YAML config sets overlap_comm: false; evaluate enabling overlap_comm: true for ZeRO-1/2 (C4).",
            ))
        elif re.match(r"^contiguous_gradients\s*:\s*false\b", stripped, re.IGNORECASE):
            findings.append(Finding(
                path, idx, 0, "ds_contiguous_gradients_disabled", "high", stripped,
                "YAML config sets contiguous_gradients: false; evaluate enabling contiguous_gradients: true (C4).",
            ))
    return findings


def _iter_targets(paths: Iterable[str], include_configs: bool) -> Iterable[str]:
    exts = (".py", ".json", ".yaml", ".yml") if include_configs else (".py",)
    for path in paths:
        if os.path.isfile(path) and path.endswith(exts):
            yield path
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
                for name in sorted(files):
                    if name.endswith(exts):
                        yield os.path.join(root, name)


def scan_path(path: str) -> List[Finding]:
    if path.endswith(".py"):
        return scan_python_file(path)
    if path.endswith(".json"):
        return scan_json_config(path)
    if path.endswith((".yaml", ".yml")):
        return scan_yaml_config(path)
    return []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Files or directories to scan.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report.")
    parser.add_argument(
        "--min-severity",
        choices=["info", "medium", "high"],
        default="info",
        help="Only show findings at or above this severity (info < medium < high).",
    )
    parser.add_argument(
        "--no-configs",
        action="store_true",
        help="Skip scanning JSON/YAML DeepSpeed config files (Python AST only).",
    )
    args = parser.parse_args(argv)

    order = {"info": 0, "medium": 1, "high": 2}
    threshold = order[args.min_severity]

    all_findings: List[Finding] = []
    for target in _iter_targets(args.paths, include_configs=not args.no_configs):
        all_findings.extend(scan_path(target))

    all_findings = [f for f in all_findings if order[f.severity] >= threshold]
    all_findings.sort(key=lambda f: (f.file, f.line, f.col, f.pattern))

    if args.json:
        print(json.dumps([f.to_dict() for f in all_findings], ensure_ascii=False, indent=2))
    elif not all_findings:
        print("No candidates found (or all filtered by --min-severity).")
    else:
        by_sev: Dict[str, int] = {}
        for f in all_findings:
            by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
        summary = ", ".join(
            "%d %s" % (cnt, sev)
            for sev, cnt in sorted(by_sev.items(), key=lambda kv: -order[kv[0]])
        )
        print("%d static candidate(s): %s\n" % (len(all_findings), summary))
        for f in all_findings:
            print("%s:%d:%d  [%s] %s" % (f.file, f.line, f.col, f.severity, f.pattern))
            if f.snippet:
                print("    %s" % f.snippet)
            print("    note: %s\n" % f.note)

    print(
        "These are STATIC candidates only. They do not prove communication is the bottleneck "
        "or distinguish true exposed communication from slow-rank skew. Confirm with "
        "trace-bottleneck-report (T5) first (see references/triage-and-rccl.md).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
