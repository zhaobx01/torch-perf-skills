#!/usr/bin/env python3
"""Static scanner for host-device sync candidates in PyTorch code.

This is a STATIC scanner. It finds call sites that commonly cause an
implicit host-device synchronization (blocking D2H copy, scalar read that
forces the GPU queue to drain, explicit barrier/synchronize) or that are
preconditions for one (boolean mask indexing, ``nonzero``, ``unique``, ...).
It does NOT prove that any of these sites actually execute, or that they
actually synchronize at runtime -- some may be on code paths that never run,
operate on tensors already on CPU, or run when the GPU queue happens to be
idle anyway. Treat every finding as a [静态] candidate to verify with a
profiler trace (prefer trace-bottleneck-report for GPU starvation, or see
references/profiling.md and scripts/analyze_trace_sync.py).

Usage:
    python3 scan_sync_points.py <file-or-dir> [<file-or-dir> ...]
    python3 scan_sync_points.py --json <file-or-dir> ...
    python3 scan_sync_points.py --min-severity medium <file-or-dir> ...

Only the Python standard library is used (compatible with Python 3.6+).
"""

import argparse
import ast
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple


# (severity, human note)
# severity: "high" sync-causing | "medium" likely dynamic-shape precondition | "info" verify-only
_METHOD_PATTERNS = {
    "item": ("high", "Reads a tensor value to Python; a common blocking D2H sync."),
    "cpu": ("high", "Moves a tensor to host; blocking unless handled asynchronously with pinned memory and a proper wait."),
    "tolist": ("high", "Reads tensor values to a Python list; implies a D2H sync if the tensor is on a CUDA/HIP device."),
    "numpy": ("high", "Converts a tensor to a NumPy array; if chained after .cpu() it is part of the same transfer."),
    "nonzero": ("medium", "Output shape depends on tensor contents; may force host sync if downstream allocates from its length."),
    "masked_select": ("medium", "Output shape depends on the mask; same dynamic-shape precondition as nonzero."),
    "unique": ("medium", "Output size depends on tensor contents and triggers sorting; check if redundant after nonzero."),
    "bincount": ("medium", "Output size can depend on tensor contents unless `minlength` fixes it."),
    "repeat_interleave": ("medium", "Without `output_size`, PyTorch may need to read `repeats` to size the output."),
    "synchronize": ("high", "Explicit synchronization call (cuda.synchronize / stream.synchronize / event.synchronize)."),
    "barrier": ("high", "Explicit distributed barrier; check if mistakenly placed inside gradient-accumulation micro-steps."),
    "all": ("info", "Often precedes a Python-side branch on a boolean tensor; check the call site, not this call alone."),
    "any": ("info", "Often precedes a Python-side branch on a boolean tensor; check the call site, not this call alone."),
}

_TORCH_FUNCTION_PATTERNS = {
    "torch.cuda.synchronize": ("high", "Explicit device-wide synchronization."),
    "torch.distributed.barrier": ("high", "Explicit distributed barrier; verify it is not on every micro-step."),
    "dist.barrier": ("high", "Explicit distributed barrier; verify it is not on every micro-step."),
    "torch.cuda.Event": ("info", "Event objects are fine; verify .synchronize() calls elsewhere and record()/wait() pairs."),
    "torch.nonzero": ("medium", "Same as tensor.nonzero(); output shape depends on contents."),
    "torch.argwhere": ("medium", "Equivalent to nonzero(); output shape depends on tensor contents."),
    "torch.masked_select": ("medium", "Same as tensor.masked_select()."),
    "torch.unique": ("medium", "Same as tensor.unique(); output size depends on tensor contents."),
    "torch.unique_consecutive": ("medium", "Output size depends on tensor contents."),
}

_BUILTIN_SCALAR_CASTS = {"int", "float", "bool"}


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
        in_loop: bool = False,
    ) -> None:
        self.file = file
        self.line = line
        self.col = col
        self.pattern = pattern
        self.severity = severity
        self.snippet = snippet
        self.note = note
        self.in_loop = in_loop

    def to_dict(self) -> Dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "col": self.col,
            "pattern": self.pattern,
            "severity": self.severity,
            "in_loop": self.in_loop,
            "snippet": self.snippet,
            "note": self.note,
        }


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Str):
        return node.s
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_literal(node: ast.AST) -> bool:
    literal_types = (ast.Num, ast.Str, ast.Bytes, ast.NameConstant)
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):
        return True
    return isinstance(node, literal_types)


def _is_to_cpu_call(node: ast.Call) -> bool:
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "to"):
        return False
    if node.args:
        val = _const_str(node.args[0])
        if val is not None and val.lower().startswith("cpu"):
            return True
    for kw in node.keywords:
        if kw.arg == "device":
            val = _const_str(kw.value)
            if val is not None and val.lower().startswith("cpu"):
                return True
    return False


def _receiver_already_cpu_transfer(expr: ast.AST) -> bool:
    """Detect direct chains like `x.detach().cpu().tolist()` or `x.to('cpu').tolist()`."""
    cur = expr
    while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
        attr = cur.func.attr
        if attr == "cpu" or _is_to_cpu_call(cur):
            return True
        if attr in ("detach", "contiguous", "flatten", "squeeze", "unsqueeze", "view", "reshape", "long", "int", "float"):
            cur = cur.func.value
            continue
        break
    return False


def _dotted_name(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str, source_lines: List[str]) -> None:
        self.filename = filename
        self.source_lines = source_lines
        self.findings: List[Finding] = []
        self._loop_depth = 0

    def _snippet(self, node: ast.AST) -> str:
        line_no = getattr(node, "lineno", None)
        if line_no is None or line_no > len(self.source_lines):
            return ""
        return self.source_lines[line_no - 1].strip()

    def _emit(self, node: ast.AST, pattern: str, severity: str, note: str) -> None:
        in_loop = self._loop_depth > 0
        full_note = note
        if in_loop and severity in ("high", "medium"):
            full_note = "[in-loop] " + note
        self.findings.append(
            Finding(
                file=self.filename,
                line=getattr(node, "lineno", -1),
                col=getattr(node, "col_offset", -1),
                pattern=pattern,
                severity=severity,
                snippet=self._snippet(node),
                note=full_note,
                in_loop=in_loop,
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

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
            dotted = _dotted_name(func)
            if dotted == "torch.where":
                # Only single-argument torch.where(cond) behaves like nonzero().
                # 3-argument torch.where(cond, a, b) is an elementwise device op.
                if len(node.args) == 1 and not node.keywords:
                    self._emit(
                        node,
                        "torch.where(cond)",
                        "medium",
                        "Single-argument torch.where(cond) behaves like nonzero (output shape depends on data).",
                    )
            elif dotted in _TORCH_FUNCTION_PATTERNS:
                severity, note = _TORCH_FUNCTION_PATTERNS[dotted]
                self._emit(node, dotted, severity, note)
            elif _is_to_cpu_call(node):
                self._emit(
                    node,
                    '.to("cpu")',
                    "high",
                    "Moves a tensor to CPU; blocking D2H sync if the tensor is on GPU and non_blocking/pinned is not used.",
                )
            elif name in _METHOD_PATTERNS:
                severity, note = _METHOD_PATTERNS[name]
                if name in ("tolist", "numpy") and _receiver_already_cpu_transfer(func.value):
                    severity = "info"
                    note = (
                        "Chained directly on .cpu()/.to('cpu'); the tensor is already on CPU "
                        "after the preceding transfer (not a second sync)."
                    )
                self._emit(node, ".%s()" % name, severity, note)
        elif isinstance(func, ast.Name) and func.id in _BUILTIN_SCALAR_CASTS:
            if len(node.args) == 1 and not _is_literal(node.args[0]):
                self._emit(
                    node,
                    "%s(...)" % func.id,
                    "info",
                    "%s() on a non-literal argument; if the argument is a GPU tensor this reads its value to Python (D2H sync). Verify the argument type." % func.id,
                )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        test = node.test
        if isinstance(test, (ast.Call, ast.Attribute)):
            self._emit(
                node,
                "if <call-or-attr>",
                "info",
                "Branch condition is a call or attribute access; if it evaluates a 0-dim GPU tensor's truthiness, this triggers an implicit D2H sync. Verify the expression type.",
            )
        self.generic_visit(node)


def _iter_python_files(paths: Iterable[str]) -> Iterable[str]:
    for path in paths:
        if os.path.isfile(path) and path.endswith(".py"):
            yield path
        elif os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for name in sorted(files):
                    if name.endswith(".py"):
                        yield os.path.join(root, name)


def _parse_source(source: str, path: str) -> Optional[ast.AST]:
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
            except SyntaxError as exc2:
                print("warning: could not parse %s: %s" % (path, exc2), file=sys.stderr)
                return None
        print("warning: could not parse %s: %s" % (path, exc), file=sys.stderr)
        return None


def scan_file(path: str) -> List[Finding]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read()
    except OSError as exc:
        print("warning: could not read %s: %s" % (path, exc), file=sys.stderr)
        return []
    tree = _parse_source(source, path)
    if tree is None:
        return []
    visitor = _Visitor(path, source.splitlines())
    visitor.visit(tree)
    return visitor.findings


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Files or directories to scan.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    parser.add_argument(
        "--min-severity",
        choices=["info", "medium", "high"],
        default="info",
        help="Only show findings at or above this severity (info < medium < high).",
    )
    args = parser.parse_args(argv)

    order = {"info": 0, "medium": 1, "high": 2}
    threshold = order[args.min_severity]

    all_findings: List[Finding] = []
    for path in _iter_python_files(args.paths):
        all_findings.extend(scan_file(path))

    all_findings = [f for f in all_findings if order[f.severity] >= threshold]
    all_findings.sort(key=lambda f: (f.file, f.line, f.col, f.pattern))

    if args.json:
        print(json.dumps([f.to_dict() for f in all_findings], ensure_ascii=False, indent=2))
    elif not all_findings:
        print("No candidates found (or all filtered by --min-severity).")
    else:
        by_severity: Dict[str, int] = {}
        for finding in all_findings:
            by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
        summary = ", ".join(
            "%d %s" % (count, sev)
            for sev, count in sorted(by_severity.items(), key=lambda kv: -order[kv[0]])
        )
        print("%d static candidate(s): %s\n" % (len(all_findings), summary))
        for finding in all_findings:
            loop_tag = " [in-loop]" if finding.in_loop else ""
            print("%s:%d:%d  [%s]%s %s" % (
                finding.file, finding.line, finding.col, finding.severity, loop_tag, finding.pattern
            ))
            if finding.snippet:
                print("    %s" % finding.snippet)
            print("    note: %s\n" % finding.note)

    print(
        "These are STATIC candidates only. They do not prove the code path runs, "
        "or that it actually synchronizes at runtime. Confirm with a profiler "
        "trace before changing anything (prefer trace-bottleneck-report for GPU starvation; "
        "see references/profiling.md).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
