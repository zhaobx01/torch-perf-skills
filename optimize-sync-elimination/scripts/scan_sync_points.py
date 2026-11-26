#!/usr/bin/env python3
"""Static scanner for host-device sync candidates in PyTorch code.

This is a STATIC scanner. It finds call sites that commonly cause an
implicit host-device synchronization (blocking D2H copy, scalar read that
forces the GPU queue to drain, explicit barrier/synchronize) or that are
preconditions for one (boolean mask indexing, ``nonzero``, ``unique``, ...).
It does NOT prove that any of these sites actually execute, or that they
actually synchronize at runtime -- some may be on code paths that never run,
operate on tensors already on CPU, or run when the GPU queue happens to be
idle anyway. It does not prove a hotspot or a speedup. Treat every finding as
a [静态] candidate to verify with a profiler trace (prefer
trace-bottleneck-report for GPU starvation, or see references/profiling.md and
scripts/analyze_trace_sync.py).

Usage:
    python3 scan_sync_points.py <file-or-dir> [<file-or-dir> ...]
    python3 scan_sync_points.py --json <file-or-dir> ...
    python3 scan_sync_points.py --min-severity medium <file-or-dir> ...

Only the Python standard library is used (compatible with Python 3.6+).
JSON output includes ``disclaimer``, ``files_scanned``, ``files_skipped``,
``skipped_files``, and ``findings``. Exit 1 indicates a partial or empty scan,
and exit 2 indicates invalid CLI arguments.
"""

import argparse
import ast
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple


_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules", ".venv", "venv"}
_DISCLAIMER = (
    "These are STATIC candidates only. They do not prove the code path runs, "
    "or that it synchronizes at runtime, forms a hotspot, or offers a speedup. "
    "Use a profiler trace for runtime evidence (prefer trace-bottleneck-report for GPU starvation; "
    "see references/profiling.md)."
)

# (severity, human note)
# severity: "high" sync-causing | "medium" likely dynamic-shape precondition | "info" verify-only
_METHOD_PATTERNS = {
    "item": ("high", "Reads a tensor value to Python; a common blocking D2H sync."),
    "cpu": ("high", "Moves a tensor to host; a device tensor normally incurs a blocking D2H copy."),
    "tolist": ("high", "Reads tensor values to a Python list; implies a D2H sync if the tensor is on a CUDA/HIP device."),
    "numpy": ("high", "Converts a tensor to a NumPy array; if chained after .cpu() it is part of the same transfer."),
    "nonzero": ("medium", "Output shape depends on tensor contents; check for host synchronization on the target device."),
    "masked_select": ("medium", "Output shape depends on the mask; same dynamic-shape precondition as nonzero."),
    "unique": ("medium", "Output size depends on tensor contents and triggers sorting; check if redundant after nonzero."),
    "bincount": ("medium", "Output size can depend on tensor contents; minlength is only a lower bound."),
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
_TORCH_FACTORIES = {
    "arange", "as_tensor", "empty", "empty_like", "eye", "full", "full_like",
    "linspace", "ones", "ones_like", "rand", "rand_like", "randn", "randn_like",
    "tensor", "zeros", "zeros_like",
}
_MASK_PRESERVING_METHODS = {
    "clone", "contiguous", "detach", "flatten", "reshape", "squeeze", "unsqueeze", "view",
}
_DEVICE_PRESERVING_METHODS = _MASK_PRESERVING_METHODS | {
    "abs", "all", "any", "bool", "double", "float", "int", "long", "max", "mean", "min", "sum",
}
_TENSOR_COMPARISONS = {"eq", "ge", "gt", "le", "lt", "ne"}
_MASK_FUNCTIONS = _TENSOR_COMPARISONS | {
    "isfinite", "isinf", "isnan", "logical_and", "logical_not", "logical_or", "logical_xor",
}
_COMPARISON_OPS = (ast.Eq, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.NotEq)
_LEGACY_LITERAL_TYPES = tuple(
    cls for cls in (
        getattr(ast, "Num", None),
        getattr(ast, "Str", None),
        getattr(ast, "Bytes", None),
        getattr(ast, "NameConstant", None),
    )
    if cls is not None
)


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


def _const_str(node: Optional[ast.AST]) -> Optional[str]:
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return node.s
    return None


def _is_literal(node: ast.AST) -> bool:
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):
        return True
    return bool(_LEGACY_LITERAL_TYPES) and isinstance(node, _LEGACY_LITERAL_TYPES)


def _keyword_value(node: ast.Call, name: str) -> Optional[ast.AST]:
    return next((keyword.value for keyword in node.keywords if keyword.arg == name), None)


def _is_nonnegative_int(node: Optional[ast.AST]) -> bool:
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):
        return type(node.value) is int and node.value >= 0
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        return type(node.n) is int and node.n >= 0
    return False


def _repeat_interleave_has_fixed_size(node: ast.Call, dotted: str) -> bool:
    repeats = _keyword_value(node, "repeats")
    index = 1 if dotted == "torch.repeat_interleave" else 0
    if repeats is None and len(node.args) > index:
        repeats = node.args[index]
    return _is_nonnegative_int(repeats) or _is_nonnegative_int(_keyword_value(node, "output_size"))


def _is_cpu_device(expr: Optional[ast.AST]) -> bool:
    if isinstance(expr, ast.Call) and _dotted_name(expr.func) == "torch.device":
        expr = expr.args[0] if expr.args else _keyword_value(expr, "type")
    device = _const_str(expr)
    return device is not None and re.fullmatch(r"cpu(?::[0-9]+)?", device) is not None


def _is_to_cpu_call(node: ast.Call) -> bool:
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "to"):
        return False
    device = _keyword_value(node, "device")
    return _is_cpu_device(device) or bool(node.args and _is_cpu_device(node.args[0]))


def _subscript_items(node: ast.AST) -> List[ast.AST]:
    # Before Python 3.9, dimensions may be wrapped in Index or ExtSlice.
    if hasattr(ast, "Index") and isinstance(node, ast.Index):
        return _subscript_items(node.value)
    if hasattr(ast, "ExtSlice") and isinstance(node, ast.ExtSlice):
        return [item for dim in node.dims for item in _subscript_items(dim)]
    return node.elts if isinstance(node, ast.Tuple) else [node]


def _writes_in(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)):
            if child.name:
                names.add(child.name)
        elif isinstance(child, ast.alias):
            names.add(child.asname or child.name.split(".")[0])
        elif type(child).__name__ in ("MatchAs", "MatchStar", "MatchMapping"):
            name = getattr(child, "name", None) or getattr(child, "rest", None)
            if name:
                names.add(name)
    return names


def _writes_in_nodes(nodes: Iterable[ast.AST]) -> Set[str]:
    names: Set[str] = set()
    for node in nodes:
        names.update(_writes_in(node))
    return names


def _dotted_name(node: Optional[ast.AST]) -> str:
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
        self._facts: Dict[str, Set[str]] = {}

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

    def _expression_facts(self, expr: Optional[ast.AST]) -> Set[str]:
        if isinstance(expr, ast.Name):
            return self._facts.get(expr.id, set())
        if isinstance(expr, ast.Subscript):
            return self._expression_facts(expr.value)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            return self._expression_facts(expr.left) & self._expression_facts(expr.right) & {"mask", "cpu"}
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Invert):
            return self._expression_facts(expr.operand)
        if isinstance(expr, ast.Compare):
            if len(expr.ops) == 1 and isinstance(expr.ops[0], _COMPARISON_OPS) and any(
                self._expression_facts(operand) for operand in [expr.left] + expr.comparators
            ):
                return {"mask"}
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Attribute):
            return set()

        name = expr.func.attr
        torch_function = _dotted_name(expr.func) == "torch." + name
        if torch_function and name in _MASK_FUNCTIONS:
            return {"mask"}
        dtype = _keyword_value(expr, "dtype")
        if torch_function and name in _TORCH_FACTORIES:
            facts = {"tensor"}
            if _is_cpu_device(_keyword_value(expr, "device")):
                facts.add("cpu")
            if _dotted_name(dtype) == "torch.bool":
                facts.add("mask")
            return facts

        receiver = self._expression_facts(expr.func.value)
        if name == "cpu":
            return receiver | {"cpu"}
        if name == "to":
            facts = {"cpu"} if _is_to_cpu_call(expr) else set()
            if _dotted_name(dtype) == "torch.bool" or (expr.args and _dotted_name(expr.args[0]) == "torch.bool"):
                facts.add("mask")
            elif facts and dtype is None and len(expr.args) <= 1:
                facts.update(receiver & {"mask"})
            return facts
        if name in _MASK_PRESERVING_METHODS:
            return receiver
        if name == "bool" or (name in _TENSOR_COMPARISONS and receiver):
            return (receiver & {"cpu"}) | {"mask"}
        if name in _DEVICE_PRESERVING_METHODS:
            return receiver & {"cpu", "tensor"}
        return set()

    def _visit_assignment(self, node, targets) -> None:
        facts = self._expression_facts(node.value)
        if node.value is not None:
            self.visit(node.value)
        for target in targets:
            self.visit(target)
            if isinstance(target, ast.Name) and facts:
                self._facts[target.id] = facts

    def visit_Assign(self, node: ast.Assign) -> None:
        self._visit_assignment(node, node.targets)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        self._visit_assignment(node, [node.target])

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._facts.pop(node.id, None)

    def visit_alias(self, node: ast.alias) -> None:
        self._facts.pop(node.asname or node.name.split(".")[0], None)

    def _visit_scope(self, node: ast.AST) -> None:
        previous_facts, previous_depth = self._facts, self._loop_depth
        self._facts, self._loop_depth = {}, 0
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for arg in (child for child in ast.iter_child_nodes(node.args) if isinstance(child, ast.arg)):
                if _dotted_name(arg.annotation) == "torch.Tensor":
                    self._facts[arg.arg] = {"tensor"}
        self.generic_visit(node)
        self._facts, self._loop_depth = previous_facts, previous_depth
        if hasattr(node, "name"):
            self._facts.pop(node.name, None)

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = visit_Lambda = _visit_scope

    def _visit_control_flow(self, node: ast.AST) -> None:
        # Track assignments within a branch and forget potential rebindings
        # at joins. Avoid assuming any particular branch or loop trip count.
        writes = _writes_in(node)
        entry = {name: facts for name, facts in self._facts.items() if name not in writes}
        for _, field in ast.iter_fields(node):
            self._facts = entry.copy()
            if isinstance(field, list):
                for child in field:
                    if isinstance(child, ast.AST):
                        self.visit(child)
            elif isinstance(field, ast.AST):
                self.visit(field)
        self._facts = entry

    visit_Try = visit_TryStar = _visit_control_flow
    visit_Match = visit_match_case = visit_ExceptHandler = _visit_control_flow

    def visit_With(self, node: ast.With) -> None:
        # Context managers like torch.no_grad() / autocast() execute sequentially;
        # preserve bindings produced inside the block for subsequent statements.
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    visit_AsyncWith = visit_With

    def visit_For(self, node: ast.For) -> None:
        # Evaluate the iterable once before entering the loop body.
        self.visit(node.iter)
        writes = _writes_in(node.target) | _writes_in_nodes(node.body) | _writes_in_nodes(node.orelse)
        loop_entry = {name: facts for name, facts in self._facts.items() if name not in writes}
        self._facts = loop_entry.copy()
        self.visit(node.target)
        self._loop_depth += 1
        self._facts = loop_entry.copy()
        for stmt in node.body:
            self.visit(stmt)
        self._loop_depth -= 1
        self._facts = loop_entry.copy()
        for stmt in node.orelse:
            self.visit(stmt)
        self._facts = loop_entry

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        writes = _writes_in(node.test) | _writes_in_nodes(node.body) | _writes_in_nodes(node.orelse)
        loop_entry = {name: facts for name, facts in self._facts.items() if name not in writes}
        self._loop_depth += 1
        self._facts = loop_entry.copy()
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        self._loop_depth -= 1
        self._facts = loop_entry.copy()
        for stmt in node.orelse:
            self.visit(stmt)
        self._facts = loop_entry

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if any("mask" in self._expression_facts(item) for item in _subscript_items(node.slice)):
            self._emit(
                node,
                "boolean indexing",
                "medium",
                "Index has a local boolean tensor source. Verify the device, read/write lowering, and dynamic-shape cost.",
            )
        self.generic_visit(node)

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
                    "CPU destination. Verify source device and copy mode; wait before reading an asynchronous D2H result.",
                )
            elif name in _METHOD_PATTERNS:
                if name == "repeat_interleave" and _repeat_interleave_has_fixed_size(node, dotted):
                    self.generic_visit(node)
                    return
                severity, note = _METHOD_PATTERNS[name]
                if name in ("item", "tolist", "numpy") and "cpu" in self._expression_facts(func.value):
                    severity = "info"
                    note = (
                        "Receiver has a local CPU origin. The read does not issue another D2H copy; "
                        "wait before reading if an earlier asynchronous copy is still pending."
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
        # Evaluate the branch condition before invalidating variables written inside branches.
        self.visit(test)
        pre_branch = self._facts.copy()
        writes = _writes_in_nodes(node.body) | _writes_in_nodes(node.orelse)
        for branch in (node.body, node.orelse):
            self._facts = pre_branch.copy()
            for stmt in branch:
                self.visit(stmt)
        self._facts = {name: facts for name, facts in pre_branch.items() if name not in writes}


def _iter_python_files(paths: Iterable[str], on_error) -> Iterable[str]:
    for path in paths:
        if os.path.isfile(path) and path.endswith(".py"):
            yield path
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path, onerror=on_error):
                dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
                for name in sorted(files):
                    if name.endswith(".py"):
                        yield os.path.join(root, name)


def _parse_source(source: str, path: str) -> Tuple[Optional[ast.AST], Optional[str]]:
    try:
        return ast.parse(source, filename=path), None
    except SyntaxError as exc:
        if "future feature annotations is not defined" in str(exc):
            stripped = "\n".join(
                "" if line.strip() == "from __future__ import annotations" else line
                for line in source.splitlines()
            )
            try:
                return ast.parse(stripped, filename=path), None
            except SyntaxError as exc2:
                reason = str(exc2)
                print("warning: could not parse %s: %s" % (path, reason), file=sys.stderr)
                return None, reason
        reason = str(exc)
        print("warning: could not parse %s: %s" % (path, reason), file=sys.stderr)
        return None, reason


def scan_file(path: str) -> Tuple[Optional[List[Finding]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read()
    except OSError as exc:
        reason = str(exc)
        print("warning: could not read %s: %s" % (path, reason), file=sys.stderr)
        return None, reason
    tree, reason = _parse_source(source, path)
    if tree is None:
        return None, reason
    visitor = _Visitor(path, source.splitlines())
    visitor.visit(tree)
    return visitor.findings, None


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
    for path in args.paths:
        if not (os.path.isdir(path) or (os.path.isfile(path) and path.endswith(".py"))):
            parser.error("expected an existing Python file or directory: %s" % path)

    order = {"info": 0, "medium": 1, "high": 2}
    threshold = order[args.min_severity]

    all_findings: List[Finding] = []
    scanned = 0
    skipped_files: List[Dict[str, str]] = []

    def walk_error(error):
        path_name = getattr(error, "filename", None) or "<walk>"
        reason = str(error)
        skipped_files.append({"file": str(path_name), "reason": reason})
        print("warning: skipped %s" % reason, file=sys.stderr)

    for path in _iter_python_files(args.paths, walk_error):
        result, reason = scan_file(path)
        if result is None:
            skipped_files.append({"file": path, "reason": reason or "unknown error"})
        else:
            scanned += 1
            all_findings.extend(result)

    all_findings = [f for f in all_findings if order[f.severity] >= threshold]
    all_findings.sort(key=lambda f: (f.file, f.line, f.col, f.pattern))
    skipped = len(skipped_files)

    if skipped or not scanned:
        print("Python files: %d scanned, %d skipped." % (scanned, skipped), file=sys.stderr)
    if args.json:
        payload = {
            "disclaimer": _DISCLAIMER,
            "files_scanned": scanned,
            "files_skipped": skipped,
            "skipped_files": skipped_files,
            "findings": [f.to_dict() for f in all_findings],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif not scanned:
        print("No Python files scanned.")
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

    print(_DISCLAIMER, file=sys.stderr)
    return 1 if skipped or not scanned else 0


if __name__ == "__main__":
    raise SystemExit(main())
