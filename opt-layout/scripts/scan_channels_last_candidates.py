#!/usr/bin/env python3
"""Static scanner for channels-last CANDIDATES and layout boundary traps in PyTorch code.

It summarizes 2D/3D convolution subgraphs per file and flags high-signal layout
boundaries or traps worth checking. It does NOT prove convolutions are hot or
that channels_last / channels_last_3d will improve performance. Confirm with a
profiler trace first.

Detected patterns (all name/AST-based, so false positives are expected):
  conv2d_subgraph          Per-file summary of PyTorch 2D convolution modules and functional calls
  conv3d_subgraph          Per-file summary of PyTorch 3D convolution modules and functional calls
  rearrange_5d_4d          einops.rearrange switching between 5D and 4D (layout transition boundary)
  permute_nhwc_trap        .permute(0,2,3,1) / (0,2,3,4,1) / (0,3,1,2) (logical axis vs memory_format trap)
  existing_channels_last   Existing channels_last / channels_last_3d / convert_conv*d_weight_memory_format / force_stride_order
  miopen_suggest_env       Python environment access for PYTORCH_MIOPEN_SUGGEST_NHWC / NDHWC
  optimizer_state_save     Optimizer state_dict() in checkpoint/save paths (check contiguous checkpoint)

Usage:
    python3 scan_channels_last_candidates.py <file-or-dir> [...]
    python3 scan_channels_last_candidates.py --json <file-or-dir> [...]
    python3 scan_channels_last_candidates.py --patterns conv3d_subgraph,rearrange_5d_4d <file-or-dir>

Only the Python standard library is used (Python 3.6+ compatible).
Use an interpreter that can parse the target project. Skipped files are
reported on stderr, including in JSON mode. This scans Python files only.
"""

import argparse
import ast
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}

_NOTES = {
    "conv2d_subgraph": (
        "2D convolution sites found in this file; if profiler confirms Conv2d is hot, evaluate "
        "scoped torch.channels_last on this subgraph/backbone (L1, L2)."
    ),
    "conv3d_subgraph": (
        "3D convolution sites found in this file; if profiler confirms Conv3d is hot, evaluate "
        "torch.channels_last_3d and check backward weight-grad regression (L1, L3)."
    ),
    "rearrange_5d_4d": (
        "einops.rearrange transitions between 5D and 4D tensors; verify explicit channels_last / "
        "channels_last_3d restoration at this boundary to avoid silent layout loss (L2)."
    ),
    "permute_nhwc_trap": (
        "Logical NHWC/NDHWC axis permutation; do not confuse logical permute with physical "
        "torch.channels_last memory_format or blindly convert non-image tensors (L2)."
    ),
    "existing_channels_last": (
        "Existing channels-last / stride-order code; verify scope, custom-op/cache alignment, "
        "compile guards, and clean rollback path (L1, L5, L6)."
    ),
    "miopen_suggest_env": (
        "ROCm MIOpen layout hint reference; verify target PyTorch build reads this variable, "
        "keep tensor/weight conversion explicit, and isolate from MIOPEN_FIND_MODE in ablations (L7)."
    ),
    "optimizer_state_save": (
        "Optimizer checkpoint save path; when weights are channels_last, check whether "
        "optimizer.state_dict() tensors must be made .contiguous() before saving (L4)."
    ),
}

_NHWC_PERMUTES = {
    (0, 2, 3, 1),
    (0, 3, 1, 2),
    (0, 2, 3, 4, 1),
    (0, 4, 1, 2, 3),
}

_REARRANGE_ARROW = re.compile(r"^(.+?)\s*->\s*(.+?)$")
_CONV_MODULES = {
    "Conv2d": 2,
    "ConvTranspose2d": 2,
    "Conv3d": 3,
    "ConvTranspose3d": 3,
}
_CONV_FUNCTIONS = {
    "conv2d": 2,
    "conv_transpose2d": 2,
    "conv3d": 3,
    "conv_transpose3d": 3,
}
_LAYOUT_FORMATS = {"torch.channels_last", "torch.channels_last_3d"}
_LAYOUT_HELPERS = {
    "torch.nn.utils.convert_conv2d_weight_memory_format",
    "torch.nn.utils.convert_conv3d_weight_memory_format",
    "torch.nn.utils.memory_format.convert_conv2d_weight_memory_format",
    "torch.nn.utils.memory_format.convert_conv3d_weight_memory_format",
    "torch._inductor.inductor_prims.force_stride_order",
}
_MIOPEN_KEYS = {"PYTORCH_MIOPEN_SUGGEST_NHWC", "PYTORCH_MIOPEN_SUGGEST_NDHWC"}
_ENV_CALLS = {
    "os.getenv", "os.putenv", "os.unsetenv",
    "os.environ.get", "os.environ.setdefault", "os.environ.pop",
}
_PERMUTE_CALL = re.compile(r"\bpermute\s*\(")


class Finding(object):
    __slots__ = ("file", "line", "pattern", "snippet", "note")

    def __init__(self, file, line, pattern, snippet, note):
        # type: (str, int, str, str, str) -> None
        self.file = file
        self.line = line
        self.pattern = pattern
        self.snippet = snippet
        self.note = note

    def to_dict(self):
        # type: () -> Dict[str, object]
        return {
            "file": self.file,
            "line": self.line,
            "pattern": self.pattern,
            "snippet": self.snippet,
            "note": self.note,
        }


def _dotted(node):
    # type: (ast.AST) -> str
    parts = []  # type: List[str]
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _str_const(node):
    # type: (Optional[ast.AST]) -> Optional[str]
    if node is None:
        return None
    if isinstance(node, ast.Str):
        return node.s
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _int_const(node):
    # type: (Optional[ast.AST]) -> Optional[int]
    if node is None:
        return None
    if isinstance(node, ast.Num) and isinstance(node.n, int) and not isinstance(node.n, bool):
        return node.n
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _count_einops_axes(side):
    # type: (str) -> int
    # Collapse parenthesized groups like "(b t)" into a single axis token
    collapsed = re.sub(r"\([^)]*\)", " GROUP ", side)
    tokens = [tok for tok in collapsed.strip().split() if tok]
    if "..." in tokens:
        return -1
    return len(tokens)


def _is_5d_4d_rearrange(spec):
    # type: (Optional[str]) -> bool
    if not spec:
        return False
    match = _REARRANGE_ARROW.match(spec.strip())
    if not match:
        return False
    left_n = _count_einops_axes(match.group(1))
    right_n = _count_einops_axes(match.group(2))
    return (left_n, right_n) in ((5, 4), (4, 5))


def _extract_permute_tuple(args):
    # type: (List[ast.AST]) -> Optional[Tuple[int, ...]]
    elems = args
    if len(args) == 1 and isinstance(args[0], (ast.Tuple, ast.List)):
        elems = args[0].elts
    vals = []  # type: List[int]
    for el in elems:
        v = _int_const(el)
        if v is None:
            return None
        vals.append(v)
    return tuple(vals)


class _Visitor(ast.NodeVisitor):
    def __init__(self, path, lines):
        # type: (str, List[str]) -> None
        self.path = path
        self.lines = lines
        self.findings = []  # type: List[Finding]
        self.conv2d_lines = []  # type: List[int]
        self.conv3d_lines = []  # type: List[int]
        self._imports = [{}]  # type: List[Dict[str, str]]
        self._save_context = []  # type: List[bool]
        self._emitted_keys = set()  # type: Set[Tuple[int, str]]

    def _emit(self, line, pattern, snippet=None):
        # type: (int, str, Optional[str]) -> None
        key = (line, pattern)
        if key in self._emitted_keys:
            return
        self._emitted_keys.add(key)
        if snippet is None:
            snippet = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        self.findings.append(Finding(self.path, line, pattern, snippet, _NOTES[pattern]))

    def _qualified(self, name):
        root, dot, rest = name.partition(".")
        for imports in reversed(self._imports):
            if root in imports:
                module = imports[root]
                return module + dot + rest if module else ""
        return ""

    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            self._imports[-1][alias.asname or root] = alias.name if alias.asname else root

    def visit_ImportFrom(self, node):
        for alias in node.names:
            qualified = node.module + "." + alias.name if node.module and not node.level else ""
            self._imports[-1][alias.asname or alias.name] = qualified

    def _record_conv(self, qualified, line, allow_function=False):
        module, _, symbol = qualified.rpartition(".")
        dimension = _CONV_MODULES.get(symbol) if module in ("torch.nn", "torch.nn.modules.conv") else None
        if dimension is None and allow_function and module in ("torch", "torch.nn.functional"):
            dimension = _CONV_FUNCTIONS.get(symbol)
        if dimension == 2:
            self.conv2d_lines.append(line)
        elif dimension == 3:
            self.conv3d_lines.append(line)

    def visit_ClassDef(self, node):
        for base in node.bases:
            self._record_conv(self._qualified(_dotted(base)), node.lineno)
        self._imports.append({})
        self.generic_visit(node)
        self._imports.pop()

    def visit_FunctionDef(self, node):
        lname = node.name.lower()
        arguments = node.args.args + node.args.kwonlyargs + getattr(node.args, "posonlyargs", [])
        self._imports.append({arg.arg: "" for arg in arguments})
        takes_save_path = any(arg.arg in ("save_dir", "save_path") for arg in arguments)
        self._save_context.append("save" in lname or "checkpoint" in lname or takes_save_path)
        self.generic_visit(node)
        self._save_context.pop()
        self._imports.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _permute_line(self, node):
        # Python 3.6 can report the first call in a chain for later calls.
        for line in range(node.args[0].lineno, node.lineno - 1, -1):
            if _PERMUTE_CALL.search(self.lines[line - 1]):
                return line
        return node.args[0].lineno

    def visit_Call(self, node):
        # type: (ast.Call) -> None
        name = _dotted(node.func)
        tail = name.rsplit(".", 1)[-1]
        qualified = self._qualified(name)

        self._record_conv(qualified, node.lineno, allow_function=True)
        if qualified in ("einops.rearrange", "einops.einops.rearrange"):
            spec = _str_const(node.args[1]) if len(node.args) >= 2 else None
            if _is_5d_4d_rearrange(spec):
                self._emit(node.lineno, "rearrange_5d_4d")
        elif tail == "permute":
            tup = _extract_permute_tuple(node.args)
            if tup in _NHWC_PERMUTES:
                self._emit(self._permute_line(node), "permute_nhwc_trap")
        elif qualified in _LAYOUT_HELPERS:
            self._emit(node.lineno, "existing_channels_last")
        elif tail == "state_dict" and any(self._save_context) and "optim" in name.lower():
            self._emit(node.lineno, "optimizer_state_save")
        elif qualified in _ENV_CALLS:
            key = node.args[0] if node.args else next(
                (kw.value for kw in node.keywords if kw.arg == "key"), None
            )
            if _str_const(key) in _MIOPEN_KEYS:
                self._emit(node.lineno, "miopen_suggest_env")

        self.generic_visit(node)

    def visit_Subscript(self, node):
        if self._qualified(_dotted(node.value)) == "os.environ":
            key = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            if _str_const(key) in _MIOPEN_KEYS:
                self._emit(node.lineno, "miopen_suggest_env")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if self._qualified(_dotted(node)) in _LAYOUT_FORMATS:
            self._emit(node.lineno, "existing_channels_last")
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and self._qualified(node.id) in _LAYOUT_FORMATS:
            self._emit(node.lineno, "existing_channels_last")


def scan_file(path, allowed_patterns=None):
    # type: (str, Optional[Set[str]]) -> Optional[List[Finding]]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=path)
    except (OSError, SyntaxError) as exc:
        print("warning: skipped {}: {}".format(path, exc), file=sys.stderr)
        return None
    lines = source.splitlines()
    visitor = _Visitor(path, lines)
    visitor.visit(tree)

    findings = list(visitor.findings)
    if visitor.conv2d_lines:
        first = visitor.conv2d_lines[0]
        snippet = "{} Conv2d/ConvTranspose2d site(s); first: {}".format(
            len(visitor.conv2d_lines),
            lines[first - 1].strip() if 0 < first <= len(lines) else "",
        )
        findings.append(Finding(path, first, "conv2d_subgraph", snippet, _NOTES["conv2d_subgraph"]))
    if visitor.conv3d_lines:
        first = visitor.conv3d_lines[0]
        snippet = "{} Conv3d/ConvTranspose3d site(s); first: {}".format(
            len(visitor.conv3d_lines),
            lines[first - 1].strip() if 0 < first <= len(lines) else "",
        )
        findings.append(Finding(path, first, "conv3d_subgraph", snippet, _NOTES["conv3d_subgraph"]))

    if allowed_patterns is not None:
        findings = [f for f in findings if f.pattern in allowed_patterns]
    return findings


def _iter_files(paths, on_error):
    for path in paths:
        if os.path.isfile(path) and path.endswith(".py"):
            yield path
        elif os.path.isdir(path):
            for root, dirs, names in os.walk(path, onerror=on_error):
                dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
                for name in sorted(names):
                    if name.endswith(".py"):
                        yield os.path.join(root, name)


def main(argv=None):
    # type: (Optional[List[str]]) -> int
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--patterns",
        default="",
        help="Comma-separated subset of pattern names to report (default: all).",
    )
    args = parser.parse_args(argv)
    for path in args.paths:
        if not (os.path.isdir(path) or (os.path.isfile(path) and path.endswith(".py"))):
            parser.error("expected an existing Python file or directory: {}".format(path))

    allowed = None  # type: Optional[Set[str]]
    if args.patterns.strip():
        requested = [p.strip() for p in args.patterns.split(",") if p.strip()]
        unknown = [p for p in requested if p not in _NOTES]
        if unknown:
            parser.error("unknown pattern(s): {} (valid: {})".format(
                ", ".join(unknown), ", ".join(sorted(_NOTES))
            ))
        allowed = set(requested)

    findings = []  # type: List[Finding]
    scanned = 0
    skipped = 0

    def walk_error(error):
        nonlocal skipped
        skipped += 1
        print("warning: skipped {}".format(error), file=sys.stderr)

    for path in _iter_files(args.paths, walk_error):
        result = scan_file(path, allowed_patterns=allowed)
        if result is None:
            skipped += 1
        else:
            scanned += 1
            findings.extend(result)
    findings.sort(key=lambda f: (f.file, f.line, f.pattern))

    if skipped or not scanned:
        print("Scanned {} Python file(s); {} skipped.".format(scanned, skipped), file=sys.stderr)
    print(
        "STATIC candidates only: they do not prove convolutions are hot or that channels-last "
        "will speed up this model. Confirm with a profiler first.",
        file=sys.stderr,
    )

    if args.json:
        print(json.dumps([f.to_dict() for f in findings], ensure_ascii=False, indent=2))
    elif not findings:
        print("No candidates found.")
    else:
        counts = {}  # type: Dict[str, int]
        for f in findings:
            counts[f.pattern] = counts.get(f.pattern, 0) + 1
        print(
            "{} static candidate(s): ".format(len(findings))
            + ", ".join("{} {}".format(n, p) for p, n in sorted(counts.items()))
        )
        print()
        for f in findings:
            print("{}:{}  [{}]".format(f.file, f.line, f.pattern))
            print("    {}".format(f.snippet))
            print("    note: {}\n".format(f.note))
    return 1 if skipped and not scanned else 0


if __name__ == "__main__":
    raise SystemExit(main())
