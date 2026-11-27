#!/usr/bin/env python3
"""Static scanner for PyTorch precision & mixed-precision candidates (ROCm/HIP & CUDA).

Scans Python source files (AST) and line-based YAML environment settings for:
  cuda_only_bf16_check          Direct CUDA-version gates around BF16 checks (PR2)
  tf32_disabled_explicitly      Explicit IEEE/highest/allow_tf32=False or HIPBLASLT_ALLOW_TF32=0 (PR1/PR6)
  tf32_env_config               Backend precision environment settings (PR1)
  fp32_precision_config         Explicit non-IEEE PyTorch fp32_precision settings (PR1)
  fp32_autocast_disabled        Disabled autocast or an explicit FP32 dtype (PR2/PR4)
  fp16_autocast_cue             Explicit FP16 autocast (PR2)
  bf16_autocast_cue             Explicit BF16 autocast (PR2)
  tf32_hook_missing_cache_tag   Constant TF32 get_float32_precision without a local tag assignment (PR3)

These are STATIC candidates only. They do not prove FP32 is a bottleneck or that
lowering precision meets convergence tolerance. Dynamic settings, cross-file tags,
and effective YAML inheritance require manual checks.

Usage:
    python3 scan_precision_candidates.py <file-or-dir> [...]
    python3 scan_precision_candidates.py --json <file-or-dir> [...]

Only the Python standard library is used (Python 3.6+ compatible).
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
import tokenize
from typing import Dict, Iterable, List, Optional, Tuple

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_DISCLAIMER = (
    "STATIC candidates only: they do not prove FP32 is a bottleneck, that TF32/BF16 is "
    "supported on the target build, or that lowering precision meets convergence tolerance."
)

_PRECISION_ENV_KEYS = {
    "HIPBLASLT_ALLOW_TF32",
    "MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32",
    "ROCBLAS_MATH_MODE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "TRITON_F32_DEFAULT",
    "DISABLE_ADDMM_CUDA_LT",
    "TORCHINDUCTOR_EMULATE_PRECISION_CASTS",
}

_FP32_PRECISION_TARGETS = {
    "torch.backends.fp32_precision",
    "torch.backends.cuda.matmul.fp32_precision",
    "torch.backends.cudnn.fp32_precision",
    "torch.backends.cudnn.conv.fp32_precision",
    "torch.backends.cudnn.rnn.fp32_precision",
}
_ALLOW_TF32_TARGETS = {"torch.backends.cuda.matmul.allow_tf32", "torch.backends.cudnn.allow_tf32"}
_AUTOCAST_POSITIONS = {
    "torch.autocast": (1, 2),
    "torch.amp.autocast": (1, 2),
    "torch.cuda.amp.autocast": (1, 0),
    "torch.cpu.amp.autocast": (1, 0),
}

_NOTES = {
    "cuda_only_bf16_check": (
        "BF16 capability check is gated by torch.version.cuda without a HIP branch in this scope; "
        "on ROCm/HIP this can silently fall back to FP32 (PR2)."
    ),
    "tf32_disabled_explicitly": (
        "TF32 is explicitly disabled or set to highest FP32 precision; verify whether this is "
        "an intentional accuracy guard (PR6) or an optimization opportunity (PR1)."
    ),
    "tf32_env_config": (
        "Backend precision/math environment setting detected; verify target ROCm/CUDA build "
        "support, single-variable ablation, and convergence tolerance (PR1)."
    ),
    "fp32_precision_config": (
        "Explicit PyTorch FP32 math policy; verify API/build support and actual backend selection (PR1)."
    ),
    "fp32_autocast_disabled": (
        "Autocast is disabled or requests FP32; inspect the actual input/compute dtype "
        "and whether conversion protects a sensitive layer (PR4)."
    ),
    "fp16_autocast_cue": (
        "FP16 autocast detected; verify GradScaler/overflow handling and whether native BF16 "
        "is preferable on the target hardware (PR2)."
    ),
    "bf16_autocast_cue": (
        "BF16 autocast is explicitly requested; check representative operator dispatch and "
        "training or inference tolerance on the target build (PR2)."
    ),
    "tf32_hook_missing_cache_tag": (
        "Constant TF32 precision helper without a recognized cache-tag assignment in this file; "
        "check whether it is installed and how the active cache is isolated (PR3)."
    ),
}


class Finding(object):
    __slots__ = ("file", "line", "pattern", "severity", "snippet", "note")

    def __init__(self, file, line, pattern, severity, snippet, note):
        # type: (str, int, str, str, str, str) -> None
        self.file = file
        self.line = line
        self.pattern = pattern
        self.severity = severity
        self.snippet = snippet
        self.note = note

    def to_dict(self):
        # type: () -> Dict[str, object]
        return {
            "file": self.file,
            "line": self.line,
            "pattern": self.pattern,
            "severity": self.severity,
            "snippet": self.snippet,
            "note": self.note,
        }


def _dotted(node):
    # type: (Optional[ast.AST]) -> str
    if node is None:
        return ""
    parts = []  # type: List[str]
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _str_const(node):
    # type: (Optional[ast.AST]) -> Optional[str]
    if node is None:
        return None
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return node.s
    return None


def _bool_const(node):
    # type: (Optional[ast.AST]) -> Optional[bool]
    if node is None:
        return None
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant) and isinstance(node.value, bool):
        return node.value
    return None


def _is_none_const(node):
    # type: (Optional[ast.AST]) -> bool
    if node is None:
        return False
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):
        return node.value is None
    if hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant):
        return node.value is None
    return False


def _resolve_dotted(node, aliases):
    # type: (Optional[ast.AST], Dict[str, str]) -> str
    name = _dotted(node)
    if not name:
        return ""
    head, sep, tail = name.partition(".")
    target = aliases.get(head)
    if target is None:
        return name
    return target + sep + tail if sep else target


def _kw_map(node):
    # type: (ast.Call) -> Dict[str, ast.AST]
    return {kw.arg: kw.value for kw in node.keywords if kw.arg}


def _argument(node, keywords, name, position):
    if name in keywords:
        return keywords[name]
    if position >= len(node.args) or any(isinstance(arg, ast.Starred) for arg in node.args[:position + 1]):
        return None
    return node.args[position]


def _scope_nodes(node):
    """Walk a function body without treating nested definitions as its code."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield from _scope_nodes(child)


def _has_bf16_call(node, aliases):
    return any(isinstance(item, ast.Call) and _resolve_dotted(item.func, aliases) == "torch.cuda.is_bf16_supported"
               for item in _scope_nodes(node))


def _cuda_version_gate(node, aliases):
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return any(_cuda_version_gate(value, aliases) for value in node.values)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        if isinstance(node.ops[0], (ast.IsNot, ast.NotEq)):
            left = node.left
            other = node.comparators[0]
            if _is_none_const(other) and _resolve_dotted(left, aliases) == "torch.version.cuda":
                return True
            if _is_none_const(left) and _resolve_dotted(other, aliases) == "torch.version.cuda":
                return True
            return False
    return _resolve_dotted(node, aliases) == "torch.version.cuda"


def _has_cuda_only_bf16_gate(function, aliases):
    nodes = list(_scope_nodes(function))
    if any(isinstance(node, ast.Attribute) and _resolve_dotted(node, aliases) == "torch.version.hip" for node in nodes):
        return False
    for node in nodes:
        if isinstance(node, ast.If) and _cuda_version_gate(node.test, aliases):
            if any(_has_bf16_call(stmt, aliases) for stmt in node.body):
                return True
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            for index, value in enumerate(node.values):
                if _cuda_version_gate(value, aliases) and any(_has_bf16_call(arg, aliases) for arg in node.values[index + 1:]):
                    return True
    return False


def _constant_string(node, constants):
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if not isinstance(node, ast.JoinedStr):
        return _str_const(node)
    parts = []
    for value in node.values:
        if isinstance(value, ast.FormattedValue) and value.conversion == -1 and value.format_spec is None:
            part = _constant_string(value.value, constants)
        else:
            part = _str_const(value)
        if part is None:
            return None
        parts.append(part)
    return "".join(parts)


class _Visitor(ast.NodeVisitor):
    def __init__(self, path, lines, tree):
        # type: (str, List[str], ast.Module) -> None
        self.path = path
        self.lines = lines
        self.findings = []  # type: List[Finding]
        self._has_cache_tag = False
        self._tf32_hook_lines = []  # type: List[int]
        self._aliases = {}  # type: Dict[str, str]
        self._constants = {}  # type: Dict[str, Optional[str]]
        nodes = list(ast.walk(tree))
        # Conservatively skip import aliases that are also bound elsewhere in the file.
        rebound = {node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))}
        rebound.update(node.arg for node in nodes if isinstance(node, ast.arg))
        rebound.update(node.name for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._constants[target.id] = _str_const(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                self._constants[node.target.id] = _str_const(node.value)
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname and alias.asname not in rebound:
                        self._aliases[alias.asname] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    if local_name in rebound:
                        continue
                    self._aliases[local_name] = "{}.{}".format(node.module, alias.name)

    def _emit(self, line, pattern, severity, snippet=None):
        # type: (int, str, str, Optional[str]) -> None
        if snippet is None:
            snippet = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        self.findings.append(Finding(self.path, line, pattern, severity, snippet, _NOTES[pattern]))

    def visit_FunctionDef(self, node):
        # type: (ast.FunctionDef) -> None
        if node.name == "get_float32_precision":
            body = node.body[1:] if ast.get_docstring(node) is not None else node.body
            if body and isinstance(body[-1], ast.Return):
                returns = [item for item in _scope_nodes(node) if isinstance(item, ast.Return)]
                if returns and all(_constant_string(item.value, self._constants) in ("tf32", "'tf32'", '"tf32"')
                                   for item in returns):
                    self._tf32_hook_lines.append(node.lineno)
        if _has_cuda_only_bf16_gate(node, self._aliases):
            self._emit(node.lineno, "cuda_only_bf16_check", "high")
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node):
        condition = _bool_const(node.test)
        if condition is None:
            self.generic_visit(node)
        else:
            for statement in node.body if condition else node.orelse:
                self.visit(statement)

    def visit_Assign(self, node):
        # type: (ast.Assign) -> None
        val_bool = _bool_const(node.value)
        val_str = _str_const(node.value)
        has_tag_value = val_str != "" and not _is_none_const(node.value)
        for tgt in node.targets:
            tname = _resolve_dotted(tgt, self._aliases)
            if tname in _ALLOW_TF32_TARGETS and val_bool is False:
                self._emit(node.lineno, "tf32_disabled_explicitly", "medium")
            elif tname in _FP32_PRECISION_TARGETS:
                if val_str == "ieee":
                    self._emit(node.lineno, "tf32_disabled_explicitly", "medium")
                elif val_str == "tf32" or (tname == "torch.backends.cuda.matmul.fp32_precision" and val_str == "bfx9"):
                    self._emit(node.lineno, "fp32_precision_config", "info")
            elif tname == "torch.compiler.config.cache_key_tag" and has_tag_value:
                self._has_cache_tag = True
            elif isinstance(tgt, ast.Subscript):
                base = _resolve_dotted(tgt.value, self._aliases)
                if base in ("os.environ", "environ"):
                    sl = tgt.slice.value if hasattr(ast, "Index") and isinstance(tgt.slice, ast.Index) else tgt.slice
                    key = _str_const(sl)
                    if key == "TORCH_COMPILE_CACHE_KEY_TAG" and has_tag_value:
                        self._has_cache_tag = True
                    elif key in _PRECISION_ENV_KEYS:
                        if key == "HIPBLASLT_ALLOW_TF32" and val_str == "0":
                            self._emit(node.lineno, "tf32_disabled_explicitly", "info")
                        else:
                            self._emit(node.lineno, "tf32_env_config", "info")
        self.generic_visit(node)

    def visit_Call(self, node):
        # type: (ast.Call) -> None
        name = _resolve_dotted(node.func, self._aliases)
        kw = _kw_map(node)

        if name == "torch.set_float32_matmul_precision":
            arg = _argument(node, kw, "precision", 0)
            if _str_const(arg) == "highest":
                self._emit(node.lineno, "tf32_disabled_explicitly", "medium")
        elif name in _AUTOCAST_POSITIONS:
            dtype_pos, enabled_pos = _AUTOCAST_POSITIONS[name]
            enabled = _bool_const(_argument(node, kw, "enabled", enabled_pos))
            dtype = _resolve_dotted(_argument(node, kw, "dtype", dtype_pos), self._aliases)
            if enabled is False or dtype in ("torch.float32", "torch.float", "float32"):
                self._emit(node.lineno, "fp32_autocast_disabled", "info")
            elif dtype in ("torch.float16", "torch.half", "float16"):
                self._emit(node.lineno, "fp16_autocast_cue", "info")
            elif dtype in ("torch.bfloat16", "bfloat16"):
                self._emit(node.lineno, "bf16_autocast_cue", "info")

        self.generic_visit(node)


def scan_python_file(path):
    # type: (str) -> Tuple[Optional[List[Finding]], Optional[str]]
    try:
        with tokenize.open(path) as fh:
            source = fh.read()
        tree = ast.parse(source, filename=path)
        lines = source.splitlines()
        visitor = _Visitor(path, lines, tree)
        visitor.visit(tree)
    except (OSError, SyntaxError, ValueError, RecursionError) as exc:
        reason = str(exc)
        print("warning: skipped {}: {}".format(path, reason), file=sys.stderr)
        return None, reason

    findings = list(visitor.findings)
    if visitor._tf32_hook_lines and not visitor._has_cache_tag:
        line = visitor._tf32_hook_lines[0]
        snippet = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        findings.append(Finding(path, line, "tf32_hook_missing_cache_tag", "high", snippet, _NOTES["tf32_hook_missing_cache_tag"]))
    return findings, None


_RE_YAML_ENV = re.compile(
    r"""^\s*(?P<key>HIPBLASLT_ALLOW_TF32|MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32|ROCBLAS_MATH_MODE|"""
    r"""TORCH_ALLOW_TF32_CUBLAS_OVERRIDE|TRITON_F32_DEFAULT|DISABLE_ADDMM_CUDA_LT|"""
    r"""TORCHINDUCTOR_EMULATE_PRECISION_CASTS)\s*:\s*(?P<val>[^#\n]+)"""
)
_RE_YAML_BLOCK_SCALAR = re.compile(
    r"""^\s*(?:-\s+[^#\n]*:\s*|[^#\n]+:\s*|-\s+)[|>](?:[+-]\d*|\d+[+-]?)?\s*(?:#.*)?$"""
)


def scan_yaml_config(path):
    # type: (str) -> Tuple[Optional[List[Finding]], Optional[str]]
    try:
        with open(path, "rb") as fh:
            data = fh.read(_MAX_CONFIG_BYTES + 1)
        if len(data) > _MAX_CONFIG_BYTES:
            raise ValueError("YAML exceeds the {} byte scan limit".format(_MAX_CONFIG_BYTES))
        lines = data.decode("utf-8-sig").splitlines()
    except (OSError, ValueError) as exc:
        reason = str(exc)
        print("warning: skipped {}: {}".format(path, reason), file=sys.stderr)
        return None, reason

    findings = []  # type: List[Finding]
    block_indent = None  # type: Optional[int]
    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.lstrip(" \t")
        indent = len(raw) - len(stripped)
        if block_indent is not None:
            if not stripped or indent > block_indent:
                continue
            block_indent = None
        if not stripped or stripped.startswith("#"):
            continue
        if _RE_YAML_BLOCK_SCALAR.match(raw):
            block_indent = indent
            continue
        m = _RE_YAML_ENV.match(raw)
        if not m:
            continue
        key = m.group("key")
        val = m.group("val").strip().strip("'\"")
        if key == "HIPBLASLT_ALLOW_TF32" and val == "0":
            findings.append(Finding(path, lineno, "tf32_disabled_explicitly", "info", stripped, _NOTES["tf32_disabled_explicitly"]))
        else:
            findings.append(Finding(path, lineno, "tf32_env_config", "info", stripped, _NOTES["tf32_env_config"]))
    return findings, None


def _iter_targets(paths, on_error):
    # type: (Iterable[str], object) -> Iterable[Tuple[str, str]]
    for target in paths:
        if os.path.isfile(target):
            low = target.lower()
            if low.endswith(".py"):
                yield ("py", target)
            elif low.endswith((".yaml", ".yml")):
                yield ("yaml", target)
        elif os.path.isdir(target):
            for root, dirs, files in os.walk(target, onerror=on_error):
                dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
                for name in sorted(files):
                    low = name.lower()
                    full = os.path.join(root, name)
                    if low.endswith(".py"):
                        yield ("py", full)
                    elif low.endswith((".yaml", ".yml")):
                        yield ("yaml", full)


def main(argv=None):
    # type: (Optional[List[str]]) -> int
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Python files, YAML configs, or directories to scan.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    for p in args.paths:
        if not (os.path.isdir(p) or (os.path.isfile(p) and p.lower().endswith((".py", ".yaml", ".yml")))):
            parser.error("expected an existing Python/YAML file or directory: {}".format(p))

    findings = []  # type: List[Finding]
    scanned = 0
    skipped_files = []  # type: List[Dict[str, str]]

    def walk_error(err):
        name = getattr(err, "filename", None) or "<walk>"
        reason = str(err)
        skipped_files.append({"file": str(name), "reason": reason})
        print("warning: skipped {}".format(reason), file=sys.stderr)

    for kind, path in _iter_targets(args.paths, walk_error):
        res, reason = scan_python_file(path) if kind == "py" else scan_yaml_config(path)
        if res is None:
            skipped_files.append({"file": path, "reason": reason or "unknown error"})
        else:
            scanned += 1
            findings.extend(res)

    findings.sort(key=lambda f: (f.file, f.line, f.pattern))
    skipped = len(skipped_files)

    if skipped or not scanned:
        print("Files: {} scanned, {} skipped.".format(scanned, skipped), file=sys.stderr)
    if args.json:
        payload = {
            "disclaimer": _DISCLAIMER,
            "files_scanned": scanned,
            "files_skipped": skipped,
            "skipped_files": skipped_files,
            "findings": [f.to_dict() for f in findings],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif not scanned:
        print("No supported files scanned.")
    elif not findings:
        print("No static precision candidates found.")
    else:
        print("{} static candidate(s):\n".format(len(findings)))
        for f in findings:
            print("{}:{} [{}] ({}) {}".format(f.file, f.line, f.severity.upper(), f.pattern, f.snippet))
            print("    note: {}\n".format(f.note))

    print(_DISCLAIMER, file=sys.stderr)
    return 1 if skipped or not scanned else 0


if __name__ == "__main__":
    raise SystemExit(main())
