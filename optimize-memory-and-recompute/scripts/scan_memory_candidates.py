#!/usr/bin/env python3
"""Static scanner for PyTorch training memory, recomputation, and GC candidates.

This is a STATIC scanner. It scans Python source files (AST, with regex fallback
for newer Python syntax on older runtimes) and YAML/JSON config files for:
  - Activation checkpointing (AC) sites: with_cp=True, gradient_checkpointing,
    checkpoint_wrapper, apply_activation_checkpointing, torch.utils.checkpoint.*
    (flagging use_reentrant=True as high severity).
  - Explicit training-time KV cache toggles: use_cache=True (downgraded to info
    inside inference/generate/eval helpers).
  - Hot-path torch.cuda.empty_cache() and gc.collect() calls (ignoring
    gc.collect() paired with gc.freeze() inside a freeze helper).

It does NOT measure baseline peak memory headroom (max_memory_allocated /
max_memory_reserved), prove OOM safety under worst-case shapes, or verify
multi-rank GC jitter or host RSS leaks. Always measure peak memory and
step-time distribution before disabling AC or freezing GC.

Usage:
    python3 scan_memory_candidates.py <file-or-dir> [<file-or-dir> ...]
    python3 scan_memory_candidates.py --json <file-or-dir> ...
    python3 scan_memory_candidates.py --min-severity medium <file-or-dir> ...
    python3 scan_memory_candidates.py --no-configs <file-or-dir> ...

Only the Python standard library is used (compatible with Python 3.6+).
"""

import argparse
import ast
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple


_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
}
_MAX_CONFIG_BYTES = 2 * 1024 * 1024

_INFER_FUNC_TOKENS = (
    "generate",
    "infer",
    "inference",
    "eval",
    "evaluate",
    "sample",
    "decode",
    "predict",
    "rollout",
)

_CHECKPOINT_WRAPPER_NAMES = {
    "checkpoint_wrapper",
    "apply_activation_checkpointing",
    "ptd_checkpoint_wrapper",
}
_CHECKPOINT_ENABLE_METHODS = {
    "gradient_checkpointing_enable",
    "enable_gradient_checkpointing",
}
_CHECKPOINT_BOOL_METHODS = {
    "set_gradient_checkpointing",
    "set_grad_checkpointing",
}
_CHECKPOINT_DOTTED_CALLS = {
    "torch.utils.checkpoint.checkpoint",
    "torch.utils.checkpoint.checkpoint_sequential",
    "utils.checkpoint.checkpoint",
    "utils.checkpoint.checkpoint_sequential",
    "checkpoint.checkpoint",
    "checkpoint.checkpoint_sequential",
    "cp.checkpoint",
    "cp.checkpoint_sequential",
    "ckpt.checkpoint",
    "ckpt.checkpoint_sequential",
    "torch_checkpoint",
}
_EMPTY_CACHE_PREFIXES = (
    "torch.cuda.",
    "cuda.",
    "torch.npu.",
    "npu.",
    "torch.xpu.",
    "xpu.",
    "torch.mps.",
    "mps.",
    "torch.mlu.",
    "mlu.",
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


def _const_str(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    if isinstance(node, ast.Str):
        return node.s
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_infer_func_name(name: str) -> bool:
    lower = name.lower()
    for token in _INFER_FUNC_TOKENS:
        if token in lower:
            return True
    return False


def _subtree_calls_gc_freeze(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            dotted = _dotted_name(child.func)
            if dotted == "gc.freeze" or dotted.endswith(".freeze"):
                if dotted == "gc.freeze":
                    return True
    return False


def _dict_has_reentrant_true(node: Optional[ast.AST]) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    for k, v in zip(node.keys, node.values):
        if _const_str(k) == "use_reentrant" and _const_bool(v) is True:
            return True
    return False


class MemoryCandidateVisitor(ast.NodeVisitor):
    def __init__(self, path: str, lines: List[str]) -> None:
        self.path = path
        self.lines = lines
        self.findings: List[Finding] = []
        self.loop_depth = 0
        self.func_stack: List[str] = []
        self.in_gc_freeze_func = 0
        self.imported_checkpoint_names = set()  # type: set

    def _snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()
        return ""

    def _add(self, node: ast.AST, pattern: str, severity: str, note: str) -> None:
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        self.findings.append(
            Finding(
                file=self.path,
                line=lineno,
                col=col,
                pattern=pattern,
                severity=severity,
                snippet=self._snippet(lineno),
                note=note,
            )
        )

    def _in_infer_func(self) -> bool:
        for fn in self.func_stack:
            if _is_infer_func_name(fn):
                return True
        return False

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if "checkpoint" in mod:
            for alias in node.names:
                if alias.name in ("checkpoint", "checkpoint_sequential", "checkpoint_wrapper"):
                    self.imported_checkpoint_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.func_stack.append(node.name)
        has_freeze = _subtree_calls_gc_freeze(node)
        if has_freeze:
            self.in_gc_freeze_func += 1
        self.generic_visit(node)
        if has_freeze:
            self.in_gc_freeze_func -= 1
        self.func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.func_stack.append(node.name)
        has_freeze = _subtree_calls_gc_freeze(node)
        if has_freeze:
            self.in_gc_freeze_func += 1
        self.generic_visit(node)
        if has_freeze:
            self.in_gc_freeze_func -= 1
        self.func_stack.pop()

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Assign(self, node: ast.Assign) -> None:
        val_bool = _const_bool(node.value)
        if val_bool is True:
            for tgt in node.targets:
                name = ""
                if isinstance(tgt, ast.Name):
                    name = tgt.id
                elif isinstance(tgt, ast.Attribute):
                    name = tgt.attr
                if name == "with_cp":
                    self._add(
                        node,
                        "ac_with_cp_true",
                        "medium",
                        "with_cp=True enables module activation checkpointing (M1); check baseline peak memory headroom before disabling.",
                    )
                elif name in ("gradient_checkpointing", "use_checkpoint", "use_checkpointing"):
                    self._add(
                        node,
                        "ac_checkpoint_call",
                        "medium",
                        "{}=True enables activation checkpointing (M1/M2); confirm peak memory headroom or evaluate selective AC.".format(
                            name
                        ),
                    )
                elif name == "use_cache":
                    sev = "info" if self._in_infer_func() else "high"
                    note = (
                        "use_cache=True inside inference/generation helper (keep enabled for autoregressive decoding)."
                        if sev == "info"
                        else "use_cache=True allocates KV cache (M3); disable on training prefill paths to save memory."
                    )
                    self._add(node, "kv_cache_enabled", sev, note)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for k, v in zip(node.keys, node.values):
            key = _const_str(k)
            val = _const_bool(v)
            if key == "with_cp" and val is True:
                self._add(
                    v,
                    "ac_with_cp_true",
                    "medium",
                    "Dict config sets 'with_cp': True (M1); verify baseline peak memory headroom before disabling.",
                )
            elif key in ("gradient_checkpointing", "use_checkpoint", "use_checkpointing") and val is True:
                self._add(
                    v,
                    "ac_checkpoint_call",
                    "medium",
                    "Dict config sets '{}': True (M1/M2); confirm peak memory headroom or evaluate selective AC.".format(
                        key
                    ),
                )
            elif key == "use_cache" and val is True:
                sev = "info" if self._in_infer_func() else "high"
                note = (
                    "Dict config sets 'use_cache': True inside inference/generation helper."
                    if sev == "info"
                    else "Dict config sets 'use_cache': True (M3); disable on training prefill paths to save memory."
                )
                self._add(v, "kv_cache_enabled", sev, note)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name(node.func)
        short = _call_short_name(node)
        kwargs = _kwarg_map(node)

        # 1. Generic kwargs on any Call (covers dict(with_cp=True), ResNet(with_cp=True), model(..., use_cache=True))
        if "with_cp" in kwargs and _const_bool(kwargs["with_cp"]) is True:
            self._add(
                node,
                "ac_with_cp_true",
                "medium",
                "with_cp=True enables module activation checkpointing (M1); measure peak memory headroom before disabling.",
            )

        if "use_cache" in kwargs and _const_bool(kwargs["use_cache"]) is True:
            sev = "info" if self._in_infer_func() else "high"
            note = (
                "use_cache=True inside inference/generation helper (keep enabled for autoregressive decoding)."
                if sev == "info"
                else "use_cache=True allocates KV cache (M3); disable on training prefill paths to save activation memory."
            )
            self._add(node, "kv_cache_enabled", sev, note)

        if (
            "gradient_checkpointing" in kwargs
            and _const_bool(kwargs["gradient_checkpointing"]) is True
        ):
            self._add(
                node,
                "ac_checkpoint_call",
                "medium",
                "gradient_checkpointing=True enables activation checkpointing (M1/M2); check peak memory headroom or selective AC.",
            )

        # 2. Activation checkpointing API calls
        if short in _CHECKPOINT_ENABLE_METHODS:
            gc_kwargs = kwargs.get("gradient_checkpointing_kwargs")
            if _dict_has_reentrant_true(gc_kwargs) or _const_bool(kwargs.get("use_reentrant")) is True:
                self._add(
                    node,
                    "ac_reentrant_true",
                    "high",
                    "{}() with use_reentrant=True (M1/M2); prefer use_reentrant=False for safer DDP/compile interop and verify peak memory headroom.".format(
                        short
                    ),
                )
            else:
                self._add(
                    node,
                    "ac_checkpoint_call",
                    "medium",
                    "{}() enables activation checkpointing (M1/M2); check peak memory headroom before disabling or tuning.".format(
                        short
                    ),
                )
        elif short in _CHECKPOINT_BOOL_METHODS:
            arg_val = _const_bool(node.args[0]) if node.args else _const_bool(kwargs.get("enable"))
            if arg_val is True or (not node.args and "enable" not in kwargs):
                self._add(
                    node,
                    "ac_checkpoint_call",
                    "medium",
                    "{}(True) enables activation checkpointing (M1/M2); check peak memory headroom before reducing.".format(
                        short
                    ),
                )
        elif short in _CHECKPOINT_WRAPPER_NAMES:
            if _const_bool(kwargs.get("use_reentrant")) is True:
                self._add(
                    node,
                    "ac_reentrant_true",
                    "high",
                    "{}() with use_reentrant=True (M1/M2); prefer use_reentrant=False and verify AC/SAC scope.".format(
                        short
                    ),
                )
            else:
                self._add(
                    node,
                    "ac_checkpoint_call",
                    "medium",
                    "{}() wraps modules for activation checkpointing (M1/M2); verify peak memory headroom or selective AC policy.".format(
                        short
                    ),
                )
        elif (
            dotted in _CHECKPOINT_DOTTED_CALLS
            or (
                short in ("checkpoint", "checkpoint_sequential")
                and (
                    short in self.imported_checkpoint_names
                    or "use_reentrant" in kwargs
                    or "preserve_rng_state" in kwargs
                    or "context_fn" in kwargs
                    or len(node.args) >= 2
                )
            )
        ):
            if _const_bool(kwargs.get("use_reentrant")) is True:
                self._add(
                    node,
                    "ac_reentrant_true",
                    "high",
                    "checkpoint() called with use_reentrant=True (M1/M2); prefer use_reentrant=False unless legacy hooks require reentrant mode.",
                )
            else:
                self._add(
                    node,
                    "ac_checkpoint_call",
                    "medium",
                    "Activation checkpoint call '{}' (M1/M2); verify baseline peak memory headroom before skipping layers or switching to SAC.".format(
                        dotted or short
                    ),
                )

        # 3. Hot-path empty_cache() and gc.collect()
        if short == "empty_cache" and (
            any(dotted.startswith(pfx) for pfx in _EMPTY_CACHE_PREFIXES)
            or dotted.endswith(".cuda.empty_cache")
            or dotted.endswith(".npu.empty_cache")
        ):
            sev = "high" if self.loop_depth > 0 else "medium"
            ctx = "inside a loop" if self.loop_depth > 0 else "outside an explicit loop (check if called per step)"
            self._add(
                node,
                "hotpath_empty_cache",
                sev,
                "{}() called {} (M6); frequent cache flushes trigger allocator stalls and cross-rank jitter.".format(
                    dotted or short, ctx
                ),
            )

        if dotted == "gc.collect" and self.in_gc_freeze_func == 0:
            sev = "medium" if self.loop_depth > 0 else "info"
            ctx = "inside a loop" if self.loop_depth > 0 else "outside a loop"
            self._add(
                node,
                "hotpath_gc_collect",
                sev,
                "gc.collect() called {} (M6); evaluate warmup-time multi-rank aligned gc.collect(2) + gc.freeze() and verify host RSS stability.".format(
                    ctx
                ),
            )

        self.generic_visit(node)


_RE_WITH_CP_TRUE = re.compile(r"""(?:\bwith_cp\s*=|['"]with_cp['"]\s*:)\s*True\b""")
_RE_GC_ENABLE = re.compile(
    r"""\b(?:gradient_checkpointing_enable|enable_gradient_checkpointing|checkpoint_wrapper|apply_activation_checkpointing|ptd_checkpoint_wrapper)\s*\("""
)
_RE_CKPT_CALL = re.compile(
    r"""\b(?:torch\.utils\.checkpoint|utils\.checkpoint|checkpoint|cp|ckpt)\.checkpoint(?:_sequential)?\s*\("""
)
_RE_REENTRANT_TRUE = re.compile(r"""\buse_reentrant\s*=\s*True\b|['"]use_reentrant['"]\s*:\s*True\b""")
_RE_USE_CACHE_TRUE = re.compile(r"""(?:\buse_cache\s*=|['"]use_cache['"]\s*:)\s*True\b""")
_RE_EMPTY_CACHE = re.compile(r"""\b(?:torch\.)?(?:cuda|npu|xpu|mps|mlu)\.empty_cache\s*\(""")
_RE_GC_COLLECT = re.compile(r"""\bgc\.collect\s*\(""")
_RE_GC_FREEZE = re.compile(r"""\bgc\.freeze\s*\(""")
_RE_DEF_LINE = re.compile(r"""^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(""")
_RE_LOOP_LINE = re.compile(r"""^\s*(?:async\s+)?(?:for|while)\b""")


def _scan_python_regex_fallback(path: str, text: str) -> List[Finding]:
    """Fallback line scanner when host Python cannot ast.parse newer Python syntax."""
    findings: List[Finding] = []
    lines = text.splitlines()
    has_gc_freeze = bool(_RE_GC_FREEZE.search(text))
    current_func = ""
    current_func_indent = -1
    loop_indents: List[int] = []

    for lineno, raw in enumerate(lines, start=1):
        code = raw.split("#", 1)[0]
        stripped = code.strip()
        if not stripped:
            continue
        indent = len(raw) - len(raw.lstrip(" \t"))
        while loop_indents and indent <= loop_indents[-1]:
            loop_indents.pop()
        if current_func_indent >= 0 and indent <= current_func_indent:
            current_func = ""
            current_func_indent = -1

        m_def = _RE_DEF_LINE.match(code)
        if m_def:
            current_func = m_def.group(1)
            current_func_indent = indent
            continue
        if _RE_LOOP_LINE.match(code):
            loop_indents.append(indent)

        in_loop = bool(loop_indents)
        in_infer = bool(current_func) and _is_infer_func_name(current_func)

        if _RE_WITH_CP_TRUE.search(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="ac_with_cp_true",
                    severity="medium",
                    snippet=stripped,
                    note="with_cp=True (regex fallback, M1); check baseline peak memory headroom before disabling.",
                )
            )

        if _RE_GC_ENABLE.search(code) or _RE_CKPT_CALL.search(code):
            if _RE_REENTRANT_TRUE.search(code):
                findings.append(
                    Finding(
                        file=path,
                        line=lineno,
                        col=0,
                        pattern="ac_reentrant_true",
                        severity="high",
                        snippet=stripped,
                        note="Activation checkpointing with use_reentrant=True (regex fallback, M1/M2).",
                    )
                )
            else:
                findings.append(
                    Finding(
                        file=path,
                        line=lineno,
                        col=0,
                        pattern="ac_checkpoint_call",
                        severity="medium",
                        snippet=stripped,
                        note="Activation checkpointing call (regex fallback, M1/M2); verify peak memory headroom.",
                    )
                )

        if _RE_USE_CACHE_TRUE.search(code):
            sev = "info" if in_infer else "high"
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="kv_cache_enabled",
                    severity=sev,
                    snippet=stripped,
                    note="use_cache=True (regex fallback, M3); disable on training prefill paths.",
                )
            )

        if _RE_EMPTY_CACHE.search(code):
            sev = "high" if in_loop else "medium"
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="hotpath_empty_cache",
                    severity=sev,
                    snippet=stripped,
                    note="empty_cache() call (regex fallback, M6); check call frequency for allocator stalls.",
                )
            )

        if _RE_GC_COLLECT.search(code) and not has_gc_freeze:
            sev = "medium" if in_loop else "info"
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="hotpath_gc_collect",
                    severity=sev,
                    snippet=stripped,
                    note="gc.collect() call (regex fallback, M6); consider warmup-time multi-rank gc.freeze().",
                )
            )

    return findings


def scan_python_file(path: str) -> List[Finding]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError as exc:
        sys.stderr.write("[warn] cannot read {}: {}\n".format(path, exc))
        return []

    tree = None
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        cleaned = re.sub(
            r"^\s*from\s+__future__\s+import\s+annotations\s*$",
            "",
            text,
            flags=re.MULTILINE,
        )
        try:
            tree = ast.parse(cleaned, filename=path)
        except SyntaxError:
            return _scan_python_regex_fallback(path, text)

    visitor = MemoryCandidateVisitor(path, text.splitlines())
    visitor.visit(tree)
    return visitor.findings


def _walk_json(obj: object, path_prefix: str = "") -> Iterable[Tuple[str, object]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = "{}.{}".format(path_prefix, k) if path_prefix else str(k)
            yield full, v
            for sub in _walk_json(v, full):
                yield sub
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            full = "{}[{}]".format(path_prefix, idx)
            for sub in _walk_json(item, full):
                yield sub


def scan_json_config(path: str) -> List[Finding]:
    try:
        if os.path.getsize(path) > _MAX_CONFIG_BYTES:
            return []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []

    findings: List[Finding] = []
    for key_path, val in _walk_json(data):
        leaf = key_path.rsplit(".", 1)[-1]
        if leaf == "with_cp" and val is True:
            findings.append(
                Finding(
                    file=path,
                    line=1,
                    col=0,
                    pattern="ac_with_cp_true",
                    severity="medium",
                    snippet="{}: true".format(key_path),
                    note="JSON config enables with_cp=true (M1); verify peak memory headroom before disabling.",
                )
            )
        elif leaf in ("gradient_checkpointing", "use_checkpoint") and val is True:
            findings.append(
                Finding(
                    file=path,
                    line=1,
                    col=0,
                    pattern="ac_checkpoint_call",
                    severity="medium",
                    snippet="{}: true".format(key_path),
                    note="JSON config enables {}=true (M1/M2); verify peak memory headroom or SAC.".format(
                        leaf
                    ),
                )
            )
        elif leaf == "use_cache" and val is True:
            findings.append(
                Finding(
                    file=path,
                    line=1,
                    col=0,
                    pattern="kv_cache_enabled",
                    severity="medium",
                    snippet="{}: true".format(key_path),
                    note="JSON config sets use_cache=true (M3); override to false for training prefill if used in training.",
                )
            )
    return findings


_RE_YAML_WITH_CP = re.compile(r"^\s*with_cp\s*:\s*(?:true|True|TRUE)\b")
_RE_YAML_GC = re.compile(r"^\s*(?:gradient_checkpointing|use_checkpoint)\s*:\s*(?:true|True|TRUE)\b")
_RE_YAML_USE_CACHE = re.compile(r"^\s*use_cache\s*:\s*(?:true|True|TRUE)\b")


def scan_yaml_config(path: str) -> List[Finding]:
    try:
        if os.path.getsize(path) > _MAX_CONFIG_BYTES:
            return []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except OSError:
        return []

    findings: List[Finding] = []
    for lineno, raw in enumerate(lines, start=1):
        code = raw.split("#", 1)[0]
        if not code.strip():
            continue
        if _RE_YAML_WITH_CP.match(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="ac_with_cp_true",
                    severity="medium",
                    snippet=code.strip(),
                    note="YAML config sets with_cp: true (M1); check baseline peak memory headroom before disabling.",
                )
            )
        elif _RE_YAML_GC.match(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="ac_checkpoint_call",
                    severity="medium",
                    snippet=code.strip(),
                    note="YAML config enables gradient_checkpointing (M1/M2); check peak memory headroom or SAC.",
                )
            )
        elif _RE_YAML_USE_CACHE.match(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="kv_cache_enabled",
                    severity="medium",
                    snippet=code.strip(),
                    note="YAML config sets use_cache: true (M3); disable during training prefill if not an inference-only config.",
                )
            )
    return findings


def _iter_targets(paths: List[str], include_configs: bool) -> Iterable[Tuple[str, str]]:
    for target in paths:
        if os.path.isfile(target):
            low = target.lower()
            if low.endswith(".py"):
                yield ("py", target)
            elif include_configs and low.endswith(".json"):
                yield ("json", target)
            elif include_configs and (low.endswith(".yaml") or low.endswith(".yml")):
                yield ("yaml", target)
            continue
        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in sorted(dirs) if d not in _SKIP_DIRS]
            for name in sorted(files):
                full = os.path.join(root, name)
                low = name.lower()
                if low.endswith(".py"):
                    yield ("py", full)
                elif include_configs and low.endswith(".json"):
                    yield ("json", full)
                elif include_configs and (low.endswith(".yaml") or low.endswith(".yml")):
                    yield ("yaml", full)


_SEV_ORDER = {"info": 0, "medium": 1, "high": 2}


def scan_paths(paths: List[str], include_configs: bool = True) -> List[Finding]:
    all_findings: List[Finding] = []
    for kind, path in _iter_targets(paths, include_configs=include_configs):
        if kind == "py":
            all_findings.extend(scan_python_file(path))
        elif kind == "json":
            all_findings.extend(scan_json_config(path))
        elif kind == "yaml":
            all_findings.extend(scan_yaml_config(path))
    return all_findings


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Static scanner for PyTorch activation checkpointing, KV cache, and hot-path GC/empty_cache candidates."
    )
    parser.add_argument("paths", nargs="+", help="Python files, config files, or directories to scan.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON output.")
    parser.add_argument(
        "--min-severity",
        choices=("info", "medium", "high"),
        default="info",
        help="Minimum severity level to report (default: info).",
    )
    parser.add_argument(
        "--no-configs",
        action="store_true",
        help="Skip JSON and YAML config files; scan Python files only.",
    )
    args = parser.parse_args(argv)

    missing = [p for p in args.paths if not os.path.exists(p)]
    if missing:
        sys.stderr.write("[error] path not found: {}\n".format(", ".join(missing)))
        return 2

    findings = scan_paths(args.paths, include_configs=not args.no_configs)
    min_rank = _SEV_ORDER[args.min_severity]
    filtered = [f for f in findings if _SEV_ORDER[f.severity] >= min_rank]

    sys.stderr.write(
        "[static-scan] Results are STATIC candidates only; they do NOT measure peak memory headroom, "
        "guarantee OOM safety under worst-case shapes, or prove multi-rank GC jitter.\n"
    )

    if args.as_json:
        print(json.dumps([f.to_dict() for f in filtered], indent=2, ensure_ascii=False))
        return 0

    if not filtered:
        print("No static memory/recompute candidates found.")
        return 0

    for item in filtered:
        print(
            "{}:{}: [{}] ({}) {}".format(
                item.file, item.line, item.severity.upper(), item.pattern, item.snippet
            )
        )
        print("    -> {}".format(item.note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
