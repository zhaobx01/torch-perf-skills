#!/usr/bin/env python3
"""Static scanner for PyTorch training memory, recomputation, and GC candidates.

This is a STATIC scanner. It scans Python source files (AST, with tokenized
regex fallback for newer Python syntax on older runtimes) and YAML/JSON config
files for:
  - Activation checkpointing (AC) sites: with_cp=True, gradient_checkpointing,
    use_checkpoint, use_checkpointing, checkpoint_wrapper,
    apply_activation_checkpointing, torch.utils.checkpoint.*
    (flagging explicit reentrant options as high severity).
  - Literal KV cache settings and parameter defaults: use_cache=True
    (Python sites without a clear training-name hint are informational).
  - torch.cuda.empty_cache() and gc.collect() candidates, with explicit
    PyTorch and gc import aliases recognized in AST mode.

The scanner does NOT prove hot paths, speedups, OOM safety, or GC jitter.
Check the real call path, baseline peak memory, and step-time distribution
before disabling AC or freezing GC. YAML support covers single-line boolean
values in block-style mappings.

Usage:
    python3 scan_memory_candidates.py <file-or-dir> [<file-or-dir> ...]
    python3 scan_memory_candidates.py --json <file-or-dir> ...
    python3 scan_memory_candidates.py --min-severity medium <file-or-dir> ...
    python3 scan_memory_candidates.py --no-configs <file-or-dir> ...

Only the Python standard library is used (compatible with Python 3.6+).
JSON output includes ``disclaimer``, ``files_scanned``, ``files_skipped``,
``skipped_files``, and ``findings``. Exit 1 indicates a partial or empty scan,
and exit 2 indicates invalid CLI arguments.
"""

import argparse
import ast
import io
import json
import os
import re
import sys
import tokenize
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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
_DISCLAIMER = (
    "[static-scan] Results are STATIC candidates only; they do NOT prove hot paths, speedups, "
    "peak memory headroom, OOM safety, or multi-rank GC jitter."
)

_AC_BOOL_KEYS = ("gradient_checkpointing", "use_checkpoint", "use_checkpointing")

_INFER_FUNC_TOKENS = {
    "generate",
    "infer",
    "inference",
    "eval",
    "evaluate",
    "sample",
    "decode",
    "predict",
    "rollout",
}
_TRAIN_FUNC_TOKENS = {"train", "training"}

_CHECKPOINT_WRAPPER_NAMES = {
    "checkpoint_wrapper",
    "apply_activation_checkpointing",
    "ptd_checkpoint_wrapper",
}
_CHECKPOINT_WRAPPER_MODULE = "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
_CHECKPOINT_REENTRANT_ENUM = _CHECKPOINT_WRAPPER_MODULE + ".CheckpointImpl.REENTRANT"
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
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant) and isinstance(node.value, bool):
        return node.value
    return None


def _const_str(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return node.s
    return None


def _kv_severity(func_names: Sequence[str]) -> str:
    for name in reversed(func_names):
        words = set(name.lower().split("_"))
        if words & _INFER_FUNC_TOKENS:
            return "info"
        if words & _TRAIN_FUNC_TOKENS:
            return "medium"
    return "info"


def _is_gc_setup_func(name: str) -> bool:
    return "warmup" in name.lower().split("_")


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
        self.bindings: List[Dict[str, Optional[str]]] = [{}]
        self.class_scope_indices = set()
        self.warmup_collects: Dict[ast.Call, Finding] = {}
        self.gc_freezes = set()

    def _snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()
        return ""

    def _add(self, node: ast.AST, pattern: str, severity: str, note: str) -> Finding:
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        finding = Finding(
            file=self.path,
            line=lineno,
            col=col,
            pattern=pattern,
            severity=severity,
            snippet=self._snippet(lineno),
            note=note,
        )
        self.findings.append(finding)
        return finding

    def _add_kv_candidate(self, node: ast.AST, default: bool = False, generation: bool = False) -> None:
        note = (
            "use_cache defaults to True (M3); check the effective training flag and preserve generation behavior."
            if default
            else "Literal use_cache=True (M3); confirm the caller, actual cache allocation, and training vs. generation."
        )
        severity = "info" if generation else _kv_severity(self.func_stack)
        self._add(node, "kv_cache_enabled", severity, note)

    def _resolve_dotted_with_origin(self, node: Optional[ast.AST]) -> Tuple[str, Optional[bool]]:
        if node is None:
            return "", None
        dotted = _dotted_name(node)
        if not dotted:
            return "", None
        first, separator, rest = dotted.partition(".")
        current = len(self.bindings) - 1
        for index in range(current, -1, -1):
            # Methods and nested classes do not close over a class namespace.
            if index != current and index in self.class_scope_indices:
                continue
            scope = self.bindings[index]
            if first in scope:
                name = scope[first]
                return (name + separator + rest, True) if name is not None else ("", False)
        return dotted, None

    def _resolve_dotted(self, node: Optional[ast.AST]) -> str:
        return self._resolve_dotted_with_origin(node)[0]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.bindings[-1][alias.asname] = alias.name
            else:
                root = alias.name.split(".", 1)[0]
                self.bindings[-1][root] = root

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None and node.level == 0:
            for alias in node.names:
                if alias.name != "*":
                    self.bindings[-1][alias.asname or alias.name] = node.module + "." + alias.name

    def _record_kv_defaults(self, args: ast.arguments) -> None:
        positional = getattr(args, "posonlyargs", []) + args.args
        defaulted_args = positional[len(positional) - len(args.defaults):] + args.kwonlyargs
        for arg, default in zip(defaulted_args, args.defaults + args.kw_defaults):
            if arg.arg == "use_cache" and default is not None and _const_bool(default) is True:
                self._add_kv_candidate(default, default=True)

    def _visit_function(self, node) -> None:
        self.func_stack.append(node.name)
        self._record_kv_defaults(node.args)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        arguments = getattr(node.args, "posonlyargs", []) + node.args.args + node.args.kwonlyargs
        arguments += [arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None]
        self.bindings.append({arg.arg: None for arg in arguments})
        enclosing_loop = self.loop_depth
        self.loop_depth = 0
        for statement in node.body:
            self.visit(statement)
        self.loop_depth = enclosing_loop
        self.bindings.pop()
        self.func_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)
        index = len(self.bindings)
        self.bindings.append({})
        self.class_scope_indices.add(index)
        for statement in node.body:
            self.visit(statement)
        self.class_scope_indices.remove(index)
        self.bindings.pop()

    def visit_For(self, node: ast.For) -> None:
        # Evaluate the loop iterable once outside the loop body.
        self.visit(node.iter)
        self.visit(node.target)
        self.loop_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.loop_depth -= 1
        for stmt in node.orelse:
            self.visit(stmt)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        self.loop_depth -= 1
        for stmt in node.orelse:
            self.visit(stmt)

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
                elif name in _AC_BOOL_KEYS:
                    self._add(
                        node,
                        "ac_checkpoint_call",
                        "medium",
                        "{}=True enables activation checkpointing (M1/M2); confirm peak memory headroom or evaluate selective AC.".format(
                            name
                        ),
                    )
                elif name == "use_cache":
                    self._add_kv_candidate(node)
        self.generic_visit(node)
        # Retain simple local aliases such as checkpoint = torch.utils.checkpoint.checkpoint.
        value_name = self._resolve_dotted(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.bindings[-1][target.id] = value_name or None

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
            elif key in _AC_BOOL_KEYS and val is True:
                self._add(
                    v,
                    "ac_checkpoint_call",
                    "medium",
                    "Dict config sets '{}': True (M1/M2); confirm peak memory headroom or evaluate selective AC.".format(
                        key
                    ),
                )
            elif key == "use_cache" and val is True:
                self._add_kv_candidate(v)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted, is_imported = self._resolve_dotted_with_origin(node.func)
        short = dotted.rsplit(".", 1)[-1] or _call_short_name(node)
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
            self._add_kv_candidate(node, generation=short == "generate")

        for ac_key in _AC_BOOL_KEYS:
            if ac_key in kwargs and _const_bool(kwargs[ac_key]) is True:
                self._add(
                    node,
                    "ac_checkpoint_call",
                    "medium",
                    "{}=True enables activation checkpointing (M1/M2); check peak memory headroom or selective AC.".format(
                        ac_key
                    ),
                )

        # 2. Activation checkpointing API calls
        is_foreign_wrapper = (
            is_imported is True
            and short in ("checkpoint_wrapper", "apply_activation_checkpointing")
            and not dotted.startswith("torch.")
        )
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
        elif short in _CHECKPOINT_WRAPPER_NAMES and not is_foreign_wrapper:
            implementation = kwargs.get("checkpoint_impl")
            if (
                implementation is None
                and dotted == _CHECKPOINT_WRAPPER_MODULE + ".checkpoint_wrapper"
                and len(node.args) >= 2
            ):
                implementation = node.args[1]
            if (
                _const_bool(kwargs.get("use_reentrant")) is True
                or self._resolve_dotted(implementation) == _CHECKPOINT_REENTRANT_ENUM
            ):
                self._add(
                    node,
                    "ac_reentrant_true",
                    "high",
                    "{}() selects reentrant checkpointing (M1/M2); check target API compatibility and AC/SAC scope.".format(
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
                dotted in ("checkpoint", "checkpoint_sequential")
                and any(name in kwargs for name in ("use_reentrant", "preserve_rng_state", "context_fn"))
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
                "{}() called {} (M6); check actual frequency and allocator impact.".format(
                    dotted or short, ctx
                ),
            )

        if dotted == "gc.collect":
            sev = "medium" if self.loop_depth > 0 else "info"
            ctx = "inside a loop" if self.loop_depth > 0 else "outside a loop"
            finding = self._add(
                node,
                "hotpath_gc_collect",
                sev,
                "gc.collect() called {} (M6); corroborate with GC timings and rank step times.".format(
                    ctx
                ),
            )
            if self.loop_depth == 0 and (
                not self.func_stack or _is_gc_setup_func(self.func_stack[-1])
            ):
                self.warmup_collects[node] = finding
        elif dotted == "gc.freeze":
            self.gc_freezes.add(node)

        self.generic_visit(node)

    def results(self, tree: ast.AST) -> List[Finding]:
        if not self.gc_freezes or not self.warmup_collects:
            return self.findings
        paired = set()
        for node in ast.walk(tree):
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if not isinstance(block, list):
                    continue
                for collect, freeze in zip(block, block[1:]):
                    if isinstance(collect, ast.Expr) and isinstance(freeze, ast.Expr):
                        finding = self.warmup_collects.get(collect.value)
                        if finding is not None and freeze.value in self.gc_freezes:
                            paired.add(finding)
        return [finding for finding in self.findings if finding not in paired]


_RE_WITH_CP_TRUE = re.compile(r"""(?:\bwith_cp\s*=|['"]with_cp['"]\s*:)\s*True\b""")
_RE_AC_KEY_TRUE = re.compile(
    r"""(?:\b(?:gradient_checkpointing|use_checkpoint|use_checkpointing)\s*=|['"](?:gradient_checkpointing|use_checkpoint|use_checkpointing)['"]\s*:)\s*True\b"""
)
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
_RE_GC_COLLECT_STMT = re.compile(r"""gc\.collect\s*\([^()]*\)""")
_RE_GC_FREEZE_STMT = re.compile(r"""gc\.freeze\s*\(\s*\)""")
_RE_DEF_LINE = re.compile(r"""^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(""")
_RE_LOOP_LINE = re.compile(r"""^\s*(?:async\s+)?(?:for|while)\b""")


def _mask_strings_and_comments(text: str) -> List[Tuple[int, str, str]]:
    """Replace string and comment tokens with spaces while keeping original lines."""
    original = text.splitlines()
    masked = [list(line) for line in original]
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type not in (tokenize.STRING, tokenize.COMMENT):
                continue
            for lineno in range(token.start[0], token.end[0] + 1):
                if not (1 <= lineno <= len(masked)):
                    continue
                chars = masked[lineno - 1]
                start = token.start[1] if lineno == token.start[0] else 0
                stop = token.end[1] if lineno == token.end[0] else len(chars)
                chars[start:stop] = " " * (stop - start)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [
            (lineno, raw.split("#", 1)[0].rstrip(), raw.strip())
            for lineno, raw in enumerate(original, start=1)
            if raw.split("#", 1)[0].strip()
        ]
    result = []
    for lineno, chars in enumerate(masked, start=1):
        code = "".join(chars).rstrip()
        if code.strip():
            result.append((lineno, code, original[lineno - 1].strip()))
    return result


def _scan_python_regex_fallback(path: str, text: str) -> List[Finding]:
    """Fallback line scanner when host Python cannot ast.parse newer Python syntax."""
    findings: List[Finding] = []
    lines = _mask_strings_and_comments(text)
    current_func = ""
    current_func_indent = -1
    loop_indents: List[int] = []

    for index, (lineno, code, snippet) in enumerate(lines):
        stripped = code.strip()
        indent = len(code) - len(stripped)
        while loop_indents and indent <= loop_indents[-1]:
            loop_indents.pop()
        if current_func_indent >= 0 and indent <= current_func_indent:
            current_func = ""
            current_func_indent = -1

        m_def = _RE_DEF_LINE.match(code)
        if m_def:
            current_func = m_def.group(1)
            current_func_indent = indent
        elif _RE_LOOP_LINE.match(code):
            loop_indents.append(indent)

        in_loop = bool(loop_indents)

        if _RE_USE_CACHE_TRUE.search(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="kv_cache_enabled",
                    severity=_kv_severity((current_func,)),
                    snippet=snippet,
                    note="Literal use_cache=True (regex fallback, M3); check the actual caller and cache allocation.",
                )
            )
        if m_def:
            continue

        if _RE_WITH_CP_TRUE.search(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="ac_with_cp_true",
                    severity="medium",
                    snippet=snippet,
                    note="with_cp=True (regex fallback, M1); check baseline peak memory headroom before disabling.",
                )
            )

        if _RE_AC_KEY_TRUE.search(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="ac_checkpoint_call",
                    severity="medium",
                    snippet=snippet,
                    note="Activation checkpointing flag (regex fallback, M1/M2); verify peak memory headroom.",
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
                        snippet=snippet,
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
                        snippet=snippet,
                        note="Activation checkpointing call (regex fallback, M1/M2); verify peak memory headroom.",
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
                    snippet=snippet,
                    note="empty_cache() call (regex fallback, M6); check call frequency for allocator stalls.",
                )
            )

        if _RE_GC_COLLECT.search(code):
            following = lines[index + 1][1] if index + 1 < len(lines) else ""
            if (
                not in_loop
                and (not current_func or _is_gc_setup_func(current_func))
                and _RE_GC_COLLECT_STMT.fullmatch(stripped)
                and following[:indent] == code[:indent]
                and _RE_GC_FREEZE_STMT.fullmatch(following[indent:])
            ):
                continue
            sev = "medium" if in_loop else "info"
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="hotpath_gc_collect",
                    severity=sev,
                    snippet=snippet,
                    note="gc.collect() call (regex fallback, M6); corroborate with GC timings and rank step times.",
                )
            )

    return findings


def scan_python_file(path: str) -> Tuple[Optional[List[Finding]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError as exc:
        reason = str(exc)
        sys.stderr.write("[warn] cannot read {}: {}\n".format(path, reason))
        return None, reason

    tree = None
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        cleaned = re.sub(
            r"^[ \t]*from[ \t]+__future__[ \t]+import[ \t]+annotations[ \t]*$",
            "",
            text,
            flags=re.MULTILINE,
        )
        try:
            tree = ast.parse(cleaned, filename=path)
        except SyntaxError:
            sys.stderr.write(
                "[warn] AST parse unavailable for {} under Python {}.{}; using regex fallback\n".format(
                    path, sys.version_info[0], sys.version_info[1]
                )
            )
            return _scan_python_regex_fallback(path, text), None

    visitor = MemoryCandidateVisitor(path, text.splitlines())
    visitor.visit(tree)
    return visitor.results(tree), None


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


def scan_json_config(path: str) -> Tuple[Optional[List[Finding]], Optional[str]]:
    try:
        if os.path.getsize(path) > _MAX_CONFIG_BYTES:
            return [], None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        reason = str(exc)
        sys.stderr.write("[warn] skipped JSON {}: {}\n".format(path, reason))
        return None, reason

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
        elif leaf in _AC_BOOL_KEYS and val is True:
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
                    severity="info",
                    snippet="{}: true".format(key_path),
                    note="JSON use_cache=true (M3); check the effective training configuration and caller.",
                )
            )
    return findings, None


_RE_YAML_TRUE = re.compile(
    r"""^\s*(?P<quote>['"]?)(?P<key>with_cp|gradient_checkpointing|use_checkpoint|use_checkpointing|use_cache)"""
    r"""(?P=quote)\s*:\s*(?:true|True|TRUE)(?=\s*$|[ \t]+#)"""
)
_RE_YAML_TEXT_BLOCK = re.compile(r"^ *(?!-\s)[^ #][^:]*:\s*[|>](?:[1-9][+-]?|[+-][1-9]?)?\s*(?:#.*)?$")


def scan_yaml_config(path: str) -> Tuple[Optional[List[Finding]], Optional[str]]:
    try:
        if os.path.getsize(path) > _MAX_CONFIG_BYTES:
            return [], None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except OSError as exc:
        reason = str(exc)
        sys.stderr.write("[warn] cannot read YAML {}: {}\n".format(path, reason))
        return None, reason

    findings: List[Finding] = []
    text_block_indent = None
    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(stripped)
        if text_block_indent is not None:
            if indent > text_block_indent:
                continue
            text_block_indent = None
        if _RE_YAML_TEXT_BLOCK.match(raw):
            text_block_indent = indent
            continue
        match = _RE_YAML_TRUE.match(raw)
        if match is None:
            continue
        key = match.group("key")
        snippet = match.group(0).strip()
        if key == "with_cp":
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="ac_with_cp_true",
                    severity="medium",
                    snippet=snippet,
                    note="YAML config sets with_cp: true (M1); check baseline peak memory headroom before disabling.",
                )
            )
        elif key in _AC_BOOL_KEYS:
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="ac_checkpoint_call",
                    severity="medium",
                    snippet=snippet,
                    note="YAML config enables {} (M1/M2); check peak memory headroom or SAC.".format(key),
                )
            )
        else:
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="kv_cache_enabled",
                    severity="info",
                    snippet=snippet,
                    note="YAML use_cache: true (M3); check the effective training configuration and caller.",
                )
            )
    return findings, None


def _iter_targets(paths: List[str], include_configs: bool, on_error) -> Iterable[Tuple[str, str]]:
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
        for root, dirs, files in os.walk(target, onerror=on_error):
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


def scan_paths_with_status(
    paths: List[str], include_configs: bool = True
) -> Tuple[List[Finding], int, List[Dict[str, str]]]:
    all_findings: List[Finding] = []
    scanned = 0
    skipped_files: List[Dict[str, str]] = []

    def walk_error(error):
        path_name = getattr(error, "filename", None) or "<walk>"
        reason = str(error)
        skipped_files.append({"file": str(path_name), "reason": reason})
        sys.stderr.write("[warn] skipped {}\n".format(reason))

    for kind, path in _iter_targets(paths, include_configs=include_configs, on_error=walk_error):
        if kind == "py":
            result, reason = scan_python_file(path)
        elif kind == "json":
            result, reason = scan_json_config(path)
        else:
            result, reason = scan_yaml_config(path)
        if result is None:
            skipped_files.append({"file": path, "reason": reason or "unknown error"})
        else:
            scanned += 1
            all_findings.extend(result)
    return all_findings, scanned, skipped_files


def scan_paths(paths: List[str], include_configs: bool = True) -> List[Finding]:
    findings, _, _ = scan_paths_with_status(paths, include_configs=include_configs)
    return findings


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

    findings, scanned, skipped_files = scan_paths_with_status(
        args.paths, include_configs=not args.no_configs
    )
    min_rank = _SEV_ORDER[args.min_severity]
    filtered = [f for f in findings if _SEV_ORDER[f.severity] >= min_rank]
    skipped = len(skipped_files)

    if skipped or not scanned:
        sys.stderr.write("Files: {} scanned, {} skipped.\n".format(scanned, skipped))
    sys.stderr.write(_DISCLAIMER + "\n")

    if args.as_json:
        payload = {
            "disclaimer": _DISCLAIMER,
            "files_scanned": scanned,
            "files_skipped": skipped,
            "skipped_files": skipped_files,
            "findings": [f.to_dict() for f in filtered],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1 if skipped or not scanned else 0

    if not scanned:
        print("No supported files scanned.")
        return 1

    if not filtered:
        print("No static memory/recompute candidates found.")
        return 1 if skipped else 0

    for item in filtered:
        print(
            "{}:{}: [{}] ({}) {}".format(
                item.file, item.line, item.severity.upper(), item.pattern, item.snippet
            )
        )
        print("    -> {}".format(item.note))
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
