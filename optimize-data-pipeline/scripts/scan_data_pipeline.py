#!/usr/bin/env python3
"""Static scanner for PyTorch DataLoader, blocking H2D, and multiprocessing candidates.

This is a STATIC scanner. It scans Python source files (AST, with regex fallback
for newer Python syntax on older runtimes) and YAML/JSON config files for:
  - Conservative DataLoader settings: pin_memory=False (or omitted), num_workers=0,
    persistent_workers=False (or omitted when num_workers > 0).
  - Blocking H2D transfers: .to(device) / .cuda() missing non_blocking=True, or
    explicit non_blocking=False (skipping dtype-only casts, CPU transfers, and
    one-time model placement outside loops).
  - Multiprocessing context risks: DataLoader with num_workers > 0 in a file
    importing TensorFlow/TFDS without spawn context, or explicit fork context.

It does NOT prove that data loading or H2D is a real bottleneck, check host RAM
or /dev/shm headroom, verify whether tensors are actually pinned inside custom
containers, or guarantee sample order / resume equivalence. Always confirm with
trace-bottleneck-report (§3/§5) or data-iteration timing before optimizing.

Usage:
    python3 scan_data_pipeline.py <file-or-dir> [<file-or-dir> ...]
    python3 scan_data_pipeline.py --json <file-or-dir> ...
    python3 scan_data_pipeline.py --min-severity medium <file-or-dir> ...
    python3 scan_data_pipeline.py --no-configs <file-or-dir> ...

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

_DATALOADER_NAMES = {"DataLoader", "StatefulDataLoader"}
_MODEL_RECEIVER_TOKENS = (
    "model",
    "net",
    "module",
    "encoder",
    "decoder",
    "backbone",
    "ema",
    "student",
    "teacher",
    "vae",
    "policy",
)
_HOT_H2D_FUNC_TOKENS = (
    "prepare_input",
    "prepare_batch",
    "move_to_device",
    "send_to_device",
    "to_device",
    "train_step",
    "training_step",
    "train_epoch",
    "train_loop",
    "train_one_epoch",
    "forward_train",
)
_INIT_EVAL_FUNC_TOKENS = (
    "__init__",
    "setup",
    "build_model",
    "load_checkpoint",
    "eval",
    "evaluate",
    "test",
    "infer",
    "inference",
    "generate",
)
_DTYPE_TOKENS = (
    "float",
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "bf16",
    "fp16",
    "fp32",
    "half",
    "double",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "long",
    "short",
    "bool",
    "dtype",
)
_DEVICE_TOKENS = (
    "device",
    "cuda",
    "gpu",
    "npu",
    "xpu",
    "rank",
    "local_rank",
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


def _has_star_kwargs(node: ast.Call) -> bool:
    return any(kw.arg is None for kw in node.keywords)


def _const_bool(node: Optional[ast.AST]) -> Optional[bool]:
    if node is None:
        return None
    if isinstance(node, ast.NameConstant) and isinstance(node.value, bool):
        return node.value
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _const_int(node: Optional[ast.AST]) -> Optional[int]:
    if node is None:
        return None
    if isinstance(node, ast.Num) and isinstance(node.n, int) and not isinstance(node.n, bool):
        return node.n
    if (
        hasattr(ast, "Constant")
        and isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
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


def _has_token(name: str, tokens: Tuple[str, ...]) -> bool:
    low = name.lower()
    for tok in tokens:
        if tok in low:
            return True
    return False


def _is_cpu_target(node: Optional[ast.AST]) -> bool:
    if node is None:
        return False
    s = _const_str(node)
    if s is not None and s.lower().startswith("cpu"):
        return True
    if isinstance(node, ast.Call) and _dotted_name(node.func).endswith("device"):
        if node.args and _const_str(node.args[0]) == "cpu":
            return True
    dotted = _dotted_name(node).lower()
    if dotted == "cpu" or dotted.endswith(".cpu_device"):
        return True
    return False


def _is_dtype_only_arg(node: ast.AST) -> bool:
    dotted = _dotted_name(node).lower()
    if not dotted:
        return False
    leaf = dotted.rsplit(".", 1)[-1]
    return leaf in _DTYPE_TOKENS or dotted.startswith("torch.float") or dotted.startswith("torch.bfloat") or dotted.startswith("torch.int")


def _looks_like_device_arg(node: Optional[ast.AST]) -> bool:
    if node is None:
        return False
    if _is_cpu_target(node):
        return False
    s = _const_str(node)
    if s is not None:
        low = s.lower()
        return low.startswith("cuda") or low.startswith("npu") or low.startswith("xpu") or low.startswith("mps")
    if isinstance(node, ast.Call):
        d = _dotted_name(node.func).lower()
        if d.endswith("torch.device") or d == "device" or d.endswith("current_device"):
            return True
    if _is_dtype_only_arg(node):
        return False
    dotted = _dotted_name(node).lower()
    if not dotted:
        return False
    leaf = dotted.rsplit(".", 1)[-1]
    return leaf in _DEVICE_TOKENS or any(tok in leaf for tok in ("device", "cuda", "rank"))


def _mp_context_kind(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    s = _const_str(node)
    if s is not None:
        return s.lower()
    if isinstance(node, ast.Call) and _call_short_name(node) == "get_context":
        if node.args:
            arg_s = _const_str(node.args[0])
            if arg_s is not None:
                return arg_s.lower()
    dotted = _dotted_name(node).lower()
    if "spawn" in dotted:
        return "spawn"
    if "fork" in dotted:
        return "fork"
    return ""


class DataPipelineVisitor(ast.NodeVisitor):
    def __init__(self, path: str, lines: List[str]) -> None:
        self.path = path
        self.lines = lines
        self.findings: List[Finding] = []
        self.loop_depth = 0
        self.func_stack: List[str] = []
        self.imports_tf = False

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

    def _in_hot_h2d_func(self) -> bool:
        for fn in self.func_stack:
            if _has_token(fn, _HOT_H2D_FUNC_TOKENS):
                return True
        return False

    def _in_init_or_eval_func(self) -> bool:
        for fn in self.func_stack:
            if _has_token(fn, _INIT_EVAL_FUNC_TOKENS):
                return True
        return False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name.lower()
            if name == "tensorflow" or name.startswith("tensorflow.") or name.startswith("tensorflow_datasets"):
                self.imports_tf = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = (node.module or "").lower()
        if mod == "tensorflow" or mod.startswith("tensorflow.") or mod.startswith("tensorflow_datasets"):
            self.imports_tf = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.func_stack.append(node.name)
        self.generic_visit(node)
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

    def visit_Dict(self, node: ast.Dict) -> None:
        num_workers_val = None
        for k, v in zip(node.keys, node.values):
            if _const_str(k) in ("num_workers", "workers_per_gpu"):
                num_workers_val = _const_int(v)
        for k, v in zip(node.keys, node.values):
            key = _const_str(k)
            if key == "pin_memory" and _const_bool(v) is False:
                self._add(
                    v,
                    "dataloader_pin_memory_disabled",
                    "high",
                    "Dict config sets 'pin_memory': False (D1); pinned host memory is required for async H2D.",
                )
            elif key in ("num_workers", "workers_per_gpu") and _const_int(v) == 0:
                self._add(
                    v,
                    "dataloader_num_workers_zero",
                    "high",
                    "Dict config sets '{}': 0 (D1/D2); verify if single-process loading is starving GPU or guarding fork deadlock.".format(
                        key
                    ),
                )
            elif key == "persistent_workers" and _const_bool(v) is False and num_workers_val != 0:
                self._add(
                    v,
                    "dataloader_persistent_workers_off",
                    "medium",
                    "Dict config sets 'persistent_workers': False (D1); enabling avoids per-epoch worker respawn stalls when num_workers > 0.",
                )
            elif key == "non_blocking" and _const_bool(v) is False:
                self._add(
                    v,
                    "blocking_h2d_transfer",
                    "medium",
                    "Dict config sets 'non_blocking': False (D3); enable with pinned memory to overlap H2D.",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        short = _call_short_name(node)
        dotted = _dotted_name(node.func)
        kwargs = _kwarg_map(node)
        has_star_kw = _has_star_kwargs(node)

        # 1. DataLoader(...) calls
        if short in _DATALOADER_NAMES or dotted.endswith(".DataLoader"):
            pin_val = _const_bool(kwargs.get("pin_memory"))
            nw_val = _const_int(kwargs.get("num_workers"))
            pw_val = _const_bool(kwargs.get("persistent_workers"))
            mp_kind = _mp_context_kind(kwargs.get("multiprocessing_context"))

            if pin_val is False:
                self._add(
                    node,
                    "dataloader_pin_memory_disabled",
                    "high",
                    "DataLoader explicitly sets pin_memory=False (D1); enable pin_memory=True (and custom container pinning if needed) for async H2D.",
                )
            elif "pin_memory" not in kwargs and not has_star_kw:
                self._add(
                    node,
                    "dataloader_pin_memory_disabled",
                    "medium",
                    "DataLoader omits pin_memory (defaults to False) (D1); verify host RAM headroom and enable pin_memory=True for GPU training.",
                )

            if nw_val == 0:
                self._add(
                    node,
                    "dataloader_num_workers_zero",
                    "high",
                    "DataLoader sets num_workers=0 (D1/D2); check if main-process loading causes step-head wait (use spawn + __reduce__ if guarding TF/CUDA fork deadlock).",
                )

            if pw_val is False and nw_val != 0:
                self._add(
                    node,
                    "dataloader_persistent_workers_off",
                    "medium",
                    "DataLoader sets persistent_workers=False (D1); enabling persistent_workers=True when num_workers > 0 avoids epoch-boundary stalls.",
                )
            elif (
                "persistent_workers" not in kwargs
                and not has_star_kw
                and nw_val is not None
                and nw_val > 0
            ):
                self._add(
                    node,
                    "dataloader_persistent_workers_off",
                    "medium",
                    "DataLoader sets num_workers={} without persistent_workers=True (D1); consider persistent_workers=True for multi-epoch training.".format(
                        nw_val
                    ),
                )

            if self.imports_tf and nw_val is not None and nw_val > 0 and mp_kind != "spawn":
                self._add(
                    node,
                    "mp_context_fork_with_tf",
                    "high",
                    "DataLoader with num_workers={} in a file importing TensorFlow without multiprocessing_context='spawn' (D2); fork can deadlock active TF thread pools.".format(
                        nw_val
                    ),
                )
            elif mp_kind == "fork":
                self._add(
                    node,
                    "mp_context_fork_with_tf",
                    "info",
                    "DataLoader explicitly uses fork multiprocessing_context (D2); verify no active TF/CUDA/OpenMP threads before fork.",
                )

        # 2. DataLoaderConfiguration(non_blocking=False) or generic dict(pin_memory=False)
        if short == "DataLoaderConfiguration":
            nb = _const_bool(kwargs.get("non_blocking"))
            if nb is False or ("non_blocking" not in kwargs and not has_star_kw):
                self._add(
                    node,
                    "blocking_h2d_transfer",
                    "medium",
                    "DataLoaderConfiguration leaves non_blocking disabled (D3); enable non_blocking=True alongside pin_memory=True.",
                )
        elif short == "dict":
            nw_val = _const_int(kwargs.get("num_workers") or kwargs.get("workers_per_gpu"))
            if _const_bool(kwargs.get("pin_memory")) is False:
                self._add(
                    node,
                    "dataloader_pin_memory_disabled",
                    "high",
                    "dict(...) sets pin_memory=False (D1); enable pin_memory=True for async H2D.",
                )
            if nw_val == 0:
                self._add(
                    node,
                    "dataloader_num_workers_zero",
                    "high",
                    "dict(...) sets num_workers/workers_per_gpu=0 (D1/D2).",
                )
            if _const_bool(kwargs.get("persistent_workers")) is False and nw_val != 0:
                self._add(
                    node,
                    "dataloader_persistent_workers_off",
                    "medium",
                    "dict(...) sets persistent_workers=False (D1).",
                )

        # 3. Blocking .cuda() / .to(device) transfers
        if isinstance(node.func, ast.Attribute) and short in ("cuda", "to"):
            recv_name = _dotted_name(node.func.value).lower()
            is_model_recv = _has_token(recv_name, _MODEL_RECEIVER_TOKENS)
            nb_val = _const_bool(kwargs.get("non_blocking"))

            is_h2d_call = False
            if short == "cuda":
                is_h2d_call = True
            elif short == "to":
                if "device" in kwargs and _looks_like_device_arg(kwargs["device"]):
                    is_h2d_call = True
                elif node.args and _looks_like_device_arg(node.args[0]):
                    is_h2d_call = True

            if is_h2d_call and nb_val is not True:
                # Skip one-time model.to(device) outside loops
                if is_model_recv and self.loop_depth == 0:
                    pass
                else:
                    if self.loop_depth > 0 or self._in_hot_h2d_func():
                        sev = "high"
                        ctx = "in hot path / loop"
                    elif self._in_init_or_eval_func():
                        sev = "info"
                        ctx = "in init/eval helper"
                    else:
                        sev = "medium"
                        ctx = "outside explicit loop"
                    self._add(
                        node,
                        "blocking_h2d_transfer",
                        sev,
                        ".{}() missing non_blocking=True {} (D3); confirm host tensor is pinned before enabling non_blocking=True.".format(
                            short, ctx
                        ),
                    )

        self.generic_visit(node)


_RE_PIN_FALSE = re.compile(r"""(?:\bpin_memory\s*=|['"]pin_memory['"]\s*:)\s*False\b""")
_RE_NW_ZERO = re.compile(r"""(?:\b(?:num_workers|workers_per_gpu)\s*=|['"](?:num_workers|workers_per_gpu)['"]\s*:)\s*0\b""")
_RE_PW_FALSE = re.compile(r"""(?:\bpersistent_workers\s*=|['"]persistent_workers['"]\s*:)\s*False\b""")
_RE_NB_FALSE = re.compile(r"""(?:\bnon_blocking\s*=|['"]non_blocking['"]\s*:)\s*False\b""")
_RE_TO_DEVICE = re.compile(
    r"""\.(?:cuda\s*\((?!.*non_blocking\s*=\s*True)|to\s*\([^)]*(?:\bdevice\b|['"]cuda|local_rank)[^)]*\))"""
)
_RE_DEF_LINE = re.compile(r"""^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(""")
_RE_LOOP_LINE = re.compile(r"""^\s*(?:async\s+)?(?:for|while)\b""")


def _scan_python_regex_fallback(path: str, text: str) -> List[Finding]:
    """Fallback line scanner when host Python cannot ast.parse newer Python syntax."""
    findings: List[Finding] = []
    lines = text.splitlines()
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
        in_hot = in_loop or (bool(current_func) and _has_token(current_func, _HOT_H2D_FUNC_TOKENS))

        if _RE_PIN_FALSE.search(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="dataloader_pin_memory_disabled",
                    severity="high",
                    snippet=stripped,
                    note="pin_memory=False (regex fallback, D1); enable pinned memory for async H2D.",
                )
            )
        if _RE_NW_ZERO.search(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="dataloader_num_workers_zero",
                    severity="high",
                    snippet=stripped,
                    note="num_workers=0 (regex fallback, D1/D2); check for step-head CPU starvation or fork-deadlock workaround.",
                )
            )
        if _RE_PW_FALSE.search(code) and not _RE_NW_ZERO.search(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="dataloader_persistent_workers_off",
                    severity="medium",
                    snippet=stripped,
                    note="persistent_workers=False (regex fallback, D1).",
                )
            )
        if _RE_NB_FALSE.search(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="blocking_h2d_transfer",
                    severity="medium",
                    snippet=stripped,
                    note="non_blocking=False (regex fallback, D3).",
                )
            )
        elif _RE_TO_DEVICE.search(code) and "non_blocking=True" not in code and "non_blocking = True" not in code:
            if "cpu" not in code.lower():
                sev = "high" if in_hot else "medium"
                findings.append(
                    Finding(
                        file=path,
                        line=lineno,
                        col=0,
                        pattern="blocking_h2d_transfer",
                        severity=sev,
                        snippet=stripped,
                        note="Blocking H2D transfer without non_blocking=True (regex fallback, D3).",
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

    visitor = DataPipelineVisitor(path, text.splitlines())
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
        if leaf == "pin_memory" and val is False:
            findings.append(
                Finding(
                    file=path,
                    line=1,
                    col=0,
                    pattern="dataloader_pin_memory_disabled",
                    severity="high",
                    snippet="{}: false".format(key_path),
                    note="JSON config sets pin_memory=false (D1).",
                )
            )
        elif leaf in ("num_workers", "workers_per_gpu") and val == 0 and not isinstance(val, bool):
            findings.append(
                Finding(
                    file=path,
                    line=1,
                    col=0,
                    pattern="dataloader_num_workers_zero",
                    severity="high",
                    snippet="{}: 0".format(key_path),
                    note="JSON config sets {}=0 (D1/D2).".format(leaf),
                )
            )
        elif leaf == "persistent_workers" and val is False:
            findings.append(
                Finding(
                    file=path,
                    line=1,
                    col=0,
                    pattern="dataloader_persistent_workers_off",
                    severity="medium",
                    snippet="{}: false".format(key_path),
                    note="JSON config sets persistent_workers=false (D1).",
                )
            )
        elif leaf == "non_blocking" and val is False:
            findings.append(
                Finding(
                    file=path,
                    line=1,
                    col=0,
                    pattern="blocking_h2d_transfer",
                    severity="medium",
                    snippet="{}: false".format(key_path),
                    note="JSON config sets non_blocking=false (D3).",
                )
            )
    return findings


_RE_YAML_PIN_FALSE = re.compile(r"^\s*pin_memory\s*:\s*(?:false|False|FALSE)\b")
_RE_YAML_NW_ZERO = re.compile(r"^\s*(?:num_workers|workers_per_gpu)\s*:\s*0\s*$")
_RE_YAML_PW_FALSE = re.compile(r"^\s*persistent_workers\s*:\s*(?:false|False|FALSE)\b")
_RE_YAML_NB_FALSE = re.compile(r"^\s*non_blocking\s*:\s*(?:false|False|FALSE)\b")


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
        if _RE_YAML_PIN_FALSE.match(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="dataloader_pin_memory_disabled",
                    severity="high",
                    snippet=code.strip(),
                    note="YAML config sets pin_memory: false (D1).",
                )
            )
        elif _RE_YAML_NW_ZERO.match(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="dataloader_num_workers_zero",
                    severity="high",
                    snippet=code.strip(),
                    note="YAML config sets num_workers/workers_per_gpu: 0 (D1/D2).",
                )
            )
        elif _RE_YAML_PW_FALSE.match(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="dataloader_persistent_workers_off",
                    severity="medium",
                    snippet=code.strip(),
                    note="YAML config sets persistent_workers: false (D1).",
                )
            )
        elif _RE_YAML_NB_FALSE.match(code):
            findings.append(
                Finding(
                    file=path,
                    line=lineno,
                    col=0,
                    pattern="blocking_h2d_transfer",
                    severity="medium",
                    snippet=code.strip(),
                    note="YAML config sets non_blocking: false (D3).",
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
        description="Static scanner for PyTorch DataLoader configuration, blocking H2D transfers, and multiprocessing context risks."
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
        "[static-scan] Results are STATIC candidates only; they do NOT prove data loading or H2D is "
        "a bottleneck, verify /dev/shm headroom, or guarantee sample order / resume equivalence.\n"
    )

    if args.as_json:
        print(json.dumps([f.to_dict() for f in filtered], indent=2, ensure_ascii=False))
        return 0

    if not filtered:
        print("No static data-pipeline candidates found.")
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
