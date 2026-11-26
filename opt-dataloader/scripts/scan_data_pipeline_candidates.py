#!/usr/bin/env python3
"""Static scanner for PyTorch training data-pipeline and H2D feeding candidates.

This is a STATIC scanner. It scans Python source files (AST, with partial
recovery at info severity for unparseable files) and YAML/JSON config files for:
  - DataLoader construction with num_workers=0, pin_memory=False/omitted, or
    persistent_workers=False/omitted when num_workers > 0 (downgraded to info
    inside eval/test/val helpers).
  - Accelerate Accelerator() without dataloader_config= or
    DataLoaderConfiguration with non_blocking=False/omitted.
  - Blocking H2D transfers (.cuda() / .to(device) without non_blocking=True)
    in training/loader paths (skipping pure dtype casts and CPU transfers).
  - MMCV DataContainer usage where standard pin_memory threads may skip pinning.
  - Hugging Face dataset chained row-then-column indexing (hf_dataset[idx][col])
    without select_columns(), which can trigger unnecessary image decoding.
  - Loop-based Shapely/geometry .interpolate() calls on CPU dataset workers.

It does NOT prove the DataLoader or H2D copy is a real bottleneck, check host
RAM / /dev/shm headroom, or verify sampler/resume equivalence. Check the runtime
call path and trace before treating a finding as a measured bottleneck.
JSON config findings use a key path; line 0 means the source line is ambiguous.

Usage:
    python3 scan_data_pipeline_candidates.py <file-or-dir> [<file-or-dir> ...]
    python3 scan_data_pipeline_candidates.py --json <file-or-dir> ...
    python3 scan_data_pipeline_candidates.py --min-severity medium <file-or-dir> ...
    python3 scan_data_pipeline_candidates.py --no-configs <file-or-dir> ...

Only the Python standard library is used (compatible with Python 3.6+).
"""

import argparse
import ast
import io
import json
import os
import re
import sys
import tokenize
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

_EVAL_TOKENS = (
    "eval",
    "evaluate",
    "val",
    "valid",
    "validation",
    "test",
    "infer",
    "inference",
    "predict",
)

_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3}
_CONTAINER_TRANSFER_HELPERS = {"pin_memory", "_pin_memory", "_to_cuda"}
_MODEL_RECEIVERS = {"model", "module", "net", "network", "policy"}
_ACCELERATOR_DEVICES = {"cuda", "hip", "npu", "xpu", "mps"}

_NOTES = {
    "dataloader_num_workers_zero": (
        "DataLoader uses num_workers=0; evaluate multiprocessing workers (use spawn + picklable "
        "dataset for TF/RLDS thread-pool datasets, see D1/D3)."
    ),
    "dataloader_pin_memory_disabled": (
        "DataLoader disables or omits pin_memory; verify source pinning for H2D/compute "
        "overlap and check host RAM / /dev/shm budget (see D1/D4)."
    ),
    "dataloader_persistent_workers_off": (
        "Multi-worker DataLoader disables or omits persistent_workers; enabling it avoids "
        "per-epoch worker respawn stalls when num_workers > 0 (see D1)."
    ),
    "accelerator_blocking_h2d": (
        "Accelerate defaults to non_blocking=False; evaluate DataLoaderConfiguration(non_blocking=True) "
        "together with pin_memory=True (see D1/D4)."
    ),
    "blocking_h2d_transfer": (
        "Potential blocking H2D copy (.cuda() or .to(device) without non_blocking=True); confirm "
        "whether the source tensor is pinned and whether H2D is exposed in trace (see D1/D4)."
    ),
    "custom_datacontainer_pin_risk": (
        "Custom DataContainer detected without local pin_memory handling; standard DataLoader "
        "pin_memory threads skip unknown containers unless patched (see D2)."
    ),
    "hf_dataset_full_row_index": (
        "Chained hf_dataset[...][key] indexing may decode unused image/video columns; consider "
        "select_columns([key]) or a column-pruned view (see D5)."
    ),
    "geometry_interpolate_loop": (
        ".interpolate(...) called inside a Python loop; if on a CPU dataset hot path, consider "
        "NumPy vectorized resampling (see D5)."
    ),
    "config_data_pipeline_flag": (
        "Config sets conservative DataLoader/H2D flag; verify whether training loader benefits "
        "from enabling pin_memory/persistent_workers/non_blocking or workers > 0 (see D1)."
    ),
}


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


def _dotted(node: ast.AST) -> str:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_bool_const(node: Optional[ast.AST]) -> Optional[bool]:
    if node is None:
        return None
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        return None
    if isinstance(node, ast.NameConstant) and isinstance(node.value, bool):
        return node.value
    return None


def _is_int_const(node: Optional[ast.AST]) -> Optional[int]:
    if node is None:
        return None
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        return None
    if isinstance(node, ast.Num) and isinstance(node.n, int) and not isinstance(node.n, bool):
        return node.n
    return None


def _is_str_const(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    return None


def _looks_like_eval_name(name: str) -> bool:
    low = name.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
    return any(t in _EVAL_TOKENS for t in tokens)


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self, path: str, lines: List[str], has_pin_memory_def: bool) -> None:
        self.path = path
        self.lines = lines
        self.has_pin_memory_def = has_pin_memory_def
        self.findings: List[Finding] = []
        self._func_stack: List[str] = []
        self._loop_depth = 0
        self._emitted_dc = False
        self._functional_modules = {"F"}

    def _in_eval_context(self, extra_name: str = "") -> bool:
        if extra_name and _looks_like_eval_name(extra_name):
            return True
        return any(_looks_like_eval_name(fn) for fn in self._func_stack)

    def _emit(self, node: ast.AST, pattern: str, severity: str) -> None:
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        snippet = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        self.findings.append(
            Finding(self.path, line, col, pattern, severity, snippet, _NOTES[pattern])
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        inherited_modules = self._functional_modules
        argument_names = {arg.arg for arg in ast.iter_child_nodes(node.args) if isinstance(arg, ast.arg)}
        self._functional_modules = inherited_modules - argument_names
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()
        self._functional_modules = inherited_modules

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "torch.nn.functional":
                self._functional_modules.add(alias.asname or alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if node.module == "torch.nn" and alias.name == "functional":
                self._functional_modules.add(alias.asname or alias.name)

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_Assign(self, node: ast.Assign) -> None:
        target_name = ""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
        if target_name and _looks_like_eval_name(target_name):
            self._func_stack.append(target_name)
            self.generic_visit(node)
            self._func_stack.pop()
        else:
            self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Detect chained self.hf_dataset[indices][key] or hf_dataset[indices][key]
        if isinstance(node.value, ast.Subscript):
            inner_target = _dotted(node.value.value)
            if inner_target.endswith("hf_dataset"):
                self._emit(node, "hf_dataset_full_row_index", "medium")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._check_call(node)
        self.generic_visit(node)

    def _check_call(self, node: ast.Call) -> None:
        name = _dotted(node.func)
        tail = name.rsplit(".", 1)[-1] if name else ""
        kw_map = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        has_star_kwargs = any(kw.arg is None for kw in node.keywords)
        eval_ctx = self._in_eval_context()

        if tail == "DataLoader":
            nw_val = _is_int_const(kw_map.get("num_workers"))
            pm_val = _is_bool_const(kw_map.get("pin_memory"))
            pw_val = _is_bool_const(kw_map.get("persistent_workers"))

            if nw_val == 0:
                self._emit(
                    node,
                    "dataloader_num_workers_zero",
                    "info" if eval_ctx else "high",
                )
            if pm_val is False:
                self._emit(
                    node,
                    "dataloader_pin_memory_disabled",
                    "info" if eval_ctx else "high",
                )
            elif "pin_memory" not in kw_map and not has_star_kwargs and not eval_ctx:
                self._emit(node, "dataloader_pin_memory_disabled", "medium")

            if nw_val is not None and nw_val > 0:
                if pw_val is False or ("persistent_workers" not in kw_map and not has_star_kwargs):
                    self._emit(
                        node,
                        "dataloader_persistent_workers_off",
                        "info" if eval_ctx else "medium",
                    )

        elif tail == "Accelerator":
            if "dataloader_config" not in kw_map and not has_star_kwargs:
                self._emit(
                    node,
                    "accelerator_blocking_h2d",
                    "info" if eval_ctx else "high",
                )

        elif tail == "DataLoaderConfiguration":
            nb_val = _is_bool_const(kw_map.get("non_blocking"))
            if nb_val is False or ("non_blocking" not in kw_map and not has_star_kwargs):
                self._emit(
                    node,
                    "accelerator_blocking_h2d",
                    "info" if eval_ctx else "high",
                )

        elif tail == "cuda" and isinstance(node.func, ast.Attribute):
            if name != "torch.cuda":
                self._check_transfer(node, kw_map, has_star_kwargs, eval_ctx)

        elif tail == "to" and isinstance(node.func, ast.Attribute):
            if self._looks_like_device_to(node, kw_map):
                self._check_transfer(node, kw_map, has_star_kwargs, eval_ctx)

        elif tail == "DataContainer" and not self.has_pin_memory_def and not self._emitted_dc:
            if not any(scope in _CONTAINER_TRANSFER_HELPERS for scope in self._func_stack):
                self._emitted_dc = True
                self._emit(node, "custom_datacontainer_pin_risk", "medium")

        elif tail == "interpolate" and self._loop_depth > 0 and isinstance(node.func, ast.Attribute):
            receiver = _dotted(node.func.value)
            if receiver not in self._functional_modules and not receiver.endswith(".functional"):
                self._emit(node, "geometry_interpolate_loop", "medium")

    def _check_transfer(self, node: ast.Call, kw_map: Dict[str, ast.AST],
                        has_star_kwargs: bool, eval_ctx: bool) -> None:
        nb_val = _is_bool_const(kw_map.get("non_blocking"))
        if not self._looks_like_model(node.func.value) and (nb_val is False or (nb_val is None and not has_star_kwargs)):
            self._emit(node, "blocking_h2d_transfer", "info" if eval_ctx else "medium")

    @staticmethod
    def _looks_like_model(node: ast.AST) -> bool:
        return _dotted(node).lower().rsplit(".", 1)[-1] in _MODEL_RECEIVERS

    def _looks_like_device_to(
        self, node: ast.Call, kw_map: Dict[str, ast.AST]
    ) -> bool:
        target = kw_map.get("device", node.args[0] if node.args else None)
        if isinstance(target, ast.Call) and _dotted(target.func) == "torch.device":
            device_type = _is_str_const(target.args[0]) if target.args else None
            return device_type is None or device_type.lower().split(":", 1)[0] in _ACCELERATOR_DEVICES
        device_type = _is_str_const(target)
        if device_type is not None:
            return device_type.lower().split(":", 1)[0] in _ACCELERATOR_DEVICES
        if "device" in kw_map:
            return True
        arg_name = _dotted(target).lower()
        return ("dtype" not in arg_name and not arg_name.startswith("torch.")
                and any(tok in arg_name for tok in ("device", "cuda", "rank", "gpu")))


_FALLBACK_CALLS = {"DataLoader", "Accelerator", "DataLoaderConfiguration", "DataContainer"}
_BRACKETS = {"(": ")", "[": "]", "{": "}"}
# Older CPython can report a parser-stack overflow as MemoryError.
_AST_PARSE_ERRORS = (SyntaxError, MemoryError, RecursionError)


def _bracket_pairs(tokens):
    pairs = {}
    stack = []
    for index, token in enumerate(tokens):
        if token.type != tokenize.OP:
            continue
        if token.string in _BRACKETS:
            stack.append((index, _BRACKETS[token.string]))
        elif token.string in (")", "]", "}"):
            if stack and token.string == stack[-1][1]:
                opening, _ = stack.pop()
                pairs[opening] = index
            else:
                stack.clear()
    return pairs


def _recover_call(name_token, closing_token, source, offsets):
    """Parse one balanced call within a file that the full AST cannot read."""
    start_line, start_col = name_token.start
    end_line, end_col = closing_token.end
    call = source[offsets[start_line - 1] + start_col:offsets[end_line - 1] + end_col]
    try:
        tree = ast.parse(call, mode="eval")
    except _AST_PARSE_ERRORS:
        return None
    return ast.increment_lineno(tree, start_line - 1)


def _scan_python_fallback(path: str, source: str, lines: List[str]) -> List[Finding]:
    """Recover isolated calls and recognizable HF indexing at info severity."""
    tokens = []
    try:
        tokens.extend(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        pass  # Complete calls before the malformed region are still usable.

    offsets = []
    offset = 0
    for line in source.splitlines(keepends=True):
        offsets.append(offset)
        offset += len(line)

    has_pin_def = any(
        token.type == tokenize.NAME and token.string == "def"
        and tokens[index + 1].string == "pin_memory"
        for index, token in enumerate(tokens[:-1])
    )
    visitor = _PythonVisitor(path, lines, has_pin_def)
    bracket_pairs = _bracket_pairs(tokens)
    for index, token in enumerate(tokens[1:], 1):
        previous = tokens[index - 1]
        if token.type != tokenize.OP or token.string != "(" or previous.type != tokenize.NAME:
            continue
        if previous.string not in _FALLBACK_CALLS or (index >= 2 and tokens[index - 2].string in ("def", "class")):
            continue
        closing = bracket_pairs.get(index)
        if closing is None:
            continue
        tree = _recover_call(previous, tokens[closing], source, offsets)
        if tree is not None:
            tree.body.col_offset = previous.start[1]
            # Nested constructor calls are recovered separately.
            visitor._check_call(tree.body)

    # Match on tokenized logical lines so comments and strings cannot supply
    # synthetic subscript syntax.
    statement = []
    for token in tokens:
        if token.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING):
            statement.append(token)
        elif token.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
            statement_pairs = _bracket_pairs(statement)
            for index, (first, second) in enumerate(zip(statement, statement[1:])):
                if first.string != "hf_dataset" or second.string != "[":
                    continue
                closing = statement_pairs.get(index + 1)
                if closing is not None and closing + 1 < len(statement) and statement[closing + 1].string == "[":
                    line, col = first.start
                    visitor.findings.append(Finding(
                        path, line, col, "hf_dataset_full_row_index", "info",
                        lines[line - 1].strip(), _NOTES["hf_dataset_full_row_index"],
                    ))
            statement.clear()

    for finding in visitor.findings:
        finding.severity = "info"
        finding.note += " Partial Python parse; confirm the source context."
    return visitor.findings


def scan_python_file(path: str) -> List[Finding]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except OSError as exc:
        print("warning: skipped %s: %s" % (path, exc), file=sys.stderr)
        return []
    lines = source.splitlines()
    clean_source = "\n".join(
        "" if line.strip() == "from __future__ import annotations" else line
        for line in lines
    )
    try:
        tree = ast.parse(clean_source, filename=path)
    except _AST_PARSE_ERRORS:
        return _scan_python_fallback(path, source, lines)
    has_pin_def = any(
        isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name == "pin_memory"
        for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "DataContainer"
        for method in node.body
    )
    visitor = _PythonVisitor(path, lines, has_pin_def)
    visitor.visit(tree)
    return visitor.findings


_CONFIG_KEY_CHECKS = (
    (re.compile(r"^\s*(?:pin_memory|persistent_workers|non_blocking)\s*:\s*false\b", re.I), "medium"),
    (re.compile(r"^\s*(?:num_workers|workers_per_gpu)\s*:\s*0\b", re.I), "high"),
)
_JSON_STRING = re.compile(r'("(?:[^"\\]|\\.)*")\s*(:)?')


def scan_config_file(path: str) -> List[Finding]:
    try:
        if os.path.getsize(path) > _MAX_CONFIG_BYTES:
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw_text = fh.read()
    except OSError:
        return []

    findings: List[Finding] = []
    ext = os.path.splitext(path)[1].lower()
    lines = raw_text.splitlines()

    if ext == ".json":
        try:
            obj = json.loads(raw_text)
        except Exception:
            return []
        hits = []

        def _walk(node: object, location: str = "") -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    key_path = location + "." + k if location else k
                    kl = k.lower()
                    if kl in ("pin_memory", "persistent_workers", "non_blocking") and v is False:
                        hits.append((k, key_path, v, "medium"))
                    elif kl in ("num_workers", "workers_per_gpu") and v == 0 and not isinstance(v, bool):
                        hits.append((k, key_path, v, "high"))
                    _walk(v, key_path)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    _walk(item, "{}[{}]".format(location, i))

        _walk(obj)
        locations = {key: None for key, _, _, _ in hits}
        for match in _JSON_STRING.finditer(raw_text):
            if match.group(2):
                key = json.loads(match.group(1))
                if key in locations:
                    locations[key] = raw_text.count("\n", 0, match.start()) + 1 if locations[key] is None else 0
        for key, key_path, val, sev in hits:
            findings.append(
                Finding(path, locations[key] or 0, 0, "config_data_pipeline_flag", sev,
                        "%s: %r" % (key_path, val), _NOTES["config_data_pipeline_flag"])
            )
        return findings

    for idx, raw in enumerate(lines, 1):
        stripped = raw.split("#", 1)[0]
        if not stripped.strip():
            continue
        for regex, sev in _CONFIG_KEY_CHECKS:
            if regex.search(stripped):
                findings.append(
                    Finding(path, idx, 0, "config_data_pipeline_flag", sev, raw.strip(), _NOTES["config_data_pipeline_flag"])
                )
                break
    return findings


def _iter_targets(paths: Iterable[str], include_configs: bool) -> Iterable[Tuple[str, str]]:
    config_exts = {".yaml", ".yml", ".json"}
    for path in paths:
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext == ".py":
                yield path, "py"
            elif include_configs and ext in config_exts:
                yield path, "cfg"
        elif os.path.isdir(path):
            for root, dirs, names in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
                for name in sorted(names):
                    full = os.path.join(root, name)
                    ext = os.path.splitext(name)[1].lower()
                    if ext == ".py":
                        yield full, "py"
                    elif include_configs and ext in config_exts:
                        yield full, "cfg"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="Python/config files or directories to scan")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--min-severity",
        choices=("info", "low", "medium", "high"),
        default="info",
        help="Filter out findings below this severity (default: info)",
    )
    parser.add_argument(
        "--no-configs",
        action="store_true",
        help="Skip YAML/JSON config files and scan Python files only",
    )
    args = parser.parse_args(argv)
    missing = [path for path in args.paths if not os.path.exists(path)]
    if missing:
        print("error: path not found: " + ", ".join(missing), file=sys.stderr)
        return 2

    min_rank = _SEVERITY_ORDER[args.min_severity]
    findings: List[Finding] = []
    for path, kind in _iter_targets(args.paths, include_configs=not args.no_configs):
        found = scan_python_file(path) if kind == "py" else scan_config_file(path)
        for item in found:
            if _SEVERITY_ORDER[item.severity] >= min_rank:
                findings.append(item)

    findings.sort(key=lambda f: (-_SEVERITY_ORDER[f.severity], f.file, f.line, f.pattern))

    if args.json:
        print(json.dumps([f.to_dict() for f in findings], ensure_ascii=False, indent=2))
    elif not findings:
        print("No candidates found.")
    else:
        counts: Dict[str, int] = {}
        for f in findings:
            counts[f.pattern] = counts.get(f.pattern, 0) + 1
        summary = ", ".join("%d %s" % (n, p) for p, n in sorted(counts.items()))
        print("%d static candidate(s): %s\n" % (len(findings), summary))
        for f in findings:
            location = "%s:%d" % (f.file, f.line) if f.line else f.file
            print("%s  [%s | %s]" % (location, f.severity, f.pattern))
            print("    %s" % f.snippet)
            print("    note: %s\n" % f.note)

    print(
        "STATIC candidates only: they do not prove data loading or H2D copies are a real "
        "bottleneck, verify host RAM / /dev/shm headroom, or confirm speedup. Confirm the runtime "
        "path and use a profiler trace when available.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
