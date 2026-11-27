#!/usr/bin/env python3
"""Static scanner for PyTorch/ROCm backend runtime environment variables.

Scans block YAML RuntimeConfig, literal .env presets, simple shell launch lines,
and Python environment operations for the 5 categories of backend settings:

  precision_env_config  Backend precision/math env switches (Cat 1)
  precision_env_guard   Candidate precision/stability guards (Cat 1)
  layout_miopen_env     MIOpen search & layout env switches (Cat 2)
  memory_hw_env         HW queue, PCIe & allocator env switches (Cat 3)
  comm_thread_env       NCCL/RCCL & CPU thread env switches (Cat 4)
  topology_bound_env    High-risk machine/topology/device bindings (Cat 4/device)
  compile_cache_env     Inductor/Triton compile & cache env switches (Cat 5)
  env_unset             Explicit environment removal (RuntimeConfig/shell/Python unset)
  env_override_conflict Different literal assignments in one recognized scope

These are STATIC findings only. They do not prove the effective runtime value
after overrides, backend build support, or end-to-end speedup. Shell findings on
continuations refer to the logical command's starting line. Check dynamic
shell/Python control flow, YAML anchors/flow mappings, and sourced files manually.

Usage:
    python3 scan_runtime_env.py <file-or-dir> [...]
    python3 scan_runtime_env.py --json <file-or-dir> [...]

Pure Python standard library (Python 3.6+ compatible).
Exit 1 indicates a partial or empty scan; exit 2 indicates invalid CLI arguments.
"""

import argparse
import ast
import json
import os
import re
import shlex
import sys
import tokenize
from collections import namedtuple
from typing import Dict, Iterable, List, Optional, Tuple

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
_MAX_TEXT_BYTES = 2 * 1024 * 1024
_DISCLAIMER = (
    "STATIC findings only: they do not prove final runtime precedence after shell/CLI "
    "overrides, target ROCm/CUDA build support, numerical tolerance, or steady-state speedup."
)

# Category definitions
_PRECISION_ENV_KEYS = {
    "HIPBLASLT_ALLOW_TF32",
    "TORCH_BLAS_PREFER_HIPBLASLT",
    "ROCBLAS_MATH_MODE",
    "MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32",
    "ENABLE_MIOPEN_DEFORMER_CONV_FP16",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "TRITON_F32_DEFAULT",
    "DISABLE_ADDMM_CUDA_LT",
    "TORCHINDUCTOR_EMULATE_PRECISION_CASTS",
}
_LAYOUT_MIOPEN_KEYS = {
    "MIOPEN_FIND_MODE",
    "MIOPEN_FIND_ENFORCE",
    "PYTORCH_MIOPEN_SUGGEST_NHWC",
    "PYTORCH_MIOPEN_SUGGEST_NDHWC",
}
_MEMORY_HW_KEYS = {
    "GPU_MAX_HW_QUEUES",
    "HSA_FORCE_FINE_GRAIN_PCIE",
    "HIP_BUFFER_EXTRA_SIZE",
    "HIPBLASLT_WORKSPACE_SIZE",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
}
_COMM_THREAD_KEYS = {
    "NCCL_ALGO",
    "ALLREDUCE_STREAM_WITH_COMPUTE",
    "TORCH_NCCL_ENABLE_MONITORING",
    "TOKENIZERS_PARALLELISM",
    "OMP_NUM_THREADS",
}
_TOPOLOGY_BOUND_KEYS = {
    "NCCL_RINGS",
    "NCCL_TOPO_FILE",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
}
_COMPILE_CACHE_KEYS = {
    "TORCHINDUCTOR_LAYOUT_OPTIMIZATION",
    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING",
    "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "TORCHINDUCTOR_ALIGNMENT_ASSERTS",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCH_COMPILE_CACHE_KEY_TAG",
}

_ALL_TRACKED_KEYS = (
    _PRECISION_ENV_KEYS
    | _LAYOUT_MIOPEN_KEYS
    | _MEMORY_HW_KEYS
    | _COMM_THREAD_KEYS
    | _TOPOLOGY_BOUND_KEYS
    | _COMPILE_CACHE_KEYS
)

_NOTES = {
    "precision_env_config": (
        "Backend precision/math env switch (Cat 1); verify target build support, actual math "
        "dispatch, and training/inference numerical tolerance in a new process."
    ),
    "precision_env_guard": (
        "Explicit precision or backend restriction (Cat 1); verify whether it protects "
        "correctness or stability before changing it."
    ),
    "layout_miopen_env": (
        "MIOpen search or layout env switch (Cat 2); measure cold-start search time separately "
        "from steady-state time and verify shape stability."
    ),
    "memory_hw_env": (
        "Hardware queue, PCIe, or allocator env switch (Cat 3); check target ROCm/PyTorch build "
        "support, peak memory/retries, and graph compatibility."
    ),
    "comm_thread_env": (
        "Collective communication or CPU thread env switch (Cat 4); verify multi-rank stability "
        "(no hang/deadlock) and CPU core contention."
    ),
    "topology_bound_env": (
        "High-risk machine topology or device mask binding; do not reuse blindly across different "
        "nodes, interconnects, or GPU counts."
    ),
    "compile_cache_env": (
        "Inductor/Triton compile or cache env switch (Cat 5); separate first-compile time from "
        "steady-state time and verify cache isolation (check /dev/shm capacity if used)."
    ),
    "env_override_conflict": (
        "Different literal values appear in the same recognized assignment scope; verify intent "
        "and effective precedence."
    ),
}


class Finding(object):
    __slots__ = ("file", "line", "pattern", "severity", "variable", "value", "snippet", "note")

    def __init__(self, file, line, pattern, severity, variable, value, snippet, note):
        # type: (str, int, str, str, str, str, str, str) -> None
        self.file = file
        self.line = line
        self.pattern = pattern
        self.severity = severity
        self.variable = variable
        self.value = value
        self.snippet = snippet
        self.note = note

    def to_dict(self):
        # type: () -> Dict[str, object]
        return {
            "file": self.file,
            "line": self.line,
            "pattern": self.pattern,
            "severity": self.severity,
            "variable": self.variable,
            "value": self.value,
            "snippet": self.snippet,
            "note": self.note,
        }


def _classify(key, val):
    # type: (str, str) -> Tuple[str, str, str]
    low = val.strip().lower()
    if key in _TOPOLOGY_BOUND_KEYS:
        return "topology_bound_env", "high", _NOTES["topology_bound_env"]
    if key in _PRECISION_ENV_KEYS:
        if ((key == "HIPBLASLT_ALLOW_TF32" and low == "0")
                or (key == "DISABLE_ADDMM_CUDA_LT" and low in ("1", "true"))):
            return "precision_env_guard", "medium", _NOTES["precision_env_guard"]
        return "precision_env_config", "info", _NOTES["precision_env_config"]
    if key in _LAYOUT_MIOPEN_KEYS:
        return "layout_miopen_env", "info", _NOTES["layout_miopen_env"]
    if key in _MEMORY_HW_KEYS:
        return "memory_hw_env", "info", _NOTES["memory_hw_env"]
    if key in _COMM_THREAD_KEYS:
        return "comm_thread_env", "info", _NOTES["comm_thread_env"]
    note = _NOTES["compile_cache_env"]
    sev = "info"
    if key == "TORCHINDUCTOR_ALIGNMENT_ASSERTS" and low == "0":
        sev = "medium"
        note = (
            "Disables Inductor alignment asserts (build workaround / correctness risk); verify "
            "whether the target PyTorch build still needs this."
        )
    elif key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR") and "/dev/shm" in val:
        sev = "medium"
        note = (
            "Compile cache placed under /dev/shm; verify shared-memory capacity and multi-process "
            "isolation to prevent out-of-space compile failures."
        )
    return "compile_cache_env", sev, note


_Entry = namedtuple("_Entry", "line key value snippet scope operation")


def _record_and_check_conflicts(path, entries):
    findings = []  # type: List[Finding]
    seen_values = {}
    reported_conflict = set()
    for lineno, key, val, snippet, scope, operation in entries:
        if operation == "unset":
            note = "Explicit removal in {}; check whether this layer applies and which later overrides run.".format(scope)
            findings.append(Finding(path, lineno, "env_unset", "info", key, val, snippet, note))
            continue
        pat, sev, note = _classify(key, val)
        if scope and scope.startswith("environment."):
            note = "{}: {}".format(scope, note)
        if operation == "default":
            note = "Conditional default (setdefault); {}".format(note)
        findings.append(Finding(path, lineno, pat, sev, key, val, snippet, note))
        if not scope or operation != "set" or val == "<dynamic>":
            continue
        slot = (scope, key)
        previous = seen_values.get(slot)
        if previous and previous[1] != val and slot not in reported_conflict:
            reported_conflict.add(slot)
            first_line, first_val = previous
            conflict_note = "{} (line {} set {!r}, line {} set {!r})".format(
                _NOTES["env_override_conflict"], first_line, first_val, lineno, val
            )
            findings.append(Finding(path, lineno, "env_override_conflict", "medium", key, val, snippet, conflict_note))
        seen_values[slot] = (lineno, val)
    return findings


def _strip_quotes(raw_val):
    # type: (str) -> str
    val = raw_val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        return val[1:-1]
    return val


def _read_text_limited(path):
    # type: (str) -> Tuple[Optional[List[str]], Optional[str]]
    try:
        with open(path, "rb") as fh:
            data = fh.read(_MAX_TEXT_BYTES + 1)
        if len(data) > _MAX_TEXT_BYTES:
            raise ValueError("file exceeds the {} byte scan limit".format(_MAX_TEXT_BYTES))
        return data.decode("utf-8-sig").splitlines(), None
    except (OSError, ValueError) as exc:
        reason = str(exc)
        print("warning: skipped {}: {}".format(path, reason), file=sys.stderr)
        return None, reason


_RE_YAML_KV = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*:(?:\s+(?P<val>.*))?$")
_RE_YAML_SCALAR = re.compile(r"""^(?:'(?:[^']|'')*'|"(?:[^"\\]|\\.)*"|[^'"\[\]{},&*!>|][^\[\]{},]*)$""")
_RE_YAML_BLOCK_SCALAR = re.compile(
    r"""^\s*(?:-\s+[^#\n]*:\s*|[^#\n]+:\s*|-\s+)[|>](?:[+-]\d*|\d+[+-]?)?\s*(?:#.*)?$"""
)
_YAML_SET = {("environment", "set"), ("environment", "accuracy", "set")}
_YAML_UNSET = {("environment", "unset"), ("environment", "accuracy", "unset")}
_YAML_GENERIC_ENV_PARENTS = {"env", "environment", "variables"}


def _line_syntax(line, quote=None):
    positions = []
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index], positions, quote, False
        elif char in ";&|()<>":
            positions.append((index, char))
    return line, positions, quote, escaped


def _yaml_unsets(names, lineno, scope, snippet):
    for name in names:
        key = _strip_quotes(name)
        if key in _ALL_TRACKED_KEYS:
            yield _Entry(lineno, key, "<unset>", snippet, ".".join(scope), "unset")


def scan_yaml_file(path):
    # type: (str) -> Tuple[Optional[List[Finding]], Optional[str]]
    lines, reason = _read_text_limited(path)
    if lines is None:
        return None, reason

    entries = []
    sections = []
    block_indent = None  # type: Optional[int]
    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.lstrip(" \t")
        indent = len(raw) - len(stripped)
        if block_indent is not None:
            if not stripped or indent > block_indent:
                continue
            block_indent = None
        text = _line_syntax(stripped)[0].rstrip()
        if not text:
            continue
        is_list_item = text.startswith("- ")
        while sections and (indent < sections[-1][0] or (indent == sections[-1][0] and not is_list_item)):
            sections.pop()
        scope = tuple(key for _, key in sections)
        if _RE_YAML_BLOCK_SCALAR.match(raw):
            block_indent = indent
            continue
        if is_list_item and scope in _YAML_UNSET:
            entries.extend(_yaml_unsets([text[2:]], lineno, scope, stripped))
            continue
        m = _RE_YAML_KV.match(text)
        if not m:
            continue
        key, val = m.group("key", "val")
        if val is None:
            sections.append((indent, key))
        elif scope + (key,) in _YAML_UNSET and val.startswith("[") and val.endswith("]"):
            entries.extend(_yaml_unsets(val[1:-1].split(","), lineno, scope + (key,), stripped))
        elif key in _ALL_TRACKED_KEYS and (not scope or scope in _YAML_SET or scope[-1] in _YAML_GENERIC_ENV_PARENTS) and _RE_YAML_SCALAR.match(val):
            where = ".".join(scope) if scope else "YAML root"
            entries.append(_Entry(lineno, key, _strip_quotes(val), stripped, where, "set"))
    return _record_and_check_conflicts(path, entries), None


_RE_HEREDOC = re.compile(r"""<<-?\s*(?:'([^']+)'|"([^"]+)"|([A-Za-z0-9_-]+))(?=$|[\s;&|<>])""")
_RE_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_RE_LAUNCH_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SHELL_FLOW = {"if", "then", "elif", "else", "fi", "case", "esac", "for", "while", "until",
               "do", "done", "function", "{", "}", "source", ".", "eval", "unset"}


def _shell_line(code):
    """Split simple commands at unquoted separators and locate heredoc starts."""
    commands = []
    heredocs = []
    start = 0
    complex_scope = False
    for index, char in _line_syntax(code)[1]:
        if char == "<" and code[max(0, index - 1):index] != "<" and not code.startswith("<<<", index):
            match = _RE_HEREDOC.match(code, index)
            if match:
                heredocs.append(next(group for group in match.groups() if group is not None))
        if char in "()":
            # Later tokens may be a substitution, subshell, or function body.
            complex_scope = True
            break
        if char in ";&|":
            commands.append(code[start:index])
            start = index + 1
            if char in "&|":
                complex_scope = True
    commands.append(code[start:])
    return commands, heredocs, complex_scope


def _shell_assignments(command):
    try:
        words = shlex.split(command, comments=False)
    except ValueError:
        return [], False
    if not words:
        return [], False
    first = words[0]
    exported = first == "export"
    entries = []
    if first == "unset":
        for word in words[1:]:
            if word.startswith("-"):
                continue
            if _RE_ENV_NAME.fullmatch(word) and word in _ALL_TRACKED_KEYS:
                entries.append((word, "<unset>", False, "unset"))
        return entries, True
    if first == "export":
        words = words[1:]
    elif first == "env":
        rest = []
        idx = 1
        while idx < len(words):
            token = words[idx]
            if token == "--":
                rest.extend(words[idx + 1:])
                break
            if token in ("-u", "--unset"):
                if idx + 1 < len(words):
                    target = words[idx + 1]
                    if _RE_ENV_NAME.fullmatch(target) and target in _ALL_TRACKED_KEYS:
                        entries.append((target, "<unset>", False, "unset"))
                    idx += 2
                    continue
                break
            if token.startswith("--unset="):
                target = token.split("=", 1)[1]
                if _RE_ENV_NAME.fullmatch(target) and target in _ALL_TRACKED_KEYS:
                    entries.append((target, "<unset>", False, "unset"))
                idx += 1
                continue
            if token.startswith("-"):
                idx += 1
                continue
            rest.extend(words[idx:])
            break
        words = rest
    elif not _RE_LAUNCH_ASSIGN.match(command.lstrip()):
        return [], first in _SHELL_FLOW
    for word in words:
        key, equals, value = word.partition("=")
        if not equals or not _RE_ENV_NAME.fullmatch(key):
            if exported and _RE_ENV_NAME.fullmatch(word):
                continue
            break
        if key in _ALL_TRACKED_KEYS:
            value = "<dynamic>" if "$" in value or "`" in value else value
            entries.append((key, value, exported, "set"))
    return entries, first in _SHELL_FLOW


def scan_shell_file(path):
    lines, reason = _read_text_limited(path)
    if lines is None:
        return None, reason

    entries = []
    heredocs = []
    complex_scope = False
    parts = []
    quote = None
    for lineno, raw in enumerate(lines, start=1):
        if heredocs:
            if raw.lstrip("\t") == heredocs[0]:
                heredocs.pop(0)
            continue
        if not parts:
            command_line = lineno
        text, _, quote, continued = _line_syntax(raw, quote)
        parts.append(text[:-1] if continued else text)
        if continued:
            continue
        if quote:
            parts.append("\n")
            continue
        code = "".join(parts)
        parts.clear()
        commands, heredocs, complex_line = _shell_line(code)
        complex_scope |= complex_line
        for command in commands:
            matches, control = _shell_assignments(command)
            complex_scope |= control
            for key, val, exported, operation in matches:
                if operation == "unset":
                    scope = "shell unset"
                else:
                    scope = "shell export" if exported else None
                entries.append(_Entry(command_line, key, val, lines[command_line - 1].strip(), scope, operation))
    if complex_scope:
        entries = [entry._replace(scope=None) if entry.operation != "unset" else entry for entry in entries]
    return _record_and_check_conflicts(path, entries), None


def scan_dotenv_file(path):
    """Read literal KEY=VALUE presets, matching the OpenWAM input format."""
    lines, reason = _read_text_limited(path)
    if lines is None:
        return None, reason
    entries = []
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if "\0" not in raw and (not line or line.startswith("#")):
            continue
        raw_key, equals, value = line.partition("=")
        key, value = raw_key.strip(), value.strip()
        bad_quotes = value[:1] in ("'", '"') and (len(value) < 2 or value[-1] != value[0])
        if not equals or not _RE_ENV_NAME.fullmatch(key) or bad_quotes or "\0" in raw:
            reason = "line {}: expected a literal KEY=VALUE preset".format(lineno)
            print("warning: skipped {}: {}".format(path, reason), file=sys.stderr)
            return None, reason
        if key in _ALL_TRACKED_KEYS:
            entries.append(_Entry(lineno, key, _strip_quotes(value), line, "dotenv preset", "set"))
    return _record_and_check_conflicts(path, entries), None


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


_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _local_nodes(node):
    yield node
    if not isinstance(node, _DEFINITIONS + (ast.Lambda,) + _COMPREHENSIONS):
        for child in ast.iter_child_nodes(node):
            yield from _local_nodes(child)


def _local_bindings(body, params=None):
    bindings = {}
    if params is not None:
        bindings.update((node.arg, None) for node in ast.iter_child_nodes(params) if isinstance(node, ast.arg))
    for statement in body:
        for node in _local_nodes(statement):
            if isinstance(node, ast.Import) or (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
                for alias in node.names:
                    target = "{}.{}".format(node.module, alias.name) if isinstance(node, ast.ImportFrom) else alias.name
                    local = alias.asname or alias.name.split(".", 1)[0]
                    if isinstance(node, ast.Import) and alias.asname is None:
                        target = local
                    if local in bindings and bindings[local] != target:
                        target = None
                    bindings[local] = target
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bindings[node.id] = None
            elif isinstance(node, _DEFINITIONS):
                bindings[node.name] = None
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                bindings.update((name, None) for name in node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bindings[node.name] = None
    return bindings


class _PyEnvVisitor(ast.NodeVisitor):
    def __init__(self, lines, tree):
        # type: (List[str], ast.Module) -> None
        self.lines = lines
        self.entries = []  # type: List[_Entry]
        self._top_level = {id(n.value if isinstance(n, ast.Expr) else n) for n in tree.body}
        self._scopes = [(_local_bindings(tree.body), False)]

    def _snippet(self, line):
        # type: (int) -> str
        return self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""

    def _name(self, node):
        head, sep, tail = _dotted(node).partition(".")
        for index, (bindings, is_class) in enumerate(reversed(self._scopes)):
            if head in bindings and (index == 0 or not is_class):
                target = bindings[head]
                return target + sep + tail if target else ""
        return ""

    def _visit_scope(self, body, params=None, is_class=False):
        self._scopes.append((_local_bindings(body, params), is_class))
        try:
            for statement in body:
                self.visit(statement)
        finally:
            self._scopes.pop()

    def visit_FunctionDef(self, node):
        self._visit_scope(node.body, node.args)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._visit_scope(node.body, is_class=True)

    def visit_Lambda(self, node):
        self._visit_scope([node.body], node.args)

    def visit_ListComp(self, node):
        # Dynamic iteration can introduce a shadow binding in its own scope.
        return None

    visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_ListComp

    def _record(self, node, key, value, operation="set"):
        if key in _ALL_TRACKED_KEYS:
            scope = "Python module" if id(node) in self._top_level else None
            val = _str_const(value)
            self.entries.append(_Entry(node.lineno, key, val if val is not None else "<dynamic>",
                                       self._snippet(node.lineno), scope, operation))

    def visit_If(self, node):
        condition = _bool_const(node.test)
        if condition is None:
            self.generic_visit(node)
        else:
            for statement in (node.body if condition else node.orelse):
                self.visit(statement)

    def visit_Assign(self, node):
        # type: (ast.Assign) -> None
        for tgt in node.targets:
            if isinstance(tgt, ast.Subscript) and self._name(tgt.value) == "os.environ":
                subscript = tgt.slice.value if hasattr(ast, "Index") and isinstance(tgt.slice, ast.Index) else tgt.slice
                self._record(node, _str_const(subscript), node.value)
        self.generic_visit(node)

    def visit_Delete(self, node):
        # type: (ast.Delete) -> None
        for tgt in node.targets:
            if isinstance(tgt, ast.Subscript) and self._name(tgt.value) == "os.environ":
                subscript = tgt.slice.value if hasattr(ast, "Index") and isinstance(tgt.slice, ast.Index) else tgt.slice
                key = _str_const(subscript)
                if key in _ALL_TRACKED_KEYS:
                    self.entries.append(_Entry(node.lineno, key, "<unset>", self._snippet(node.lineno),
                                               "Python del os.environ", "unset"))
        self.generic_visit(node)

    def visit_Call(self, node):
        # type: (ast.Call) -> None
        func_name = self._name(node.func)
        if func_name in ("os.environ.setdefault", "os.putenv"):
            if len(node.args) >= 2:
                operation = "default" if func_name == "os.environ.setdefault" else "putenv"
                self._record(node, _str_const(node.args[0]), node.args[1], operation)
        elif func_name == "os.environ.update":
            if node.args and isinstance(node.args[0], ast.Dict):
                for k_node, v_node in zip(node.args[0].keys, node.args[0].values):
                    self._record(node, _str_const(k_node), v_node)
            for kw in node.keywords:
                self._record(node, kw.arg, kw.value)
        elif func_name in ("os.environ.pop", "os.unsetenv"):
            if node.args:
                key = _str_const(node.args[0])
                if key in _ALL_TRACKED_KEYS:
                    self.entries.append(_Entry(node.lineno, key, "<unset>", self._snippet(node.lineno),
                                               "Python " + func_name, "unset"))
        self.generic_visit(node)


def scan_python_file(path):
    # type: (str) -> Tuple[Optional[List[Finding]], Optional[str]]
    try:
        with tokenize.open(path) as fh:
            source = fh.read()
        tree = ast.parse(source, filename=path)
        lines = source.splitlines()
        visitor = _PyEnvVisitor(lines, tree)
        visitor.visit(tree)
    except (OSError, SyntaxError, ValueError, RecursionError) as exc:
        reason = str(exc)
        print("warning: skipped {}: {}".format(path, reason), file=sys.stderr)
        return None, reason
    return _record_and_check_conflicts(path, visitor.entries), None


def _file_kind(path):
    # type: (str) -> Optional[str]
    low = path.lower()
    if low.endswith((".yaml", ".yml")):
        return "yaml"
    if low.endswith((".sh", ".bash")):
        return "sh"
    if low.endswith(".env"):
        return "dotenv"
    if low.endswith(".py"):
        return "py"
    return None


def _iter_targets(paths, on_error):
    # type: (Iterable[str], object) -> Iterable[Tuple[str, str]]
    for target in paths:
        if os.path.isfile(target):
            kind = _file_kind(target)
            if kind:
                yield (kind, target)
        elif os.path.isdir(target):
            for root, dirs, files in os.walk(target, onerror=on_error):
                dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
                for name in sorted(files):
                    full = os.path.join(root, name)
                    kind = _file_kind(full)
                    if kind:
                        yield (kind, full)


def main(argv=None):
    # type: (Optional[List[str]]) -> int
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="YAML configs, .env presets, shell/Python launchers, or directories.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    for p in args.paths:
        if not (os.path.isdir(p) or (os.path.isfile(p) and _file_kind(p) is not None)):
            parser.error("expected an existing YAML/.env/Shell/Python file or directory: {}".format(p))

    findings = []  # type: List[Finding]
    scanned = 0
    skipped_files = []  # type: List[Dict[str, str]]

    def walk_error(err):
        name = getattr(err, "filename", None) or "<walk>"
        reason = str(err)
        skipped_files.append({"file": str(name), "reason": reason})
        print("warning: skipped {}".format(reason), file=sys.stderr)

    for kind, path in _iter_targets(args.paths, walk_error):
        if kind == "yaml":
            res, reason = scan_yaml_file(path)
        elif kind == "sh":
            res, reason = scan_shell_file(path)
        elif kind == "dotenv":
            res, reason = scan_dotenv_file(path)
        else:
            res, reason = scan_python_file(path)
        if res is None:
            skipped_files.append({"file": path, "reason": reason or "unknown error"})
        else:
            scanned += 1
            findings.extend(res)

    findings.sort(key=lambda f: (f.file, f.line, f.pattern, f.variable))
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
        print("No tracked runtime environment settings found.")
    else:
        print("{} static finding(s):\n".format(len(findings)))
        for f in findings:
            print("{}:{} [{}] ({}) {}={} :: {}".format(
                f.file, f.line, f.severity.upper(), f.pattern, f.variable, f.value, f.snippet
            ))
            print("    note: {}\n".format(f.note))

    print(_DISCLAIMER, file=sys.stderr)
    return 1 if skipped or not scanned else 0


if __name__ == "__main__":
    raise SystemExit(main())
