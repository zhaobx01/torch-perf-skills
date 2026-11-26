#!/usr/bin/env python3
"""Static scanner for distributed communication candidates in PyTorch projects.

This STATIC scanner reads Python ASTs and DeepSpeed JSON or simple YAML.
It finds DDP/FSDP/ZeRO settings, explicit tensor and object collectives,
and potential loop or custom-stream ordering issues.

It does NOT prove that communication is a bottleneck, distinguish true
exposed communication from slow-rank skew, or guarantee that enabling a
flag (such as static_graph=True or reshard_after_forward=False) is safe for
the target model's control flow or memory budget. Confirm performance
candidates with trace-triage (T5) when trace evidence is available.

Usage:
    python3 scan_comm_candidates.py <file-or-dir> [<file-or-dir> ...]
    python3 scan_comm_candidates.py --json <file-or-dir> ...
    python3 scan_comm_candidates.py --min-severity medium <file-or-dir> ...
    python3 scan_comm_candidates.py --no-configs <file-or-dir> ...

Uses the Python standard library (3.6+). Use a Python version that can parse
the project for import and constant-config resolution. Dynamic options and
YAML aliases/merges within the supported keys require manual inspection.
"""

import argparse
import ast
import io
import json
import os
import re
import sys
import tokenize
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional


_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules", ".venv", "venv"}
_MAX_CONFIG_BYTES = 2 * 1024 * 1024

_DDP_NAMES = {
    "torch.nn.parallel.DistributedDataParallel",
    "torch.nn.parallel.distributed.DistributedDataParallel",
    "mmcv.parallel.MMDistributedDataParallel",
    "mmcv.parallel.distributed.MMDistributedDataParallel",
}
_DDP_KWARGS_NAMES = {
    "accelerate.DistributedDataParallelKwargs",
    "accelerate.utils.DistributedDataParallelKwargs",
    "accelerate.utils.dataclasses.DistributedDataParallelKwargs",
}
_FSDP1_NAMES = {
    "torch.distributed.fsdp.FullyShardedDataParallel",
    "torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel",
}
_FSDP2_NAMES = {
    "torch.distributed.fsdp.fully_shard",
    "torch.distributed._composable.fsdp.fully_shard",
}
# Zero-based positional index of async_op in each supported collective API.
_COLLECTIVE_ASYNC_POSITION = {
    "all_gather": 3,
    "all_gather_into_tensor": 3,
    "_all_gather_base": 3,
    "all_reduce": 3,
    "reduce_scatter": 4,
    "reduce_scatter_tensor": 4,
    "_reduce_scatter_base": 4,
    "all_to_all": 3,
    "all_to_all_single": 5,
    "broadcast": 3,
    "reduce": 4,
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
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _const_bool(node: Optional[ast.AST]) -> Optional[bool]:
    if node is None:
        return None
    if isinstance(node, ast.NameConstant) and isinstance(node.value, bool):
        return node.value
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _known_sync_call(node, options, unknown_options, async_position):
    if unknown_options or any(isinstance(arg, ast.Starred) for arg in node.args):
        return False
    if "async_op" in options:
        return _const_bool(options["async_op"]) is False
    if len(node.args) > async_position:
        return _const_bool(node.args[async_position]) is False
    return True


def _can_check_wrapper_defaults(node, unknown_options):
    return (
        not unknown_options
        and len(node.args) <= 1
        and not any(isinstance(arg, ast.Starred) for arg in node.args)
    )


class _LocalNames(ast.NodeVisitor):
    """Find lexical shadows without entering child scopes."""

    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Import(self, node):
        self.names.update(item.asname or item.name.split(".")[0] for item in node.names)

    def visit_ImportFrom(self, node):
        self.visit_Import(node)

    def visit_FunctionDef(self, node):
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_ExceptHandler(self, node):
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node):
        if node.rest:
            self.names.add(node.rest)
        self.generic_visit(node)

    def visit_ListComp(self, node):
        pass

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


class _PyVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, source_lines: List[str]) -> None:
        self.filename = filename
        self.source_lines = source_lines
        self.findings: List[Finding] = []
        self._loop_depth = 0
        self._stream_depth = 0
        self._scopes = [{}]
        self._class_scopes = [False]

    def _binding(self, name):
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def _qualified_name(self, node):
        name = _dotted_name(node)
        head, _, tail = name.partition(".")
        imported = self._binding(head)
        if isinstance(imported, str):
            return ".".join((imported, tail)) if tail else imported
        return ""

    def visit_Import(self, node):
        for item in node.names:
            head = item.name.split(".")[0]
            self._scopes[-1][item.asname or head] = item.name if item.asname else head

    def visit_ImportFrom(self, node):
        for item in node.names:
            name = item.asname or item.name
            self._scopes[-1][name] = f"{node.module}.{item.name}" if node.level == 0 else None

    def _mapping(self, node):
        if isinstance(node, ast.Dict):
            keys = [key.s if isinstance(key, ast.Str) else None for key in node.keys]
            if None not in keys and len(set(keys)) == len(keys):
                return dict(zip(keys, node.values))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
            and not any("dict" in scope for scope in self._scopes)
            and not node.args
            and all(kw.arg is not None for kw in node.keywords)
        ):
            return {kw.arg: kw.value for kw in node.keywords}
        return None

    def _call_options(self, node):
        options = {}
        unknown = False
        for kw in node.keywords:
            if kw.arg is not None:
                options[kw.arg] = kw.value
                continue
            value = kw.value
            if isinstance(value, ast.Name):
                # Track mutable config only in its defining scope.
                mapping = self._scopes[-1].get(value.id)
            else:
                mapping = self._mapping(value)
            if not isinstance(mapping, dict) or options.keys() & mapping.keys():
                unknown = True
            else:
                options.update(mapping)
        return options, unknown

    def _invalidate(self, node):
        if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            root = _dotted_name(node).split(".")[0]
            if isinstance(node, ast.Subscript):
                root = _dotted_name(node.value).split(".")[0]
            if isinstance(node, ast.Name) or isinstance(self._binding(root), dict):
                self._scopes[-1][root] = None
        elif isinstance(node, (ast.Tuple, ast.List)):
            for value in node.elts:
                self._invalidate(value)
        elif isinstance(node, ast.Starred):
            self._invalidate(node.value)

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            self._invalidate(target)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self._scopes[-1][node.targets[0].id] = self._mapping(node.value)
        if isinstance(node.value, ast.Name) and isinstance(self._binding(node.value.id), dict):
            self._invalidate(node.value)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit(node.value)
            self._invalidate(node.target)
            if isinstance(node.target, ast.Name):
                self._scopes[-1][node.target.id] = self._mapping(node.value)
            if isinstance(node.value, ast.Name) and isinstance(self._binding(node.value.id), dict):
                self._invalidate(node.value)

    def visit_AugAssign(self, node):
        self.visit(node.value)
        self._invalidate(node.target)

    def visit_Delete(self, node):
        for target in node.targets:
            self._invalidate(target)

    @contextmanager
    def _nested_scope(self, bindings, *, is_class=False):
        previous_scopes = self._scopes
        previous_classes = self._class_scopes
        # Class namespace lookups apply only to the class body. A child scope
        # can close over a surrounding function, but skips class attributes.
        self._scopes = [scope for scope, is_old_class in zip(previous_scopes, previous_classes) if not is_old_class]
        self._class_scopes = [False] * len(self._scopes) + [is_class]
        self._scopes.append(bindings)
        try:
            yield
        finally:
            self._scopes = previous_scopes
            self._class_scopes = previous_classes

    def _visit_scope(self, body, arguments=None, *, is_class=False):
        local = _LocalNames()
        for stmt in body:
            local.visit(stmt)
        if arguments is not None:
            args = list(getattr(arguments, "posonlyargs", [])) + arguments.args + arguments.kwonlyargs
            args.extend(arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None)
            local.names.update(arg.arg for arg in args)
        loop_depth, stream_depth = self._loop_depth, self._stream_depth
        self._loop_depth = self._stream_depth = 0
        with self._nested_scope(dict.fromkeys(local.names), is_class=is_class):
            for stmt in body:
                self.visit(stmt)
        self._loop_depth, self._stream_depth = loop_depth, stream_depth

    def visit_FunctionDef(self, node):
        self._scopes[-1][node.name] = None
        self._visit_scope(node.body, node.args)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._scopes[-1][node.name] = None
        self._visit_scope(node.body, is_class=True)

    def visit_Lambda(self, node):
        self._visit_scope([node.body], node.args)

    def _visit_branches(self, *branches):
        scope = self._scopes[-1]
        changed = set()
        for statements in branches:
            trial = scope.copy()
            self._scopes[-1] = trial
            for stmt in statements:
                self.visit(stmt)
            changed.update(key for key in trial if key not in scope or trial[key] != scope[key])
        self._scopes[-1] = scope
        scope.update(dict.fromkeys(changed))

    def visit_If(self, node):
        self.visit(node.test)
        self._visit_branches(node.body, node.orelse)

    def visit_Try(self, node):
        self._visit_branches(node.body)
        self._visit_branches(node.orelse, *([handler] for handler in node.handlers))
        for stmt in node.finalbody:
            self.visit(stmt)

    def visit_ExceptHandler(self, node):
        if node.name:
            self._scopes[-1][node.name] = None
        self.generic_visit(node)

    def visit_Match(self, node):
        self.visit(node.subject)
        self._visit_branches(*([case] for case in node.cases))

    def visit_match_case(self, node):
        captures = _LocalNames()
        captures.visit(node.pattern)
        self._scopes[-1].update(dict.fromkeys(captures.names))
        if node.guard is not None:
            self.visit(node.guard)
        for stmt in node.body:
            self.visit(stmt)

    def _visit_comprehension(self, node, outputs):
        first, *remaining = node.generators
        self.visit(first.iter)
        self._loop_depth += 1
        with self._nested_scope({}):
            self._invalidate(first.target)
            for cond in first.ifs:
                self.visit(cond)
            for generator in remaining:
                self.visit(generator.iter)
                self._invalidate(generator.target)
                for cond in generator.ifs:
                    self.visit(cond)
            for output in outputs:
                self.visit(output)
        self._loop_depth -= 1

    def visit_ListComp(self, node):
        self._visit_comprehension(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        self._visit_comprehension(node, [node.key, node.value])

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

    def visit_For(self, node):
        self.visit(node.iter)
        self._invalidate(node.target)
        self._loop_depth += 1
        self._visit_branches(node.body)
        self._loop_depth -= 1
        self._visit_branches(node.orelse)

    visit_AsyncFor = visit_For

    def visit_While(self, node):
        self.visit(node.test)
        self._loop_depth += 1
        self._visit_branches(node.body)
        self._loop_depth -= 1
        self._visit_branches(node.orelse)

    def visit_With(self, node):
        entered = False
        for item in node.items:
            expr = item.context_expr
            if isinstance(expr, ast.Call) and self._qualified_name(expr.func) == "torch.cuda.stream":
                entered = True
            self.visit(expr)
            if item.optional_vars is not None:
                self._invalidate(item.optional_vars)
        if entered:
            self._stream_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        if entered:
            self._stream_depth -= 1

    visit_AsyncWith = visit_With

    def visit_Call(self, node: ast.Call) -> None:
        dotted = self._qualified_name(node.func)
        namespace, _, short = dotted.rpartition(".")
        kwargs, unknown_options = self._call_options(node)

        # 1. DDP constructor / Accelerate DistributedDataParallelKwargs
        if dotted in _DDP_NAMES or dotted in _DDP_KWARGS_NAMES:
            find_unused = _const_bool(kwargs.get("find_unused_parameters"))
            if find_unused is True and _const_bool(kwargs.get("static_graph")) is not True:
                self._emit(
                    node,
                    "ddp_find_unused_true",
                    "high",
                    "DDP find_unused_parameters=True; check whether static_graph is active and whether the reducer needs unused-parameter search (C1).",
                )
            if dotted in _DDP_NAMES and _can_check_wrapper_defaults(node, unknown_options):
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
        elif dotted in _FSDP1_NAMES and (node.args or node.keywords):
            limit_ag = _const_bool(kwargs.get("limit_all_gathers"))
            if limit_ag is True:
                self._emit(
                    node,
                    "fsdp1_limit_all_gathers_true",
                    "high",
                    "FSDP1 limit_all_gathers=True protects memory; confirm exposed GPU communication and memory headroom before evaluating False (C2).",
                )
            missing_pf = [k for k in ("forward_prefetch", "backward_prefetch") if k not in kwargs]
            if missing_pf and _can_check_wrapper_defaults(node, unknown_options):
                self._emit(
                    node,
                    "fsdp1_missing_prefetch",
                    "medium",
                    "FSDP1 constructor omits %s; check version defaults and assess forward prefetch for fixed layer order and available memory (C2)." % ", ".join(missing_pf),
                )

        # 3. FSDP2 fully_shard(...)
        elif dotted in _FSDP2_NAMES and (node.args or node.keywords):
            reshard = _const_bool(kwargs.get("reshard_after_forward"))
            if reshard is True:
                self._emit(
                    node,
                    "fsdp2_reshard_after_forward_true",
                    "medium",
                    "FSDP2 reshard_after_forward=True re-gathers weights in backward; if memory headroom permits, evaluate False (C3).",
                )
            elif "reshard_after_forward" not in kwargs and not unknown_options:
                self._emit(
                    node,
                    "fsdp2_implicit_reshard",
                    "info",
                    "FSDP2 call omits reshard_after_forward; verify the installed version and root or child unit behavior (C3).",
                )

        # 4. Explicit torch.distributed collectives
        elif namespace in {"torch.distributed", "torch.distributed.distributed_c10d"} and short in _COLLECTIVE_ASYNC_POSITION:
            if self._stream_depth > 0:
                self._emit(
                    node,
                    "collective_in_custom_stream",
                    "high",
                    "Collective %s invoked inside a custom CUDA/HIP stream; on RCCL verify canonical per-group ordering (R1) and cross-stream buffer lifetime." % dotted,
                )
            elif self._loop_depth > 0 and _known_sync_call(node, kwargs, unknown_options, _COLLECTIVE_ASYNC_POSITION[short]):
                self._emit(
                    node,
                    "collective_in_loop_sync",
                    "medium",
                    "Collective %s inside a loop with default or explicit async_op=False; inspect the loop and T5 before coalescing." % dotted,
                )
            else:
                self._emit(
                    node,
                    "collective_call",
                    "info",
                    "Explicit collective %s; inspect overlap and message size in trace-triage T5." % dotted,
                )

        # 5. Distributed object collectives (RCCL gather_object caution)
        elif namespace in {"torch.distributed", "torch.distributed.distributed_c10d"} and short in _OBJECT_COLLECTIVE_NAMES:
            self._emit(
                node,
                "dist_gather_object",
                "info",
                "Object collective %s; for DCP save-plan metadata on affected HIP/RCCL stacks, verify whether a dedicated Gloo group applies (R2)." % dotted,
            )

        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            receiving = _dotted_name(node.func.value).split(".")[0]
            if isinstance(self._binding(receiving), dict):
                self._scopes[-1][receiving] = None
        for arg in list(node.args) + [kw.value for kw in node.keywords if kw.arg is not None]:
            if isinstance(arg, ast.Name) and isinstance(self._binding(arg.id), dict):
                self._invalidate(arg)


_REGEX_FALLBACK_RULES = [
    (
        "ddp_find_unused_true",
        "info",
        re.compile(r"\bfind_unused_parameters\s*=\s*True\b"),
        "[regex-fallback] Found find_unused_parameters=True; verify the actual DDP call and static_graph before considering changes (C1).",
    ),
    (
        "fsdp1_limit_all_gathers_true",
        "info",
        re.compile(r"\blimit_all_gathers\s*=\s*True\b"),
        "[regex-fallback] FSDP1 limit_all_gathers=True protects memory; confirm exposed GPU communication and memory headroom before evaluating False (C2).",
    ),
    (
        "fsdp2_reshard_after_forward_true",
        "info",
        re.compile(r"\breshard_after_forward\s*=\s*True\b"),
        "[regex-fallback] FSDP2 reshard_after_forward=True re-gathers weights in backward; if memory headroom permits, evaluate False (C3).",
    ),
]


def _scan_python_regex_fallback(source: str, path: str) -> List[Finding]:
    findings: List[Finding] = []
    original = source.split("\n")
    masked = [list(line) for line in original]
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type not in (tokenize.STRING, tokenize.COMMENT):
                continue
            for number in range(token.start[0], token.end[0] + 1):
                chars = masked[number - 1]
                start = token.start[1] if number == token.start[0] else 0
                stop = token.end[1] if number == token.end[0] else len(chars)
                chars[start:stop] = " " * (stop - start)
    except (tokenize.TokenError, SyntaxError):
        return []
    for idx, chars in enumerate(masked, 1):
        line = "".join(chars)
        for pattern, severity, regex, note in _REGEX_FALLBACK_RULES:
            match = regex.search(line)
            if match:
                findings.append(Finding(path, idx, match.start(), pattern, severity, original[idx - 1].strip(), note))
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


_JSON_WHITESPACE = re.compile(r"\s*")


def _json_object_offsets(raw, start):
    """Locate direct keys in an object already validated by json.loads."""
    decode = json.JSONDecoder().raw_decode
    position = _JSON_WHITESPACE.match(raw, start + 1).end()
    while raw[position] != "}":
        key, end_key = decode(raw, position)
        colon = _JSON_WHITESPACE.match(raw, end_key).end()
        value = _JSON_WHITESPACE.match(raw, colon + 1).end()
        yield key, (position, value)
        _, end_value = decode(raw, value)
        position = _JSON_WHITESPACE.match(raw, end_value).end()
        if raw[position] == ",":
            position = _JSON_WHITESPACE.match(raw, position + 1).end()


def _scan_zero_options(path, fields, locate):
    stage = fields.get("stage")
    if type(stage) is not int or stage not in (1, 2):
        return []

    findings = []
    for key in ("overlap_comm", "contiguous_gradients"):
        if fields.get(key) is False:
            line, snippet = locate(key)
            scope = " (C4)" if stage == 2 else ""
            findings.append(Finding(
                path, line, 0, "ds_" + key + "_disabled", "high", snippet,
                f"DeepSpeed ZeRO-{stage} config sets {key}=false; check the effective config and memory budget{scope}.",
            ))
    if stage == 2 and fields.get("overlap_comm") is True:
        line, snippet = locate("overlap_comm")
        findings.append(Finding(
            path, line, 0, "ds_zero2_config", "info", snippet,
            "DeepSpeed ZeRO-2 with overlap_comm=true; if T5 still shows low backward overlap, verify the effective runtime path and DeepSpeed version (C6).",
        ))
    return findings


def scan_json_config(path: str) -> List[Finding]:
    try:
        if os.path.getsize(path) > _MAX_CONFIG_BYTES:
            return []
        with open(path, "r", encoding="utf-8-sig") as handle:
            raw = handle.read()
        data = json.loads(raw)
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict) or not isinstance(data.get("zero_optimization"), dict):
        return []

    root = _JSON_WHITESPACE.match(raw).end()
    _, zero_value = dict(_json_object_offsets(raw, root))["zero_optimization"]
    offsets = dict(_json_object_offsets(raw, zero_value))
    lines = raw.split("\n")

    def locate(key):
        line = raw.count("\n", 0, offsets[key][0]) + 1
        return line, lines[line - 1].strip()

    return _scan_zero_options(path, data["zero_optimization"], locate)


_ZERO_YAML_KEYS = {"stage", "overlap_comm", "contiguous_gradients"}
_YAML_FIELD = re.compile(
    r"""^(?:"([a-zA-Z_][\w-]*)"|'([a-zA-Z_][\w-]*)'|([a-zA-Z_][\w-]*|<<))\s*:(.*)$"""
)
_YAML_COMMENT = re.compile(r""" "(?:\\.|[^"\\])*" | '(?:''|[^'])*' | ((?<!\S)\#.*) """, re.X)
_YAML_PLAIN_SCALAR = re.compile(r"[\w.+-]+$")
_YAML_QUOTED_SCALAR = re.compile(r"""(?:"(?:\\.|[^"\\])*"|'(?:''|[^'])*')$""")
_YAML_FLOW_TOKEN = re.compile(r""" "(?:\\.|[^"\\])*" | '(?:''|[^'])*' | ["'{}\[\]] """, re.X)
_YAML_BOOLEANS = {
    "true": True, "True": True, "TRUE": True,
    "false": False, "False": False, "FALSE": False,
}


def _yaml_field(line, *, flow=False):
    match = _YAML_FIELD.fullmatch(line)
    if match:
        key = next(key for key in match.groups()[:3] if key is not None)
        value = match[4]
        if not value or value[0].isspace() or (flow and match[3] is None):
            return key, value.strip()
    return None, None


def _yaml_lines(raw):
    for number, original in enumerate(raw.splitlines(), 1):
        line = original
        for match in _YAML_COMMENT.finditer(line):
            if match[1]:
                line = line[:match.start()]
                break
        stripped = line.strip()
        if stripped:
            yield number, len(line) - len(line.lstrip()), stripped


def _yaml_complete_root_value(value):
    if not value:
        return True
    if value[0] in "\"'":
        return _YAML_QUOTED_SCALAR.fullmatch(value) is not None
    if value[0] in "!&*":
        return False
    if value[0] not in "{[":
        return True

    closing = {"{": "}", "[": "]"}
    expected = []
    for token in _YAML_FLOW_TOKEN.finditer(value):
        text = token[0]
        if len(text) > 1:  # complete quoted scalar
            continue
        if text in closing:
            expected.append(closing[text])
        elif not expected or expected.pop() != text:
            return False
        elif not expected:
            return token.end() == len(value)
    return False


def _yaml_zero_options(raw):
    """Read a single document and direct scalars under the root ZeRO mapping."""
    lines = list(_yaml_lines(raw))
    if not lines:
        return {}, {}
    if lines[0][1:] == (0, "---"):
        lines.pop(0)
    if lines and lines[-1][1:] == (0, "..."):
        lines.pop()
    if not lines:
        return {}, {}

    base_indent = min(indent for _, indent, _ in lines)
    root_fields = [
        (idx,) + _yaml_field(line)
        for idx, (_, indent, line) in enumerate(lines)
        if indent == base_indent
    ]
    sections = [(idx, value) for idx, key, value in root_fields if key == "zero_optimization"]
    if not sections:
        return {}, {}
    if len(sections) != 1 or any(key is None for _, key, _ in root_fields):
        raise ValueError("expected one document and one root zero_optimization key")
    if any(not _yaml_complete_root_value(value) for _, _, value in root_fields):
        raise ValueError("multiline or annotated root value")
    if any("\t" in line[:len(line) - len(line.lstrip())] for line in raw.splitlines()):
        raise ValueError("tab indentation")
    index, value = sections[0]
    number = lines[index][0]
    flow = value.startswith("{") and value.endswith("}")
    if flow:
        # A simple flow mapping. Nested flow values and quoted commas are outside this subset.
        entries = [(number, item.strip()) for item in value[1:-1].split(",") if item.strip()]
        for _, line in entries:
            key, scalar = _yaml_field(line, flow=True)
            if key is None or not (_YAML_PLAIN_SCALAR.fullmatch(scalar) or _YAML_QUOTED_SCALAR.fullmatch(scalar)):
                raise ValueError("complex inline zero_optimization mapping")
    elif value == "":
        children = []
        for row in lines[index + 1:]:
            if row[1] <= base_indent:
                break
            children.append(row)
        child_indent = children[0][1] if children else base_indent
        if any(row[1] < child_indent for row in children):
            raise ValueError("inconsistent zero_optimization indentation")
        entries = [(line_no, line) for line_no, indent, line in children if indent == child_indent]
        for pos, (_, indent, line) in enumerate(children[:-1]):
            if indent == child_indent and _yaml_field(line)[0] in _ZERO_YAML_KEYS:
                if children[pos + 1][1] > child_indent:
                    raise ValueError("multiline ZeRO option value")
    else:
        raise ValueError("zero_optimization is not a simple mapping")

    fields = {}
    locations = {}
    for line_no, line in entries:
        key, scalar = _yaml_field(line, flow=flow)
        if key is None or key == "<<":
            raise ValueError("complex YAML key or merge in zero_optimization")
        if key not in _ZERO_YAML_KEYS:
            continue
        if key in fields:
            raise ValueError(f"duplicate {key}")
        if not _YAML_PLAIN_SCALAR.fullmatch(scalar):
            raise ValueError(f"expected a plain scalar for {key}")
        if key == "stage":
            fields[key] = int(scalar) if scalar in {"1", "2", "3"} else None
        else:
            fields[key] = _YAML_BOOLEANS.get(scalar)
        locations[key] = line_no
    return fields, locations


def scan_yaml_config(path: str) -> List[Finding]:
    try:
        if os.path.getsize(path) > _MAX_CONFIG_BYTES:
            return []
        with open(path, "r", encoding="utf-8-sig") as handle:
            raw = handle.read()
    except (OSError, UnicodeError):
        return []
    try:
        fields, locations = _yaml_zero_options(raw)
    except ValueError as exc:
        print(f"warning: unsupported YAML in {path}: {exc}; verify the effective config manually", file=sys.stderr)
        return []
    lines = raw.splitlines()
    return _scan_zero_options(path, fields, lambda key: (locations[key], lines[locations[key] - 1].strip()))


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
        help="Skip scanning JSON/YAML DeepSpeed config files.",
    )
    args = parser.parse_args(argv)
    paths = [path for path in args.paths if not os.path.isfile(path) and not os.path.isdir(path)]
    if paths:
        parser.error("input path is not an existing file or directory: " + ", ".join(paths))

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
        "trace-triage (T5) first (see references/triage-and-rccl.md). "
        "A candidate also does not establish backend safety, numerical equivalence, or speedup.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
