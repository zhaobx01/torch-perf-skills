#!/usr/bin/env python3
"""Static scanner for torch.compile boundaries, existing entries, and risk cues.

It inventories existing torch.compile/disable sites, checkpoint & DDP/FSDP
wrapping cues, global dynamo/inductor config mutations, private compiler APIs,
and repeated nn.ModuleList candidates. It does NOT prove a region is hot, will
capture without graph breaks, or will speed up execution. Confirm with
trace-triage and TORCH_LOGS=graph_breaks,recompiles first.

Detected patterns (all name/AST-based, so false positives are expected):
  existing_compile_entry   torch.compile(...) / module.compile(...) (excludes re.compile)
  compile_disable_boundary torch.compiler.disable / torch._dynamo.disable eager boundaries
  checkpoint_wrap_cue      checkpoint_wrapper / ptd_checkpoint_wrapper / apply_activation_checkpointing /
                           torch.utils.checkpoint.* / gradient_checkpointing_enable / set_gradient_checkpointing
  distributed_wrap_cue     FullyShardedDataParallel / fully_shard / DistributedDataParallel / MMDistributedDataParallel
  global_compiler_config   Direct assignment to *_dynamo.config.* / *_inductor.config.* or suppress_errors=True (excludes config.patch)
  private_inductor_api     torch._inductor.inductor_prims / triton_heuristics / skip_code
  repeated_module_list     Per-file summary of nn.ModuleList sites (repeated-layer compile candidates)

Usage:
    python3 scan_compile_candidates.py <file-or-dir> [...]
    python3 scan_compile_candidates.py --json <file-or-dir> [...]
    python3 scan_compile_candidates.py --patterns existing_compile_entry,checkpoint_wrap_cue <file-or-dir>

Only the Python standard library is used (Python 3.6+ compatible).
Use a Python interpreter that supports the scanned source's syntax. JSON output
includes ``disclaimer``, ``files_scanned``, ``files_skipped``, ``skipped_files``,
and ``findings``. Exit 1 indicates a partial or empty scan, and exit 2 indicates
invalid CLI arguments.
"""

import argparse
import ast
import json
import os
import sys
import tokenize
from typing import Dict, Iterable, List, Optional, Set, Tuple

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
_DISCLAIMER = (
    "STATIC candidates only: they do not prove a region is hot, compiles cleanly, or speeds up "
    "execution. Confirm with trace-triage and TORCH_LOGS first."
)

_NOTES = {
    "existing_compile_entry": (
        "Existing torch.compile / module.compile entry; verify switch lifecycle, nested-compile "
        "interactions, actual runtime capture, and clean off-path rollback (P5, P6, P7)."
    ),
    "compile_disable_boundary": (
        "Explicit Dynamo/compiler disable boundary; confirm scope (including global registries "
        "if applicable), fullgraph interaction, and whether surrounding ops still compile (P4)."
    ),
    "checkpoint_wrap_cue": (
        "Activation checkpointing site; verify wrapping order relative to torch.compile and "
        "FSDP/DDP, closure caching, and backward gradient equivalence (P1, P2, P5, P7)."
    ),
    "distributed_wrap_cue": (
        "DDP/FSDP wrapper site; check wrapping order with torch.compile, static_graph vs padding "
        "contracts, dynamic class lookup, and ._orig_mod. checkpoint key compatibility (P3, P5)."
    ),
    "global_compiler_config": (
        "Global Dynamo/Inductor config mutation or suppress_errors; prefer scoped "
        "torch._dynamo.config.patch(...), verify version support, and never mask failures with suppress_errors (P4, P5)."
    ),
    "private_inductor_api": (
        "Private Inductor/Dynamo internal API; guard with explicit availability/version checks "
        "and fallback paths, and validate compile-time vs steady-state tradeoff (P5, P7)."
    ),
    "repeated_module_list": (
        "nn.ModuleList repeated layers found; candidate for per-layer vs whole-subsystem "
        "boundary comparison and repeated-layer graph reuse checks (P1, P7)."
    ),
}

_CHECKPOINT_TAILS = {
    "checkpoint_wrapper",
    "ptd_checkpoint_wrapper",
    "apply_activation_checkpointing",
    "gradient_checkpointing_enable",
    "enable_gradient_checkpointing",
    "set_gradient_checkpointing",
    "set_grad_checkpointing",
}

_DISTRIBUTED_MODULES = (
    "torch.nn.parallel",
    "torch.distributed.fsdp",
    "torch.distributed._composable.fsdp",
    "mmcv.parallel",
)

_DISTRIBUTED_TAILS = {
    "FullyShardedDataParallel",
    "fully_shard",
    "DistributedDataParallel",
    "MMDistributedDataParallel",
}

_NON_TORCH_COMPILE_RECEIVERS = {"re", "regex", "pattern", "prog", "ast"}


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


def _bool_const(node):
    # type: (Optional[ast.AST]) -> Optional[bool]
    if node is None:
        return None
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant) and isinstance(node.value, bool):
        return node.value
    return None


def _kw_map(node):
    # type: (ast.Call) -> Dict[str, ast.AST]
    return {kw.arg: kw.value for kw in node.keywords if kw.arg}


def _is_torch_distributed_name(resolved, raw, is_imported):
    # type: (str, str, Optional[bool]) -> bool
    tail = resolved.rsplit(".", 1)[-1] if resolved else ""
    if tail not in _DISTRIBUTED_TAILS:
        return False
    if is_imported is True:
        return any(
            resolved.startswith(prefix + ".") for prefix in _DISTRIBUTED_MODULES
        )
    if is_imported is False:
        return False
    return raw == tail or any(raw.startswith(prefix + ".") for prefix in _DISTRIBUTED_MODULES)


def _is_torch_module_list(resolved, raw, is_imported):
    # type: (str, str, Optional[bool]) -> bool
    if is_imported is True:
        return resolved in ("torch.nn.ModuleList", "torch.nn.modules.container.ModuleList")
    if is_imported is False:
        return False
    return raw in ("ModuleList", "nn.ModuleList", "torch.nn.ModuleList")


def _is_torch_compile_target(resolved, is_imported):
    # type: (str, Optional[bool]) -> bool
    if is_imported is True:
        return resolved == "torch.compile"
    if is_imported is False:
        return False
    return resolved == "torch.compile"


def _is_disable_target(resolved, is_imported):
    # type: (str, Optional[bool]) -> bool
    if is_imported is False:
        return False
    if is_imported is True:
        return resolved in ("torch.compiler.disable", "torch._dynamo.disable")
    return resolved.endswith("compiler.disable") or resolved.endswith("_dynamo.disable")


class _Visitor(ast.NodeVisitor):
    def __init__(self, path, lines):
        # type: (str, List[str]) -> None
        self.path = path
        self.lines = lines
        self.findings = []  # type: List[Finding]
        self.module_list_lines = []  # type: List[int]
        self._emitted_keys = set()  # type: Set[Tuple[int, str]]
        self._scopes = [{}]  # type: List[Dict[str, Optional[str]]]
        self._dist_import_lines = []  # type: List[int]
        self._has_dist_call = False

    def _emit(self, line, pattern, snippet=None):
        # type: (int, str, Optional[str]) -> None
        key = (line, pattern)
        if key in self._emitted_keys:
            return
        self._emitted_keys.add(key)
        if snippet is None:
            snippet = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        self.findings.append(Finding(self.path, line, pattern, snippet, _NOTES[pattern]))

    def _resolve(self, node):
        # type: (ast.AST) -> Tuple[str, Optional[bool]]
        raw = _dotted(node)
        if not raw:
            return "", None
        head, _, tail = raw.partition(".")
        for scope in reversed(self._scopes):
            if head in scope:
                target = scope[head]
                if isinstance(target, str):
                    return ("{}.{}".format(target, tail) if tail else target), True
                return raw, False
        return raw, None

    def _bind_local(self, target):
        # type: (ast.AST) -> None
        if isinstance(target, ast.Name):
            self._scopes[-1][target.id] = None
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_local(elt)
        elif isinstance(target, ast.Starred):
            self._bind_local(target.value)

    def _check_decorator(self, dec):
        # type: (ast.AST) -> None
        target = dec.func if isinstance(dec, ast.Call) else dec
        resolved, is_imported = self._resolve(target)
        if _is_torch_compile_target(resolved, is_imported):
            self._emit(getattr(dec, "lineno", 0), "existing_compile_entry")
        elif _is_disable_target(resolved, is_imported):
            self._emit(getattr(dec, "lineno", 0), "compile_disable_boundary")

    def visit_FunctionDef(self, node):
        # type: (ast.FunctionDef) -> None
        self._scopes[-1][node.name] = None
        for dec in node.decorator_list:
            self._check_decorator(dec)
            self.visit(dec)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        self._scopes.append({})
        for arg in (
            list(getattr(node.args, "posonlyargs", []))
            + node.args.args
            + node.args.kwonlyargs
            + ([node.args.vararg] if node.args.vararg else [])
            + ([node.args.kwarg] if node.args.kwarg else [])
        ):
            self._scopes[-1][arg.arg] = None
        for stmt in node.body:
            self.visit(stmt)
        self._scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        # type: (ast.ClassDef) -> None
        self._scopes[-1][node.name] = None
        for dec in node.decorator_list:
            self._check_decorator(dec)
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        for kw in node.keywords:
            self.visit(kw)
        self._scopes.append({})
        for stmt in node.body:
            self.visit(stmt)
        self._scopes.pop()

    def visit_Import(self, node):
        # type: (ast.Import) -> None
        for alias in node.names:
            head = alias.name.split(".")[0]
            local = alias.asname or head
            self._scopes[-1][local] = alias.name if alias.asname else head
            if "inductor_prims" in alias.name or "triton_heuristics" in alias.name:
                self._emit(node.lineno, "private_inductor_api")

    def visit_ImportFrom(self, node):
        # type: (ast.ImportFrom) -> None
        mod = node.module or ""
        if "inductor_prims" in mod or "triton_heuristics" in mod:
            self._emit(node.lineno, "private_inductor_api")
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            if node.level == 0 and mod:
                full = "{}.{}".format(mod, alias.name)
                self._scopes[-1][local] = full
            else:
                full = ""
                self._scopes[-1][local] = None
            if alias.name in ("force_stride_order", "skip_code", "triton_heuristics", "inductor_prims"):
                if not full or full.startswith("torch."):
                    self._emit(node.lineno, "private_inductor_api")
            if alias.name in _DISTRIBUTED_TAILS and any(
                full.startswith(prefix + ".") for prefix in _DISTRIBUTED_MODULES
            ):
                self._dist_import_lines.append(node.lineno)

    def visit_Assign(self, node):
        # type: (ast.Assign) -> None
        for target in node.targets:
            tname, _ = self._resolve(target)
            if (
                "_dynamo.config." in tname
                or "_inductor.config." in tname
                or tname.endswith(".suppress_errors")
            ):
                self._emit(node.lineno, "global_compiler_config")
        self.visit(node.value)
        for target in node.targets:
            self._bind_local(target)
            self.visit(target)

    def visit_AnnAssign(self, node):
        # type: (ast.AnnAssign) -> None
        if node.value is not None:
            self.visit(node.value)
        self._bind_local(node.target)
        self.visit(node.target)

    def visit_Call(self, node):
        # type: (ast.Call) -> None
        raw = _dotted(node.func)
        resolved, is_imported = self._resolve(node.func)
        tail = resolved.rsplit(".", 1)[-1] if resolved else ""
        kw = _kw_map(node)

        if _is_torch_compile_target(resolved, is_imported):
            self._emit(node.lineno, "existing_compile_entry")
        elif tail == "compile" and isinstance(node.func, ast.Attribute):
            recv = _dotted(node.func.value)
            recv_tail = recv.rsplit(".", 1)[-1].lower()
            if recv_tail not in _NON_TORCH_COMPILE_RECEIVERS:
                if ({"mode", "backend", "dynamic", "fullgraph", "options"} & set(kw.keys())) or (
                    not node.args and recv_tail in ("self", "model", "net", "module", "backbone", "encoder", "decoder", "action")
                ):
                    self._emit(node.lineno, "existing_compile_entry")
        elif _is_disable_target(resolved, is_imported):
            self._emit(node.lineno, "compile_disable_boundary")
        elif (
            tail in _CHECKPOINT_TAILS
            or resolved.endswith("checkpoint.checkpoint")
            or resolved.endswith("checkpoint.checkpoint_sequential")
        ):
            self._emit(node.lineno, "checkpoint_wrap_cue")
        elif _is_torch_distributed_name(resolved, raw, is_imported):
            self._has_dist_call = True
            self._emit(node.lineno, "distributed_wrap_cue")
        elif _is_torch_module_list(resolved, raw, is_imported):
            self.module_list_lines.append(node.lineno)
        elif tail in ("force_stride_order", "skip_code") or "triton_heuristics" in resolved:
            if is_imported is not True or resolved.startswith("torch."):
                self._emit(node.lineno, "private_inductor_api")

        if _bool_const(kw.get("suppress_errors")) is True:
            self._emit(node.lineno, "global_compiler_config")

        self.generic_visit(node)


def scan_file(path, allowed_patterns=None):
    # type: (str, Optional[Set[str]]) -> Tuple[Optional[List[Finding]], Optional[str]]
    try:
        with tokenize.open(path) as fh:
            source = fh.read()
        tree = ast.parse(source, filename=path)
    except (OSError, SyntaxError, ValueError) as exc:
        reason = str(exc)
        print("warning: skipped {}: {}".format(path, reason), file=sys.stderr)
        return None, reason
    lines = source.splitlines()
    visitor = _Visitor(path, lines)
    visitor.visit(tree)

    findings = list(visitor.findings)
    if not visitor._has_dist_call and visitor._dist_import_lines:
        first_dist = visitor._dist_import_lines[0]
        snippet_dist = lines[first_dist - 1].strip() if 0 < first_dist <= len(lines) else ""
        findings.append(Finding(path, first_dist, "distributed_wrap_cue", snippet_dist, _NOTES["distributed_wrap_cue"]))
    if visitor.module_list_lines:
        first = visitor.module_list_lines[0]
        snippet = "{} nn.ModuleList site(s); first: {}".format(
            len(visitor.module_list_lines),
            lines[first - 1].strip() if 0 < first <= len(lines) else "",
        )
        findings.append(Finding(path, first, "repeated_module_list", snippet, _NOTES["repeated_module_list"]))

    if allowed_patterns is not None:
        findings = [f for f in findings if f.pattern in allowed_patterns]
    return findings, None


def _iter_files(paths, on_error):
    # type: (Iterable[str], object) -> Iterable[str]
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
    skipped_files = []  # type: List[Dict[str, str]]

    def walk_error(error):
        path_name = getattr(error, "filename", None) or "<walk>"
        reason = str(error)
        skipped_files.append({"file": str(path_name), "reason": reason})
        print("warning: skipped {}".format(reason), file=sys.stderr)

    for path in _iter_files(args.paths, walk_error):
        result, reason = scan_file(path, allowed_patterns=allowed)
        if result is None:
            skipped_files.append({"file": path, "reason": reason or "unknown error"})
        else:
            scanned += 1
            findings.extend(result)
    findings.sort(key=lambda f: (f.file, f.line, f.pattern))
    skipped = len(skipped_files)

    if skipped or not scanned:
        print("Python files: {} scanned, {} skipped.".format(scanned, skipped), file=sys.stderr)
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
        print("No Python files scanned.")
    elif not findings:
        print("No candidates found in scanned files.")
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
    print(_DISCLAIMER, file=sys.stderr)
    return 1 if skipped or not scanned else 0


if __name__ == "__main__":
    raise SystemExit(main())
