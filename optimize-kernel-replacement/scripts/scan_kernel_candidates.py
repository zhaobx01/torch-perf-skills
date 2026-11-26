#!/usr/bin/env python3
"""Static scanner for kernel-replacement CANDIDATES in PyTorch code.

It lists places where a faster, fused, or better-tuned implementation might be
worth evaluating. It does NOT prove the code is slow, that a replacement is
available on your platform, or that a replacement would be numerically
equivalent. Every finding is a candidate to confirm with a profiler trace.

Detected patterns (all name/AST-based, so false positives are expected):
  adamw_not_fused          torch.optim.AdamW(...) / AdamW(...) without fused=/foreach=
  non_adamw_optimizer      Muon / DistributedMuon / batched_newton_schulz (batched/multi-tensor candidates)
  flex_attention_call      flex_attention / create_block_mask / create_mask (tile, H=1 mask, TF32+cache_key_tag)
  fa2_varlen_call          known FA2 varlen APIs, including the masked vendor API
  sdpa_call                F.scaled_dot_product_attention(...) (check backend choice & backward finiteness)
  mha_default_avg_weights  MHA constructor as a call-site clue, or functional MHA without weights opt-outs
  manual_attention         softmax(q @ k.T ...) style attention written by hand
  complex_rope             view_as_complex / torch.polar / view_as_real of a rotary-like product
  layernorm_module         nn.LayerNorm / nn.RMSNorm construction (vendor kernel / compile candidates)
  add_then_norm            a norm call whose input is an addition expression: norm(x + residual)
  upsample_bilinear        nn.Upsample / F.interpolate with mode="bilinear" (check Inductor gather lowering)
  custom_autograd          torch.autograd.Function subclass (check torch.compile traceability)
  custom_op_missing_fake   custom_op without a statically matched register_fake in the same file

Usage:
    python3 scan_kernel_candidates.py <file-or-dir> [...]
    python3 scan_kernel_candidates.py --json <file-or-dir> [...]
    python3 scan_kernel_candidates.py --patterns flex_attention_call,complex_rope <file-or-dir>

Only the Python standard library is used (Python 3.6+ compatible).
Use a Python interpreter that supports the scanned source's syntax. Dynamic
arguments and cross-file registrations require manual review. JSON output
includes `disclaimer`, `files_scanned`, `files_skipped`, `skipped_files`, and
`findings`. Exit 1 indicates a partial or empty scan, and exit 2 indicates
invalid CLI arguments.
"""

import argparse
import ast
import json
import os
import re
import sys
import tokenize
from typing import Dict, List, Optional, Set, Tuple

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
_FA2_VARLEN_CALLS = {
    "flash_attn_varlen_func", "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_with_mask_func",
}
_NORM_CALL_RE = re.compile(r"(?:(?:\w+_)?norm|layernorm|rmsnorm|layer_norm|rms_norm)\d*")
_DISCLAIMER = (
    "STATIC candidates only: they do not show the code is slow, that a replacement "
    "exists for your platform, or that it is numerically equivalent. Confirm with a profiler first."
)

_NOTES = {
    "adamw_not_fused": (
        "AdamW constructed without fused= or foreach=; evaluate fused=True or an Apex FusedAdam "
        "adapter (check generator materialization, rounding, and checkpoint portability)."
    ),
    "non_adamw_optimizer": (
        "Non-AdamW matrix optimizer (Muon / Newton-Schulz); evaluate multi-tensor Triton fusion "
        "(with min-element threshold), dynamic megabatch sizing, and packed layout conversion."
    ),
    "flex_attention_call": (
        "FlexAttention / BlockMask call; check H=1 head-shared BlockMask (avoid redundant dense "
        "create_mask + expand), forward/backward kernel_options tile tuning, and FP32 TF32 + cache_key_tag."
    ),
    "fa2_varlen_call": (
        "FlashAttention varlen call. Evaluate dense routing only if the input lengths and "
        "mask contract permit it. Guard private APIs with version or hash checks."
    ),
    "sdpa_call": (
        "scaled_dot_product_attention call; confirm which backend is selected on the target "
        "platform and that backward gradients are finite (avoid mistaking MATH fallback for speedup)."
    ),
    "mha_default_avg_weights": (
        "Functional MHA without an explicit weights opt-out; check whether callers consume "
        "averaged weights and whether FP32 QK matmul is hot."
    ),
    "manual_attention": (
        "Attention written by hand (softmax of a matmul); a fused attention backend may apply "
        "if masks and dtype allow."
    ),
    "complex_rope": (
        "Complex/polar RoPE (view_as_complex/view_as_real/polar); evaluate real-valued cos/sin "
        "formulation for torch.compile fusion while preserving original precision (e.g., float64)."
    ),
    "layernorm_module": (
        "Norm module construction; a vendor fused kernel or compiled tensor helper may apply "
        "for the target platform. Confirm a fallback path before replacing."
    ),
    "add_then_norm": (
        "Norm call takes an addition expression norm(x + residual); check whether residual output "
        "is reused and whether a fused add+norm kernel applies on the target backend."
    ),
    "upsample_bilinear": (
        "Bilinear Upsample/interpolate; under torch.compile check whether Inductor lowers it to "
        "a slow gather kernel (evaluate @torch.compiler.disable -> ATen or a strictly guarded native op)."
    ),
    "custom_autograd": (
        "Custom autograd.Function; check torch.compile traceability, whether torch.library.custom_op "
        "registration is needed, and whether native extensions have Python-level shape/dtype guards."
    ),
    "custom_op_missing_fake": (
        "No statically matched register_fake in this file. Check for dynamic or cross-file "
        "registration and verify fake output metadata if this op is used under torch.compile."
    ),
}


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
    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return node.s
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _bool_const(node):
    # type: (Optional[ast.AST]) -> Optional[bool]
    if node is None:
        return None
    if hasattr(ast, "NameConstant") and isinstance(node, ast.NameConstant) and isinstance(node.value, bool):
        return node.value
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _kw_map(node):
    # type: (ast.Call) -> Dict[str, ast.AST]
    result = {}
    for kw in node.keywords:
        if kw.arg is not None:
            result[kw.arg] = kw.value
        elif isinstance(kw.value, ast.Dict):
            for key_node, value in zip(kw.value.keys, kw.value.values):
                key = _str_const(key_node)
                if key is not None:
                    result[key] = value
    return result


def _call_argument(node, keywords, name, position):
    # type: (ast.Call, Dict[str, ast.AST], str, int) -> Optional[ast.AST]
    if name in keywords:
        return keywords[name]
    if len(node.args) > position and not any(
        isinstance(arg, ast.Starred) for arg in node.args[:position + 1]
    ):
        return node.args[position]
    return None


def _library_call(resolved, symbol):
    # type: (str, str) -> bool
    return resolved in (symbol, "library." + symbol, "torch.library." + symbol)


def _is_torch_nn_symbol(resolved, raw, is_imported, symbol):
    # type: (str, str, Optional[bool], str) -> bool
    if is_imported is True:
        return resolved in (
            "torch.nn." + symbol,
            "torch.nn.modules.normalization." + symbol,
            "torch.nn.modules.activation." + symbol,
        )
    if is_imported is False:
        return False
    return raw in (symbol, "nn." + symbol, "torch.nn." + symbol)


def _is_torch_adamw(resolved, raw, is_imported):
    # type: (str, str, Optional[bool]) -> bool
    if is_imported is True:
        return resolved in ("torch.optim.AdamW", "torch.optim.adamw.AdamW")
    if is_imported is False:
        return False
    return raw in ("AdamW", "optim.AdamW", "torch.optim.AdamW")


def _looks_like_rotary_product(node):
    return (
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
        and any(
            token in _dotted(part).lower()
            for part in ast.walk(node) if isinstance(part, (ast.Name, ast.Attribute))
            for token in ("freq", "rope", "rotary")
        )
    )


class _Visitor(ast.NodeVisitor):
    def __init__(self, path, lines):
        # type: (str, List[str]) -> None
        self.path = path
        self.lines = lines
        self.findings = []  # type: List[Finding]
        # Track custom_op declarations: list of (lineno, scope_id, fn_name, op_name)
        self.custom_ops = []  # type: List[Tuple[int, int, Optional[str], Optional[str]]]
        self.fake_fn_keys = set()  # type: Set[Tuple[int, str]]
        self.fake_op_names = set()  # type: Set[str]
        self.custom_op_bindings = {}  # type: Dict[int, Tuple[int, str]]
        self._scopes = [{}]  # type: List[Dict[str, Optional[str]]]
        self._scope_ids = [0]  # type: List[int]
        self._next_scope_id = 1

    def _emit(self, node, pattern, note=None):
        # type: (ast.AST, str, Optional[str]) -> None
        line = getattr(node, "lineno", 0)
        snippet = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        self.findings.append(Finding(self.path, line, pattern, snippet, note or _NOTES[pattern]))

    def _push_scope(self):
        # type: () -> None
        self._scopes.append({})
        self._scope_ids.append(self._next_scope_id)
        self._next_scope_id += 1

    def _pop_scope(self):
        # type: () -> None
        self._scopes.pop()
        self._scope_ids.pop()

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

    def _binding_scope(self, name):
        # type: (str) -> int
        for scope_id, scope in zip(reversed(self._scope_ids), reversed(self._scopes)):
            if name in scope:
                return scope_id
        return self._scope_ids[-1]

    def _bind_local(self, target):
        # type: (ast.AST) -> None
        if isinstance(target, ast.Name):
            self._scopes[-1][target.id] = None
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_local(elt)
        elif isinstance(target, ast.Starred):
            self._bind_local(target.value)

    def visit_Import(self, node):
        # type: (ast.Import) -> None
        for alias in node.names:
            head = alias.name.split(".")[0]
            local = alias.asname or head
            self._scopes[-1][local] = alias.name if alias.asname else head

    def visit_ImportFrom(self, node):
        # type: (ast.ImportFrom) -> None
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            if node.level == 0 and node.module:
                self._scopes[-1][local] = "{}.{}".format(node.module, alias.name)
            else:
                self._scopes[-1][local] = None

    def visit_FunctionDef(self, node):
        # type: (ast.FunctionDef) -> None
        enclosing_scope_id = self._scope_ids[-1]
        self._scopes[-1][node.name] = None
        if node.name == "batched_newton_schulz":
            self._emit(node, "non_adamw_optimizer")
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                dec_resolved, _ = self._resolve(dec.func)
                if _library_call(dec_resolved, "custom_op"):
                    self.custom_op_bindings[id(dec)] = (enclosing_scope_id, node.name)
            elif isinstance(dec, ast.Attribute) and dec.attr == "register_fake":
                if isinstance(dec.value, ast.Name):
                    self.fake_fn_keys.add((self._binding_scope(dec.value.id), dec.value.id))
            self.visit(dec)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        self._push_scope()
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
        self._pop_scope()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        # type: (ast.Assign) -> None
        if len(node.targets) == 1:
            self._bind_custom_op(node.targets[0], node.value)
        self.visit(node.value)
        for target in node.targets:
            self._bind_local(target)
            self.visit(target)

    def visit_AnnAssign(self, node):
        # type: (ast.AnnAssign) -> None
        if node.value is not None:
            self._bind_custom_op(node.target, node.value)
            self.visit(node.value)
        self._bind_local(node.target)
        self.visit(node.target)

    def _bind_custom_op(self, target, value):
        if isinstance(target, ast.Name) and isinstance(value, ast.Call):
            resolved, _ = self._resolve(value.func)
            if _library_call(resolved, "custom_op"):
                self.custom_op_bindings[id(value)] = (self._scope_ids[-1], target.id)

    def visit_ClassDef(self, node):
        # type: (ast.ClassDef) -> None
        self._scopes[-1][node.name] = None
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            resolved, _ = self._resolve(base)
            if resolved.endswith("autograd.Function") or resolved == "Function":
                self._emit(node, "custom_autograd")
            self.visit(base)
        for kw in node.keywords:
            self.visit(kw)
        if node.name in ("Muon", "DistributedMuon"):
            self._emit(node, "non_adamw_optimizer")
        self._push_scope()
        for stmt in node.body:
            self.visit(stmt)
        self._pop_scope()

    def visit_Call(self, node):
        # type: (ast.Call) -> None
        raw = _dotted(node.func)
        resolved, is_imported = self._resolve(node.func)
        tail = resolved.rsplit(".", 1)[-1] if resolved else ""
        kw = _kw_map(node)
        keywords = set(kw.keys())

        matched_constructor = False
        if _is_torch_adamw(resolved, raw, is_imported) and not ({"fused", "foreach"} & keywords):
            self._emit(node, "adamw_not_fused")
        elif tail in ("Muon", "DistributedMuon", "batched_newton_schulz"):
            self._emit(node, "non_adamw_optimizer")
        elif tail in ("flex_attention", "create_block_mask") or (
            tail == "create_mask"
            and resolved in ("create_mask", "flex_attention.create_mask", "torch.nn.attention.flex_attention.create_mask")
        ):
            self._emit(node, "flex_attention_call")
        elif tail in _FA2_VARLEN_CALLS:
            self._emit(node, "fa2_varlen_call")
        elif tail == "scaled_dot_product_attention":
            self._emit(node, "sdpa_call")
        elif _is_torch_nn_symbol(resolved, raw, is_imported, "MultiheadAttention"):
            matched_constructor = True
            self._emit(
                node, "mha_default_avg_weights",
                "MHA construction is a clue. Inspect each call site for need_weights, "
                "average_attn_weights and whether the weights are consumed.",
            )
        elif tail == "multi_head_attention_forward":
            avg_val = _bool_const(_call_argument(node, kw, "average_attn_weights", 23))
            need_val = _bool_const(_call_argument(node, kw, "need_weights", 15))
            if avg_val is not False and need_val is not False:
                self._emit(node, "mha_default_avg_weights")
        elif tail == "view_as_complex" or (
            tail == "polar" and resolved in ("polar", "torch.polar")
        ) or (
            tail == "view_as_real" and _looks_like_rotary_product(_call_argument(node, kw, "input", 0))
        ):
            self._emit(node, "complex_rope")
        elif (
            _is_torch_nn_symbol(resolved, raw, is_imported, "LayerNorm")
            or _is_torch_nn_symbol(resolved, raw, is_imported, "RMSNorm")
        ):
            matched_constructor = True
            self._emit(node, "layernorm_module")
        elif tail in ("Upsample", "interpolate"):
            mode_position = 2 if tail == "Upsample" else 3
            mode_val = _str_const(_call_argument(node, kw, "mode", mode_position))
            if mode_val == "bilinear":
                self._emit(node, "upsample_bilinear")
        elif _library_call(resolved, "custom_op"):
            op_name = _str_const(_call_argument(node, kw, "name", 0))
            binding = self.custom_op_bindings.get(id(node))
            if binding is not None:
                scope_id, fn_name = binding
            else:
                scope_id, fn_name = self._scope_ids[-1], None
            self.custom_ops.append((node.lineno, scope_id, fn_name, op_name))
        elif _library_call(resolved, "register_fake"):
            target = _call_argument(node, kw, "qualname", 0)
            op_name = _str_const(target)
            if op_name is not None:
                self.fake_op_names.add(op_name)
            elif isinstance(target, ast.Name):
                self.fake_fn_keys.add((self._binding_scope(target.id), target.id))
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "register_fake":
            if isinstance(node.func.value, ast.Name):
                target_name = node.func.value.id
                self.fake_fn_keys.add((self._binding_scope(target_name), target_name))

        if not matched_constructor and tail and _NORM_CALL_RE.fullmatch(tail.lower()):
            first_arg = _call_argument(node, kw, "input", 0)
            if isinstance(first_arg, ast.BinOp) and isinstance(first_arg.op, ast.Add):
                self._emit(node, "add_then_norm")

        self.generic_visit(node)


def _scan_manual_attention(tree, path, lines):
    # type: (ast.AST, str, List[str]) -> List[Finding]
    found = []  # type: List[Finding]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _dotted(node.func).rsplit(".", 1)[-1] == "softmax":
            if any(isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.MatMult) for sub in ast.walk(node)):
                line = node.lineno
                snippet = lines[line - 1].strip() if 0 < line <= len(lines) else ""
                found.append(Finding(path, line, "manual_attention", snippet, _NOTES["manual_attention"]))
    return found


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
    findings = visitor.findings + _scan_manual_attention(tree, path, lines)

    for line, scope_id, fn_name, op_name in visitor.custom_ops:
        matched = (
            (fn_name is not None and (scope_id, fn_name) in visitor.fake_fn_keys)
            or (op_name is not None and op_name in visitor.fake_op_names)
        )
        if not matched:
            snippet = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            findings.append(Finding(path, line, "custom_op_missing_fake", snippet, _NOTES["custom_op_missing_fake"]))

    if allowed_patterns is not None:
        findings = [f for f in findings if f.pattern in allowed_patterns]
    return findings, None


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
