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
  fa2_varlen_call          flash_attn_varlen*_func (check dense flash_attn_func fast path for uniform batches)
  sdpa_call                F.scaled_dot_product_attention(...) (check backend choice & backward finiteness)
  mha_default_avg_weights  nn.MultiheadAttention / multi_head_attention_forward without average_attn_weights=False
  manual_attention         softmax(q @ k.T ...) style attention written by hand
  complex_rope             view_as_complex / view_as_real / torch.polar (complex/polar RoPE)
  layernorm_module         nn.LayerNorm / nn.RMSNorm construction (vendor kernel / compile candidates)
  norm_then_add            a norm call whose result is added to another value in one expression
  upsample_bilinear        nn.Upsample / F.interpolate with mode="bilinear" (check Inductor gather lowering)
  custom_autograd          torch.autograd.Function subclass (check torch.compile traceability)
  custom_op_missing_fake   torch.library.custom_op without matching register_fake in the same file

Usage:
    python3 scan_kernel_candidates.py <file-or-dir> [...]
    python3 scan_kernel_candidates.py --json <file-or-dir> [...]
    python3 scan_kernel_candidates.py --patterns flex_attention_call,complex_rope <file-or-dir>

Only the Python standard library is used (Python 3.6+ compatible).
"""

import argparse
import ast
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}

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
        "FlashAttention varlen call; if uniform batches occur (q_lens/k_lens is None), evaluate "
        "routing to dense flash_attn_func. Guard any private FA2 internals with version/hash checks."
    ),
    "sdpa_call": (
        "scaled_dot_product_attention call; confirm which backend is selected on the target "
        "platform and that backward gradients are finite (avoid mistaking MATH fallback for speedup)."
    ),
    "mha_default_avg_weights": (
        "MultiheadAttention / multi_head_attention_forward without average_attn_weights=False; "
        "check if callers actually consume averaged weights and whether FP32 QK matmul is hot."
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
    "norm_then_add": (
        "A norm call is combined with an addition in one expression; a fused add+norm kernel "
        "is a candidate only if the profiler shows this region is hot."
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
        "torch.library.custom_op used without a matching register_fake in this file; "
        "torch.compile requires a fake implementation with matching shape, dtype, and device."
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
    if isinstance(node, ast.Str):
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
    return {kw.arg: kw.value for kw in node.keywords if kw.arg}


def _is_custom_op_decorator(dec):
    # type: (ast.AST) -> Tuple[bool, Optional[str]]
    if isinstance(dec, ast.Call):
        name = _dotted(dec.func)
        if name.endswith("library.custom_op") or name == "custom_op":
            op_name = _str_const(dec.args[0]) if dec.args else None
            if op_name is None:
                op_name = _str_const(_kw_map(dec).get("name"))
            return True, op_name
    return False, None


class _Visitor(ast.NodeVisitor):
    def __init__(self, path, lines):
        # type: (str, List[str]) -> None
        self.path = path
        self.lines = lines
        self.findings = []  # type: List[Finding]
        # Track custom_op declarations: list of (lineno, fn_name, op_name)
        self.custom_ops = []  # type: List[Tuple[int, Optional[str], Optional[str]]]
        self.fake_fn_names = set()  # type: Set[str]
        self.fake_op_names = set()  # type: Set[str]
        self.has_generic_register_fake = False
        self._in_custom_op_call_nodes = set()  # type: Set[int]

    def _emit(self, node, pattern):
        # type: (ast.AST, str) -> None
        line = getattr(node, "lineno", 0)
        snippet = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        self.findings.append(Finding(self.path, line, pattern, snippet, _NOTES[pattern]))

    def _record_register_fake_call(self, node):
        # type: (ast.Call) -> None
        func = node.func
        name = _dotted(func)
        if isinstance(func, ast.Attribute) and func.attr == "register_fake":
            if isinstance(func.value, ast.Name):
                self.fake_fn_names.add(func.value.id)
            else:
                op_name = _str_const(node.args[0]) if node.args else _str_const(_kw_map(node).get("qualname"))
                if op_name:
                    self.fake_op_names.add(op_name)
                else:
                    self.has_generic_register_fake = True
        elif name == "register_fake":
            op_name = _str_const(node.args[0]) if node.args else None
            if op_name:
                self.fake_op_names.add(op_name)
            else:
                self.has_generic_register_fake = True

    def visit_FunctionDef(self, node):
        # type: (ast.FunctionDef) -> None
        if node.name == "batched_newton_schulz":
            self._emit(node, "non_adamw_optimizer")
        for dec in node.decorator_list:
            is_op, op_name = _is_custom_op_decorator(dec)
            if is_op:
                self.custom_ops.append((dec.lineno, node.name, op_name))
                self._in_custom_op_call_nodes.add(id(dec))
            elif isinstance(dec, ast.Attribute) and dec.attr == "register_fake":
                if isinstance(dec.value, ast.Name):
                    self.fake_fn_names.add(dec.value.id)
                else:
                    self.has_generic_register_fake = True
            elif isinstance(dec, ast.Call):
                dec_name = _dotted(dec.func)
                if dec_name.endswith("register_fake") or dec_name == "register_fake":
                    self._record_register_fake_call(dec)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        # type: (ast.ClassDef) -> None
        for base in node.bases:
            base_name = _dotted(base)
            if base_name.endswith("autograd.Function") or base_name == "Function":
                self._emit(node, "custom_autograd")
        if node.name in ("Muon", "DistributedMuon"):
            self._emit(node, "non_adamw_optimizer")
        self.generic_visit(node)

    def visit_Call(self, node):
        # type: (ast.Call) -> None
        name = _dotted(node.func)
        tail = name.rsplit(".", 1)[-1]
        kw = _kw_map(node)
        keywords = set(kw.keys())

        if tail == "AdamW" and not ({"fused", "foreach"} & keywords):
            self._emit(node, "adamw_not_fused")
        elif tail in ("Muon", "DistributedMuon", "batched_newton_schulz"):
            self._emit(node, "non_adamw_optimizer")
        elif tail in ("flex_attention", "create_block_mask", "create_mask"):
            self._emit(node, "flex_attention_call")
        elif tail in ("flash_attn_varlen_func", "flash_attn_varlen_qkvpacked_func"):
            self._emit(node, "fa2_varlen_call")
        elif tail == "scaled_dot_product_attention":
            self._emit(node, "sdpa_call")
        elif tail == "MultiheadAttention" and ("nn" in name or name == tail):
            self._emit(node, "mha_default_avg_weights")
        elif tail == "multi_head_attention_forward":
            avg_val = _bool_const(kw.get("average_attn_weights"))
            need_val = _bool_const(kw.get("need_weights"))
            if avg_val is not False and need_val is not False:
                self._emit(node, "mha_default_avg_weights")
        elif tail in ("view_as_complex", "view_as_real") or (
            tail == "polar" and (name == "polar" or name.endswith("torch.polar"))
        ):
            self._emit(node, "complex_rope")
        elif tail in ("LayerNorm", "RMSNorm") and ("nn" in name or name == tail):
            self._emit(node, "layernorm_module")
        elif tail in ("Upsample", "interpolate"):
            mode_val = _str_const(kw.get("mode"))
            if mode_val == "bilinear":
                self._emit(node, "upsample_bilinear")
        elif (name.endswith("library.custom_op") or name == "custom_op") and id(node) not in self._in_custom_op_call_nodes:
            op_name = _str_const(node.args[0]) if node.args else _str_const(kw.get("name"))
            self.custom_ops.append((node.lineno, None, op_name))
        elif tail == "register_fake" or name.endswith("register_fake"):
            self._record_register_fake_call(node)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # type: (ast.Attribute) -> None
        if node.attr == "register_fake":
            if isinstance(node.value, ast.Name):
                self.fake_fn_names.add(node.value.id)
            else:
                self.has_generic_register_fake = True
        self.generic_visit(node)

    def visit_BinOp(self, node):
        # type: (ast.BinOp) -> None
        if isinstance(node.op, ast.Add):
            for side in (node.left, node.right):
                if isinstance(side, ast.Call) and "norm" in _dotted(side.func).lower():
                    self._emit(node, "norm_then_add")
                    break
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
    # type: (str, Optional[Set[str]]) -> List[Finding]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=path)
    except (OSError, SyntaxError) as exc:
        print("warning: skipped {}: {}".format(path, exc), file=sys.stderr)
        return []
    lines = source.splitlines()
    visitor = _Visitor(path, lines)
    visitor.visit(tree)
    findings = visitor.findings + _scan_manual_attention(tree, path, lines)

    for line, fn_name, op_name in visitor.custom_ops:
        matched = (
            visitor.has_generic_register_fake
            or (fn_name is not None and fn_name in visitor.fake_fn_names)
            or (op_name is not None and op_name in visitor.fake_op_names)
        )
        if not matched:
            snippet = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            findings.append(Finding(path, line, "custom_op_missing_fake", snippet, _NOTES["custom_op_missing_fake"]))

    if allowed_patterns is not None:
        findings = [f for f in findings if f.pattern in allowed_patterns]
    return findings


def _iter_files(paths):
    # type: (Iterable[str]) -> Iterable[str]
    for path in paths:
        if os.path.isfile(path) and path.endswith(".py"):
            yield path
        elif os.path.isdir(path):
            for root, dirs, names in os.walk(path):
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
    for path in _iter_files(args.paths):
        findings.extend(scan_file(path, allowed_patterns=allowed))
    findings.sort(key=lambda f: (f.file, f.line, f.pattern))

    if args.json:
        print(json.dumps([f.to_dict() for f in findings], ensure_ascii=False, indent=2))
        return 0
    if not findings:
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
    print(
        "STATIC candidates only: they do not show the code is slow, that a replacement "
        "exists for your platform, or that it is numerically equivalent. Confirm with a profiler first.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
