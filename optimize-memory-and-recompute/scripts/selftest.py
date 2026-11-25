#!/usr/bin/env python3
"""Self-test for scan_memory_candidates.py (true positives + false-positive boundaries).

Pure Python standard library (compatible with Python 3.6+).
"""

import json
import os
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCANNER = os.path.join(SCRIPT_DIR, "scan_memory_candidates.py")
sys.path.insert(0, SCRIPT_DIR)

import scan_memory_candidates as smc  # noqa: E402


SAMPLE_TRUE_POSITIVES_PY = """
import gc
import torch
import torch.utils.checkpoint as cp

model_cfg = dict(
    type="ResNet",
    depth=50,
    with_cp=True,
)
extra_cfg = {"with_cp": True, "use_cache": True}

def setup_ac(model):
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )
    cp.checkpoint(model, x, use_reentrant=False)
    checkpoint_wrapper(model.blocks[0])

def train_step(model, loader):
    out = model(x, use_cache=True)
    for batch in loader:
        loss = model(batch)
        torch.cuda.empty_cache()
        gc.collect()
    return out
"""

SAMPLE_FALSE_POSITIVES_PY = """
import gc
import torch
import torch.utils.checkpoint as cp

# Disabled flags should NOT trigger
safe_cfg = dict(type="ResNet", with_cp=False, use_cache=False)
safe_dict = {"with_cp": False, "use_cache": False, "gradient_checkpointing": False}

def setup_safe(model):
    model.set_gradient_checkpointing(False)
    model.gradient_checkpointing = False
    model.use_cache = False

def generate_tokens(model, ids):
    # Inference/generation helper should be info-only, never high
    return model(ids, use_cache=True)

def freeze_gc_after_warmup():
    # Paired gc.collect(2) + gc.freeze() is the good M6 pattern, should NOT fire
    gc.collect(2)
    gc.freeze()
"""

SAMPLE_MODERN_SYNTAX_PY = """
import gc
import torch

def train_modern(model, items):
    if (n := len(items)) > 0:
        out = model(items, use_cache=True)
        for batch in items:
            torch.cuda.empty_cache()
        return out
"""

SAMPLE_YAML = """
# with_cp: true  <-- commented out, must not fire
model:
  with_cp: true
  use_cache: true
  gradient_checkpointing: false
"""

SAMPLE_JSON = """
{
  "model": {
    "with_cp": false,
    "gradient_checkpointing": true,
    "use_cache": true
  }
}
"""


def run_cli(*args):
    proc = subprocess.run(
        [sys.executable, SCANNER] + list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tp_py = os.path.join(tmp, "tp.py")
        fp_py = os.path.join(tmp, "fp.py")
        mod_py = os.path.join(tmp, "modern.py")
        cfg_yaml = os.path.join(tmp, "train.yaml")
        cfg_json = os.path.join(tmp, "config.json")

        for path, content in (
            (tp_py, SAMPLE_TRUE_POSITIVES_PY),
            (fp_py, SAMPLE_FALSE_POSITIVES_PY),
            (mod_py, SAMPLE_MODERN_SYNTAX_PY),
            (cfg_yaml, SAMPLE_YAML),
            (cfg_json, SAMPLE_JSON),
        ):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        # 1. True positives
        rc, out, err = run_cli("--json", tp_py)
        assert rc == 0, "rc={}: {}".format(rc, err)
        assert "STATIC candidates only" in err, "missing static disclaimer in stderr: {}".format(err)
        tp_findings = json.loads(out)
        tp_patterns = {item["pattern"] for item in tp_findings}
        expected_tp = {
            "ac_with_cp_true",
            "ac_reentrant_true",
            "ac_checkpoint_call",
            "kv_cache_enabled",
            "hotpath_empty_cache",
            "hotpath_gc_collect",
        }
        missing = expected_tp - tp_patterns
        assert not missing, "missing TP patterns: {} (got {})".format(missing, tp_patterns)

        empty_cache_sevs = [
            item["severity"] for item in tp_findings if item["pattern"] == "hotpath_empty_cache"
        ]
        assert "high" in empty_cache_sevs, "loop empty_cache must be high: {}".format(empty_cache_sevs)

        # 2. False-positive boundaries
        rc, out, err = run_cli("--json", fp_py)
        assert rc == 0, err
        fp_findings = json.loads(out)
        non_info = [f for f in fp_findings if f["severity"] != "info"]
        assert not non_info, "unexpected non-info findings on safe code: {}".format(non_info)
        # Only expected info finding is generate_tokens use_cache=True; gc.collect paired with gc.freeze must be absent
        fp_patterns = [f["pattern"] for f in fp_findings]
        assert fp_patterns == ["kv_cache_enabled"], "unexpected fp patterns: {}".format(fp_findings)

        # 3. Regex fallback unit test (works on both Python 3.6 and 3.8+)
        fallback_findings = smc._scan_python_regex_fallback(mod_py, SAMPLE_MODERN_SYNTAX_PY)
        fb_patterns = {f.pattern for f in fallback_findings}
        assert "kv_cache_enabled" in fb_patterns and "hotpath_empty_cache" in fb_patterns, (
            "regex fallback missed patterns: {}".format(fb_patterns)
        )
        # Also verify CLI handles modern.py without crashing on Python 3.6+
        rc, out, err = run_cli("--json", mod_py)
        assert rc == 0, err
        mod_patterns = {item["pattern"] for item in json.loads(out)}
        assert "kv_cache_enabled" in mod_patterns and "hotpath_empty_cache" in mod_patterns, mod_patterns

        # 4. YAML & JSON config scanning + --no-configs + --min-severity high
        rc, out, err = run_cli("--json", cfg_yaml, cfg_json)
        assert rc == 0, err
        cfg_findings = json.loads(out)
        # YAML comment line must not be double-counted: 2 hits in yaml (with_cp, use_cache), 2 in json (gc, use_cache)
        yaml_hits = [f for f in cfg_findings if f["file"] == cfg_yaml]
        json_hits = [f for f in cfg_findings if f["file"] == cfg_json]
        assert len(yaml_hits) == 2, "expected 2 yaml hits, got {}".format(yaml_hits)
        assert len(json_hits) == 2, "expected 2 json hits, got {}".format(json_hits)

        rc, out, err = run_cli("--json", "--no-configs", tmp)
        assert rc == 0, err
        no_cfg = json.loads(out)
        assert all(f["file"].endswith(".py") for f in no_cfg), no_cfg

        rc, out, err = run_cli("--json", "--min-severity", "high", tp_py)
        assert rc == 0, err
        high_only = json.loads(out)
        assert high_only and all(f["severity"] == "high" for f in high_only), high_only

    print("PASS: scan_memory_candidates selftest (all TP, FP, fallback, config, and CLI checks passed)")


if __name__ == "__main__":
    main()
