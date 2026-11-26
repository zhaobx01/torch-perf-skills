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
        msg = "torch.cuda.empty_cache() # use_cache=True"
        out = model(items, use_cache=True)
        for batch in items:
            torch.cuda.empty_cache()
        return out, msg
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
    "use_checkpointing": true,
    "use_cache": true
  }
}
"""

SCANNER_REGRESSIONS = (
    (
        "checkpoint-imports.py",
        """from torch.utils import checkpoint as activation
from torch.utils.checkpoint import checkpoint as remat
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl as Impl, checkpoint_wrapper as wrap,
)
from my_db import checkpoint_wrapper as db_wrap
activation.checkpoint(block, x)
remat(block, x, use_reentrant=True)
wrap(block, checkpoint_impl=Impl.REENTRANT)
db.checkpoint("wal", 1)
db_wrap(block)
local_checkpoint = activation.checkpoint
local_checkpoint(block, x, **checkpoint_kwargs)
wrap(block, Impl.REENTRANT)
cfg = dict(use_checkpoint=True)
blk = Block(use_checkpointing=True)
""",
        [(7, "ac_checkpoint_call", "medium"), (8, "ac_reentrant_true", "high"),
         (9, "ac_reentrant_true", "high"), (13, "ac_checkpoint_call", "medium"),
         (14, "ac_reentrant_true", "high"), (15, "ac_checkpoint_call", "medium"),
         (16, "ac_checkpoint_call", "medium")],
    ),
    (
        "garbage-collection.py",
        """import gc
from gc import collect as collect_now, freeze as freeze_now
from torch.cuda import empty_cache as flush
def train():
    def freeze_after_warmup():
        gc.collect(2)
        gc.freeze()
        collect_now(2)
        freeze_now()
    for batch in get_loader(flush()):
        gc.collect(2)
        collect_now()
        flush()
""",
        [(10, "hotpath_empty_cache", "medium"), (11, "hotpath_gc_collect", "medium"),
         (12, "hotpath_gc_collect", "medium"), (13, "hotpath_empty_cache", "high")],
    ),
    (
        "kv-context.py",
        """def parse(payload):
    parse_xml(payload, use_cache=True)
def train_decoder(model, x):
    model(x, use_cache=True)
def generate_tokens(use_cache=True):
    model.generate(x, use_cache=True)
class Qwen3VLTextConfig:
    def __init__(self, *, use_cache=True):
        self.use_cache = use_cache
def train_step(model, x):
    model.generate(x, use_cache=True)
def evaluate_during_training(model, x):
    model(x, use_cache=True)
""",
        [(2, "kv_cache_enabled", "info"), (4, "kv_cache_enabled", "medium"),
         (5, "kv_cache_enabled", "info"), (6, "kv_cache_enabled", "info"),
         (8, "kv_cache_enabled", "info"), (11, "kv_cache_enabled", "info"),
         (13, "kv_cache_enabled", "info")],
    ),
    (
        "gc-scope.py",
        """import gc as gc_module
from custom import cleanup
class Helper:
    from gc import collect as cleanup
    def method(self):
        cleanup(2)
def train(gc):
    gc.collect()
    cleanup()
def training_step(batch):
    gc_module.collect(2)
    gc_module.freeze()
def freeze_gc():
    gc_module.collect(2)
    gc_module.freeze()
for _ in steps:
    freeze_gc()
""",
        [(11, "hotpath_gc_collect", "info"), (14, "hotpath_gc_collect", "info")],
    ),
    (
        "scalar-config.yaml",
        """model:
  "with_cp": true  # a plain boolean value
  use_cache: true-false
  'gradient_checkpointing': True
  with_cp: "true"
  use_cache: true#not-a-comment
  description: |-
    with_cp: true
  use_checkpoint: true
  use_checkpointing: true
""",
        [(2, "ac_with_cp_true", "medium"), (4, "ac_checkpoint_call", "medium"),
         (9, "ac_checkpoint_call", "medium"), (10, "ac_checkpoint_call", "medium")],
    ),
)

FALLBACK_GC_SAMPLE = """# gc.freeze() mentioned in a comment
def warmup():
    gc.collect(2)
    gc.freeze()
def train():
    for _ in steps:
        gc.collect(2)
def training_step(batch):
    gc.collect(2)
    gc.freeze()
"""


def run_cli(*args):
    proc = subprocess.run(
        [sys.executable, SCANNER] + list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def check_regressions(tmp):
    failures = []
    for name, content, expected in SCANNER_REGRESSIONS:
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as sample:
            sample.write(content)
        rc, out, err = run_cli("--json", path)
        assert rc == 0, err
        actual = [(f["line"], f["pattern"], f["severity"]) for f in json.loads(out)["findings"]]
        if actual != expected:
            failures.append("{}: expected {}, got {}".format(name, expected, actual))
    fallback = smc._scan_python_regex_fallback("fallback.py", FALLBACK_GC_SAMPLE)
    actual = [(f.line, f.pattern, f.severity) for f in fallback]
    expected = [(7, "hotpath_gc_collect", "medium"), (9, "hotpath_gc_collect", "info")]
    if actual != expected:
        failures.append("fallback GC: expected {}, got {}".format(expected, actual))
    fallback = smc._scan_python_regex_fallback("fallback.py", "def forward(use_cache=True):\n    pass\n")
    actual = [(f.line, f.pattern, f.severity) for f in fallback]
    expected = [(1, "kv_cache_enabled", "info")]
    if actual != expected:
        failures.append("fallback default: expected {}, got {}".format(expected, actual))
    # Regex fallback must ignore string literals containing keywords/calls
    fallback = smc._scan_python_regex_fallback(
        "strings.py", 'msg = "torch.cuda.empty_cache() # use_cache=True"\n'
    )
    if fallback:
        failures.append("fallback strings: expected [], got {}".format([f.to_dict() for f in fallback]))
    assert not failures, "\n".join(failures)


def check_cli_status(tmp):
    # Missing path -> exit 2
    rc, out, err = run_cli("--json", os.path.join(tmp, "missing.py"))
    assert rc == 2 and "path not found" in err and not out, (rc, out, err)

    # Empty directory -> exit 1
    empty_dir = os.path.join(tmp, "empty_dir")
    os.mkdir(empty_dir)
    rc, out, err = run_cli("--json", empty_dir)
    assert rc == 1, (rc, out, err)
    payload = json.loads(out)
    assert payload["files_scanned"] == 0 and payload["files_skipped"] == 0, payload
    assert "STATIC candidates only" in payload["disclaimer"], payload

    # Malformed JSON alongside valid file -> exit 1 with skipped_files
    bad_json = os.path.join(tmp, "broken.json")
    good_py = os.path.join(tmp, "ok.py")
    with open(bad_json, "w", encoding="utf-8") as handle:
        handle.write("{not valid json")
    with open(good_py, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")
    rc, out, err = run_cli("--json", good_py, bad_json)
    assert rc == 1, (rc, out, err)
    payload = json.loads(out)
    assert payload["files_scanned"] == 1 and payload["files_skipped"] == 1, payload
    assert payload["skipped_files"][0]["file"] == bad_json, payload


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

        # 1. True positives + JSON envelope
        rc, out, err = run_cli("--json", tp_py)
        assert rc == 0, "rc={}: {}".format(rc, err)
        assert "STATIC candidates only" in err, "missing static disclaimer in stderr: {}".format(err)
        payload = json.loads(out)
        assert "STATIC candidates only" in payload["disclaimer"], payload
        assert payload["files_scanned"] == 1 and payload["files_skipped"] == 0, payload
        tp_findings = payload["findings"]
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
        fp_findings = json.loads(out)["findings"]
        non_info = [f for f in fp_findings if f["severity"] != "info"]
        assert not non_info, "unexpected non-info findings on safe code: {}".format(non_info)
        # Only expected info finding is generate_tokens use_cache=True; gc.collect paired with gc.freeze must be absent
        fp_patterns = [f["pattern"] for f in fp_findings]
        assert fp_patterns == ["kv_cache_enabled"], "unexpected fp patterns: {}".format(fp_findings)

        # 3. Regex fallback unit test + CLI check (string literal on line 7 must NOT fire)
        fallback_findings = smc._scan_python_regex_fallback(mod_py, SAMPLE_MODERN_SYNTAX_PY)
        assert [(f.line, f.pattern) for f in fallback_findings] == [
            (8, "kv_cache_enabled"),
            (10, "hotpath_empty_cache"),
        ], [f.to_dict() for f in fallback_findings]
        rc, out, err = run_cli("--json", mod_py)
        assert rc == 0, err
        if sys.version_info < (3, 8):
            assert "using regex fallback" in err, err
        mod_items = [(item["line"], item["pattern"]) for item in json.loads(out)["findings"]]
        assert mod_items == [(8, "kv_cache_enabled"), (10, "hotpath_empty_cache")], mod_items

        # 4. YAML & JSON config scanning + --no-configs + --min-severity high
        rc, out, err = run_cli("--json", cfg_yaml, cfg_json)
        assert rc == 0, err
        cfg_findings = json.loads(out)["findings"]
        yaml_hits = [f for f in cfg_findings if f["file"] == cfg_yaml]
        json_hits = [f for f in cfg_findings if f["file"] == cfg_json]
        assert len(yaml_hits) == 2, "expected 2 yaml hits, got {}".format(yaml_hits)
        assert len(json_hits) == 3, "expected 3 json hits, got {}".format(json_hits)
        config_kv = [f for f in cfg_findings if f["pattern"] == "kv_cache_enabled"]
        assert len(config_kv) == 2 and all(f["severity"] == "info" for f in config_kv), config_kv

        rc, out, err = run_cli("--json", "--no-configs", tmp)
        assert rc == 0, err
        no_cfg = json.loads(out)["findings"]
        assert all(f["file"].endswith(".py") for f in no_cfg), no_cfg

        rc, out, err = run_cli("--json", "--min-severity", "high", tp_py)
        assert rc == 0, err
        high_only = json.loads(out)["findings"]
        assert high_only and all(f["severity"] == "high" for f in high_only), high_only

        check_regressions(tmp)
        check_cli_status(tmp)

    print("PASS: scan_memory_candidates selftest (candidate, scope, alias, config, fallback, JSON envelope, and CLI status passed)")


if __name__ == "__main__":
    main()
