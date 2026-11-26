#!/usr/bin/env python3
"""Self-test for scan_data_pipeline_candidates.py (true positives + false-positive boundaries).

Pure Python standard library (compatible with Python 3.6+).
"""

import json
import os
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCANNER = os.path.join(SCRIPT_DIR, "scan_data_pipeline_candidates.py")


SAMPLE_TRUE_POSITIVES_PY = """
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, DataLoaderConfiguration
from mmcv.parallel import DataContainer

def build_train(ds, device):
    loader0 = DataLoader(ds, batch_size=8, num_workers=0, pin_memory=False)
    loader1 = DataLoader(ds, batch_size=8, num_workers=4, pin_memory=True)
    acc = Accelerator()
    cfg = DataLoaderConfiguration(non_blocking=False)
    dc = DataContainer(torch.zeros(2), cpu_only=False)
    return loader0, loader1, acc, cfg, dc

def train_step(batch, device):
    x = batch["img"].cuda()
    y = batch["label"].to(device)
    return x, y

def query_rows(self, indices, key):
    return self.hf_dataset[indices][key]

def resample_lines(lines):
    out = []
    for line in lines:
        out.append(line.interpolate(0.5))
    return out
"""

SAMPLE_FALSE_POSITIVES_PY = """
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator, DataLoaderConfiguration

def build_optimized(ds, device):
    loader = DataLoader(
        ds,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    cfg = DataLoaderConfiguration(non_blocking=True)
    acc = Accelerator(dataloader_config=cfg)
    return loader, acc

def train_safe_transfers(x, device):
    a = x.to(dtype=torch.bfloat16)
    b = x.to("cpu")
    c = x.to(device, non_blocking=True)
    d = x.cuda(non_blocking=True)
    torch.cuda.synchronize()
    return a, b, c, d

def evaluate_model(ds):
    # Eval loader findings must be downgraded to info
    eval_dataloader = DataLoader(ds, batch_size=8, num_workers=0, pin_memory=False)
    return eval_dataloader

def resize_loop(tensors):
    # F.interpolate is tensor resize, not Shapely geometry loop
    out = []
    for t in tensors:
        out.append(F.interpolate(t, scale_factor=2.0))
    return out
"""

SAMPLE_MODERN_SYNTAX_PY = """
from torch.utils.data import DataLoader

def modern_loader(ds, items):
    if (n := len(items)) > 0:
        loader = DataLoader(ds, batch_size=n, num_workers=0, pin_memory=False)
        return loader
"""

SAMPLE_YAML = """
# num_workers: 0  <-- commented out, must not fire
# pin_memory: false
data:
  workers_per_gpu: 0
  pin_memory: false
  persistent_workers: true
"""

SAMPLE_JSON = """
{
  "dataloader": {
    "num_workers": 4,
    "pin_memory": true,
    "non_blocking": false
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


def check_regressions(tmp):
    # Each fixture exercises a distinct CLI inference or location boundary.
    cases = (
        ("loader_kwargs.py", "DataLoader(ds, **dict(pin_memory=True))\n", []),
        ("accelerator_kwargs.py", "Accelerator(**options)\n", []),
        ("model_cuda.py", "def make_model(self):\n    return self.model.cuda()\n", []),
        ("explicit_device.py", "batch.to(torch.device('cuda'))\nbatch.to(device=torch.device('cpu'))\n",
         [(1, "medium", "blocking_h2d_transfer")]),
        ("explicit_copy_flag.py", "batch.to('cuda', non_blocking=False, **options)\n"
         "batch.cuda(non_blocking=False, **options)\nbatch.to('cuda', **options)\n",
         [(1, "medium", "blocking_h2d_transfer"), (2, "medium", "blocking_h2d_transfer")]),
        ("interpolate_alias.py", "from torch.nn.functional import interpolate as resize\n"
         "import torch.nn.functional as ops\n"
         "for item in batch:\n    ops.interpolate(item)\n    resize(item)\n", []),
        ("interpolate_receivers.py", "for index in indices:\n"
         "    lines[index].interpolate(0.5)\n    make_line(index).interpolate(0.5)\n",
         [(2, "medium", "geometry_interpolate_loop"), (3, "medium", "geometry_interpolate_loop")]),
        ("interpolate_scope.py", "def resize_batch(items):\n    import torch.nn.functional as ops\n"
         "    for item in items:\n        ops.interpolate(item)\n"
         "def sample_geometry(ops, distances):\n    for d in distances:\n        ops.interpolate(d)\n",
         [(7, "medium", "geometry_interpolate_loop")]),
        ("local_pin.py", "def _pin_memory(batch):\n    return DataContainer(batch.pin_memory())\n"
         "def collate(batch):\n    return DataContainer(batch)\n",
         [(4, "medium", "custom_datacontainer_pin_risk")]),
        ("container_transfer.py", "def _to_cuda(batch):\n    return DataContainer(_to_cuda(batch.data))\n"
         "def collate(batch):\n    return DataContainer(batch)\n",
         [(4, "medium", "custom_datacontainer_pin_risk")]),
        ("fallback_nested.py", "if ===:\n    DataLoader(ds, collate_fn=make(), pin_memory=True)\n", []),
        ("fallback_eval.py", "def eval_loader(ds):\n    if ===:\n"
         "        DataLoader(ds, num_workers=0, pin_memory=False)\n",
         [(3, "info", "dataloader_num_workers_zero"), (3, "info", "dataloader_pin_memory_disabled")]),
        ("fallback_hf.py", "if ===:\n    row = hf_dataset[0] # hf_dataset[0]['image']\n"
         "    image = hf_dataset[0]['image']\n",
         [(3, "info", "hf_dataset_full_row_index")]),
        ("fallback_unclosed.py", "DataLoader(ds, pin_memory=False)\n" + "DataLoader(\n" * 300,
         [(1, "info", "dataloader_pin_memory_disabled")]),
    )
    failures = []
    for filename, source, expected in cases:
        path = os.path.join(tmp, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        rc, out, err = run_cli("--json", path)
        assert rc == 0, err
        actual = [(item["line"], item["severity"], item["pattern"]) for item in json.loads(out)]
        if actual != expected:
            failures.append("%s: expected %r, got %r" % (filename, expected, actual))

    path = os.path.join(tmp, "repeated_keys.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{\n  "eval": {"pin_memory": true},\n  "train": {"pin_memory": false}\n}\n')
    rc, out, err = run_cli("--json", path)
    assert rc == 0, err
    items = json.loads(out)
    if len(items) != 1 or items[0]["line"] not in (0, 3) or "train.pin_memory" not in items[0]["snippet"]:
        failures.append("repeated_keys.json: expected the train key with a correct or unknown line, got %r" % items)

    rc, out, err = run_cli("--json", os.path.join(tmp, "absent.py"))
    if rc != 2 or out or "absent.py" not in err:
        failures.append("absent.py: expected exit 2 and a path diagnostic, got %r" % ((rc, out, err),))
    assert not failures, "\n" + "\n".join(failures)


def main():
    # 1. CLI --help check
    rc, out, err = run_cli("--help")
    assert rc == 0, "Expected --help exit 0, got %d: %s" % (rc, err)
    assert "--min-severity" in out

    with tempfile.TemporaryDirectory() as tmp:
        tp_py = os.path.join(tmp, "tp.py")
        fp_py = os.path.join(tmp, "fp.py")
        mod_py = os.path.join(tmp, "modern.py")
        cfg_yaml = os.path.join(tmp, "train.yaml")
        cfg_json = os.path.join(tmp, "config.json")

        for p, text in (
            (tp_py, SAMPLE_TRUE_POSITIVES_PY),
            (fp_py, SAMPLE_FALSE_POSITIVES_PY),
            (mod_py, SAMPLE_MODERN_SYNTAX_PY),
            (cfg_yaml, SAMPLE_YAML),
            (cfg_json, SAMPLE_JSON),
        ):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)

        # 2. True positives
        rc, out, err = run_cli("--json", tp_py)
        assert rc == 0, err
        assert "STATIC candidates only" in err
        tp_items = json.loads(out)
        tp_patterns = {item["pattern"] for item in tp_items}
        expected_tp = {
            "dataloader_num_workers_zero",
            "dataloader_pin_memory_disabled",
            "dataloader_persistent_workers_off",
            "accelerator_blocking_h2d",
            "blocking_h2d_transfer",
            "custom_datacontainer_pin_risk",
            "hf_dataset_full_row_index",
            "geometry_interpolate_loop",
        }
        missing = expected_tp - tp_patterns
        assert not missing, "Missing TP patterns: %s (got %s)" % (missing, tp_patterns)

        # 3. False-positive boundaries (min-severity medium must return 0 on fp_py)
        rc, out, err = run_cli("--json", "--min-severity", "medium", fp_py)
        assert rc == 0, err
        fp_items = json.loads(out)
        assert fp_items == [], "Unexpected medium+ findings on fp.py: %s" % fp_items

        # Eval loader findings should exist only at info severity
        rc, out, err = run_cli("--json", fp_py)
        assert rc == 0, err
        fp_all = json.loads(out)
        assert all(item["severity"] == "info" for item in fp_all), fp_all

        # 4. Modern Python syntax fallback (walrus := on Python 3.6)
        rc, out, err = run_cli("--json", mod_py)
        assert rc == 0, err
        mod_patterns = {item["pattern"] for item in json.loads(out)}
        assert "dataloader_num_workers_zero" in mod_patterns
        assert "dataloader_pin_memory_disabled" in mod_patterns

        # 5. YAML & JSON configs (comments ignored, real keys detected)
        rc, out, err = run_cli("--json", cfg_yaml, cfg_json)
        assert rc == 0, err
        cfg_items = json.loads(out)
        assert len(cfg_items) == 3, "Expected 3 config hits (2 yaml + 1 json), got %s" % cfg_items

        # 6. --no-configs suppresses YAML/JSON findings
        rc, out, err = run_cli("--json", "--no-configs", cfg_yaml, cfg_json)
        assert rc == 0, err
        assert json.loads(out) == []

        check_regressions(tmp)

    print("selftest passed: all TP, FP, fallback, config, CLI, and regression checks OK.")


if __name__ == "__main__":
    main()
