#!/usr/bin/env python3
"""Self-test for optimize-distributed-comm/scripts/scan_comm_candidates.py.

Verifies:
1. True positives for DDP, FSDP1, FSDP2, custom-stream collectives (RCCL risk),
   synchronous loop collectives, gather_object, and DeepSpeed JSON/YAML configs.
2. False-positive boundaries: fully tuned DDP/FSDP1/FSDP2 wrappers, isinstance()
   checks, async_op=True loop collectives, and unrelated JSON files do not fire.
3. CLI flags (--json, --min-severity, --no-configs) and stderr disclaimer.

Usage:
    python3 scripts/selftest.py
"""

import json
import os
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_SCRIPT = os.path.join(SCRIPT_DIR, "scan_comm_candidates.py")


def _run(cmd):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    assert proc.returncode == 0, "Command failed (%s): %s\nSTDERR:\n%s" % (
        proc.returncode, cmd, proc.stderr
    )
    return proc.stdout, proc.stderr


def test_python_and_configs(tmpdir):
    bad_py = os.path.join(tmpdir, "bad_comm.py")
    with open(bad_py, "w", encoding="utf-8") as handle:
        handle.write(
            "from __future__ import annotations\n"
            "import torch\n"
            "import torch.distributed as dist\n"
            "from torch.nn.parallel import DistributedDataParallel\n"
            "from torch.distributed.fsdp import FullyShardedDataParallel, fully_shard\n"
            "\n"
            "def setup(model, blocks, stream, tensors):\n"
            "    m1 = DistributedDataParallel(model, find_unused_parameters=True)\n"
            "    m2 = FullyShardedDataParallel(model, limit_all_gathers=True)\n"
            "    fsdp_cfg = {'reshard_after_forward': True}\n"
            "    for b in blocks:\n"
            "        fully_shard(b, **fsdp_cfg)\n"
            "    with torch.cuda.stream(stream):\n"
            "        dist.all_reduce(tensors[0])\n"
            "    for t in tensors:\n"
            "        dist.all_gather_into_tensor(t, t)\n"
            "    dist.gather_object(None, None)\n"
            "    return m1, m2\n"
        )

    good_py = os.path.join(tmpdir, "good_comm.py")
    with open(good_py, "w", encoding="utf-8") as handle:
        handle.write(
            "import torch.distributed as dist\n"
            "from torch.nn.parallel import DistributedDataParallel\n"
            "from torch.distributed.fsdp import FullyShardedDataParallel, BackwardPrefetch, fully_shard\n"
            "\n"
            "def setup_good(model, block, tensors):\n"
            "    is_ddp = isinstance(model, DistributedDataParallel)\n"
            "    m1 = DistributedDataParallel(\n"
            "        model,\n"
            "        find_unused_parameters=False,\n"
            "        static_graph=True,\n"
            "        gradient_as_bucket_view=True,\n"
            "        broadcast_buffers=False,\n"
            "    )\n"
            "    m2 = FullyShardedDataParallel(\n"
            "        model,\n"
            "        limit_all_gathers=False,\n"
            "        forward_prefetch=True,\n"
            "        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,\n"
            "    )\n"
            "    fully_shard(block, reshard_after_forward=False)\n"
            "    for t in tensors:\n"
            "        dist.all_reduce(t, async_op=True)\n"
            "    return is_ddp, m1, m2\n"
        )

    ds_json = os.path.join(tmpdir, "ds_config.json")
    with open(ds_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "zero_optimization": {
                    "stage": 2,
                    "overlap_comm": False,
                    "contiguous_gradients": False,
                }
            },
            handle,
            indent=2,
        )

    other_json = os.path.join(tmpdir, "other.json")
    with open(other_json, "w", encoding="utf-8") as handle:
        json.dump({"batch_size": 16, "lr": 1e-4}, handle)

    ds_yaml = os.path.join(tmpdir, "ds_config.yaml")
    with open(ds_yaml, "w", encoding="utf-8") as handle:
        handle.write(
            "zero_optimization:\n"
            "  stage: 2\n"
            "  overlap_comm: false\n"
            "  # contiguous_gradients: false\n"
            "  contiguous_gradients: true\n"
        )

    out, err = _run([sys.executable, SCAN_SCRIPT, "--json", tmpdir])
    assert "STATIC candidates only" in err, "Missing static disclaimer in stderr"
    assert "trace-bottleneck-report" in err, "Missing T5 triage guidance in stderr"
    findings = json.loads(out)

    bad_patterns = {f["pattern"] for f in findings if f["file"] == bad_py}
    for expected in (
        "ddp_find_unused_true",
        "ddp_conservative_defaults",
        "fsdp1_limit_all_gathers_true",
        "fsdp1_missing_prefetch",
        "fsdp2_reshard_after_forward_true",
        "collective_in_custom_stream",
        "collective_in_loop_sync",
        "dist_gather_object",
    ):
        assert expected in bad_patterns, "Missing %s in %s" % (expected, bad_patterns)

    good_findings = [f for f in findings if f["file"] == good_py]
    good_patterns = {f["pattern"] for f in good_findings}
    assert not (good_patterns & {
        "ddp_find_unused_true",
        "ddp_conservative_defaults",
        "fsdp1_limit_all_gathers_true",
        "fsdp1_missing_prefetch",
        "fsdp2_reshard_after_forward_true",
        "fsdp2_implicit_reshard",
        "collective_in_loop_sync",
        "collective_in_custom_stream",
    }), "Unexpected false positives in good_comm.py: %s" % good_patterns

    json_patterns = {f["pattern"] for f in findings if f["file"] == ds_json}
    assert json_patterns == {"ds_overlap_comm_disabled", "ds_contiguous_gradients_disabled"}, json_patterns
    assert not any(f["file"] == other_json for f in findings), "Unrelated JSON should produce 0 findings"

    yaml_patterns = {f["pattern"] for f in findings if f["file"] == ds_yaml}
    assert yaml_patterns == {"ds_overlap_comm_disabled"}, "Commented YAML line must not fire: %s" % yaml_patterns

    # Check --no-configs excludes JSON/YAML
    out_nc, _ = _run([sys.executable, SCAN_SCRIPT, "--json", "--no-configs", tmpdir])
    nc_findings = json.loads(out_nc)
    assert all(f["file"].endswith(".py") for f in nc_findings)

    # Check --min-severity high
    out_hi, _ = _run([sys.executable, SCAN_SCRIPT, "--json", "--min-severity", "high", tmpdir])
    hi_findings = json.loads(out_hi)
    assert hi_findings and all(f["severity"] == "high" for f in hi_findings)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_python_and_configs(tmpdir)
    print("PASS: scan_comm_candidates.py self-tests succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
