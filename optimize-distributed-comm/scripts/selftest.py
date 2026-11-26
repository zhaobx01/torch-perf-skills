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


def test_config_boundaries(tmpdir):
    cases = (
        ("stage3.yaml", "zero_optimization:\n  stage: 3\n  overlap_comm: false\n", [], False),
        ("unrelated.yaml", "pipeline:\n  contiguous_gradients: false\n", [], False),
        ("block_string.yaml", "notes: |\n  overlap_comm: false\n", [], False),
        ("inline.yaml", 'zero_optimization: {"stage": 2, overlap_comm: false} # note\n',
         ["ds_overlap_comm_disabled"], False),
        ("inline_nested.yaml", "zero_optimization: {stage: 2, extra: {a: 1, overlap_comm: false, z: 0}}\n",
         [], True),
        ("mixed_case.yaml", "zero_optimization: {stage: 2, overlap_comm: fAlSe}\n", [], False),
        ("missing_space.yaml", "zero_optimization:\n  stage: 2\n  overlap_comm:false\n", [], True),
        ("quoted_keys.yaml", 'zero_optimization: {"stage":2,"overlap_comm":false}\n',
         ["ds_overlap_comm_disabled"], False),
        ("root_multiline.yaml", 'notes: "some text\nzero_optimization: {stage: 2, overlap_comm: false}\n'
         'next: more"\n', [], True),
        ("root_flow.yaml", "other: {\nzero_optimization: {stage: 2, overlap_comm: false}\n}\n",
         [], True),
        ("root_explicit.yaml", "zero_optimization: {stage: 2, overlap_comm: false}\n"
         "? zero_optimization\n: {stage: 3}\n", [], True),
        ("root_escaped.yaml", 'zero_optimization: {stage: 2, overlap_comm: false}\n'
         '"zero_optimiza\\u0074ion": {stage: 3}\n', [], True),
        ("nested.yml", "zero_optimization:\n  stage: 2\n  extra:\n    overlap_comm: false\n", [], False),
        ("anchor.yaml", "defaults: &defaults\n  stage: 2\nzero_optimization:\n  <<: *defaults\n"
         "  overlap_comm: false\n", [], True),
        ("anchored_stage.yml", "zero_optimization:\n  stage: &stage 2\n  overlap_comm: false\n", [], True),
        ("duplicate.yml", "zero_optimization:\n  stage: 2\n  stage: 3\n  overlap_comm: false\n", [], True),
        ("second_doc.yml", "zero_optimization: {stage: 2, overlap_comm: false}\n"
         "---\nzero_optimization: {stage: 3}\n", [], True),
        ("boolean.json", '{"zero_optimization": {"stage": true, "overlap_comm": false}}', [], False),
        ("zero2.json", '{"zero_optimization": {"stage": 2, "overlap_comm": false}}',
         ["ds_overlap_comm_disabled"], False),
    )
    errors = []
    for name, source, expected, warns in cases:
        path = os.path.join(tmpdir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        out, err = _run([sys.executable, SCAN_SCRIPT, "--json", path])
        result = [finding["pattern"] for finding in json.loads(out)]
        skipped = "unsupported YAML" in err
        if result != expected or skipped != warns:
            errors.append((name, expected, result, warns, skipped))
    assert not errors, "Config mismatch (file, expected, actual, warning expected/actual): %r" % errors

    path = os.path.join(tmpdir, "json_location.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('{"metadata": {"overlap_comm": true},\n'
                     ' "zero_optimization": {"stage": 2,\n'
                     ' "overlap_comm": false}}\n')
    out, _ = _run([sys.executable, SCAN_SCRIPT, "--json", path])
    assert [(f["line"], f["pattern"]) for f in json.loads(out)] == [(3, "ds_overlap_comm_disabled")]


def test_python_bindings(tmpdir):
    path = os.path.join(tmpdir, "bindings.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "import torch\n"
            "import torch.distributed as td\n"
            "from torch.distributed import reduce_scatter_tensor as rs\n"
            "from torch.distributed.fsdp import FullyShardedDataParallel as Wrap, fully_shard as shard\n"
            "from torch.nn.parallel import DistributedDataParallel as WrapDDP\n"
            "from accelerate.utils import DistributedDataParallelKwargs as DDPConfig\n"
            "from other import dist, DDP\n"
            "def setup(model, blocks, options):\n"
            "    DDP(model, find_unused_parameters=True)\n"
            "    DDPConfig(find_unused_parameters=True)\n"
            "    Wrap(model, **options)\n"
            "    WrapDDP(model, **options)\n"
            "    config = dict(reshard_after_forward=True)\n"
            "    WrapDDP(model, **{'find_unused_parameters': True})\n"
            "    shard(model, **config)\n"
            "    shard(model, reshard_after_forward=2)\n"
            "    shard(model)\n"
            "    config['reshard_after_forward'] = False\n"
            "    shard(model, **config)\n"
            "    with cache.stream(s):\n"
            "        td.all_reduce(t)\n"
            "    with torch.cuda.stream(s):\n"
            "        dist.all_reduce(t)\n"
            "        td.all_reduce(t)\n"
            "    for layer in blocks:\n"
            "        rs(t, t)\n"
            "def shadow(td):\n"
            "    td.all_reduce(t)\n"
        )
    out, _ = _run([sys.executable, SCAN_SCRIPT, "--json", path])
    result = {(f["line"], f["pattern"]) for f in json.loads(out)}
    expected = {
        (10, "ddp_find_unused_true"),
        (14, "ddp_find_unused_true"),
        (14, "ddp_conservative_defaults"),
        (15, "fsdp2_reshard_after_forward_true"),
        (17, "fsdp2_implicit_reshard"),
        (21, "collective_call"),
        (24, "collective_in_custom_stream"),
        (26, "collective_in_loop_sync"),
    }
    assert result == expected, (result - expected, expected - result)


def test_missing_input(tmpdir):
    path = os.path.join(tmpdir, "missing.py")
    result = subprocess.run(
        [sys.executable, SCAN_SCRIPT, "--json", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    assert result.returncode != 0 and path in result.stderr, result


def test_async_call_and_fallback(tmpdir):
    samples = {
        "positional.py": (
            "import torch.distributed as dist\n"
            "for t in tensors:\n"
            "    dist.all_reduce(t, dist.ReduceOp.SUM, None, True)\n"
            "    dist.all_reduce(t, dist.ReduceOp.SUM, None, False)\n"
            "    dist.all_reduce(t, *options)\n",
            [(3, "collective_call"), (4, "collective_in_loop_sync"), (5, "collective_call")],
        ),
        "nested_scopes.py": (
            "import torch.distributed as dist\n"
            "from torch.nn.parallel import DistributedDataParallel as DDP\n"
            "config = dict(find_unused_parameters=True)\n"
            "try:\n"
            "    config['find_unused_parameters'] = False\n"
            "finally:\n"
            "    DDP(model, **config)\n"
            "[dist.all_reduce(t) for dist in custom_backends]\n"
            "factory().dist.all_reduce(t)\n",
            [],
        ),
        "wrapper_args.py": (
            "from torch.distributed.fsdp import FullyShardedDataParallel as FSDP\n"
            "from torch.nn.parallel import DistributedDataParallel as DDP\n"
            "FSDP(model, group, strategy, backward_prefetch)\n"
            "DDP(*wrapper_args)\n"
            "DDP(model, static_graph=True, find_unused_parameters=True, gradient_as_bucket_view=True)\n",
            [],
        ),
        "class_scope.py": (
            "import torch\n"
            "from other import dist\n"
            "class State:\n"
            "    import torch.distributed as dist\n"
            "    def launch(self, s, t):\n"
            "        with torch.cuda.stream(s):\n"
            "            dist.all_reduce(t)\n",
            [],
        ),
        "match_scope.py": (
            "import torch\n"
            "import torch.distributed as dist\n"
            "match other_backends:\n"
            "    case {'preferred': dist}:\n"
            "        with torch.cuda.stream(s):\n"
            "            dist.all_reduce(t)\n",
            [],
        ),
        "syntax.py": (
            "print('find_unused_parameters=True')\n"
            "# find_unused_parameters=True\n"
            "x =\n"
            "DDP(model, find_unused_parameters=True)\n",
            [(4, "ddp_find_unused_true")],
        ),
    }
    mismatches = []
    for name, (source, expected) in samples.items():
        path = os.path.join(tmpdir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        out, err = _run([sys.executable, SCAN_SCRIPT, "--json", path])
        result = [(f["line"], f["pattern"]) for f in json.loads(out)]
        if result != expected:
            mismatches.append((name, result, expected))
        if name == "syntax.py":
            assert "using regex fallback" in err
    assert not mismatches, mismatches


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_python_and_configs(tmpdir)
        test_config_boundaries(tmpdir)
        test_python_bindings(tmpdir)
        test_missing_input(tmpdir)
        test_async_call_and_fallback(tmpdir)
    print("PASS: scan_comm_candidates.py self-tests succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
