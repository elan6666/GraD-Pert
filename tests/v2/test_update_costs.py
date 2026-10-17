"""Do not count recomputation twice or match backward by sequence alone."""

import random
import runpy
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/v2"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "update_costs.py"))


def event(cat, name, ts, dur, tid=2, **args):
    return dict(cat=cat, name=name, ts=ts, dur=dur, tid=tid, pid=1, ph="X", args=args)


def test_disjoint_kernel_ownership_thread_sequence_recompute_and_unknown(module):
    rows = [
        event("user_annotation", "gradpert/capture_window", 0, 100),
        event("user_annotation", "gradpert/component/student/x/kda_scan", 10, 10),
        event("cpu_op", "aten::bmm", 12, 2, **{"Sequence number": 7}),
        # Sequence IDs are local to each CPU thread.
        event("user_annotation", "gradpert/component/student/head/prototypes", 20, 10, tid=3),
        event("cpu_op", "aten::mm", 22, 2, tid=3, **{"Sequence number": 7}),
        event("user_annotation", "gradpert/backward", 40, 45, tid=9),
        event(
            "cpu_op", "BmmBackward0", 41, 10, tid=9, **{"Sequence number": 7, "Fwd thread id": 1002}
        ),
        event(
            "cpu_op", "MmBackward0", 60, 10, tid=9, **{"Sequence number": 7, "Fwd thread id": 1003}
        ),
        event("user_annotation", "gradpert/checkpoint/recompute", 43, 3, tid=9),
        event("user_annotation", "gradpert/component/student/x/kda_scan", 43, 2, tid=9),
        event("cuda_runtime", "cudaLaunchKernel", 13, 1, correlation=1),
        event("cuda_runtime", "cudaLaunchKernel", 42, 1, tid=9, correlation=2),
        event("cuda_runtime", "cudaLaunchKernel", 44, 1, tid=9, correlation=3),
        event("cuda_runtime", "cudaLaunchKernel", 61, 1, tid=9, correlation=4),
        event("cuda_runtime", "cudaDeviceSynchronize", 75, 5, tid=9),
        event("kernel", "k1", 14, 5, correlation=1),
        event("kernel", "k2", 47, 5, correlation=2),
        event("kernel", "k3", 48, 5, correlation=3),  # concurrent GPU events
        event("kernel", "k4", 63, 5, correlation=4),
        event("kernel", "unknown", 95, 10, correlation=999),  # clip at window
    ]
    rows.extend(
        [
            dict(cat="fwdbwd", ph="s", id=1, pid=1, tid=2, ts=12),
            dict(cat="fwdbwd", ph="f", id=1, pid=1, tid=9, ts=41),
            dict(cat="fwdbwd", ph="s", id=2, pid=1, tid=3, ts=22),
            dict(cat="fwdbwd", ph="f", id=2, pid=1, tid=9, ts=60),
        ]
    )
    random.Random(42).shuffle(rows)
    result = module["attribute_update"](lambda: iter(rows))
    costs = {(r["phase"], r["component"]): r for r in result["costs"]}
    assert costs["forward", "kda_scan"]["gpu_duration_sum_us"] == 5
    assert costs["backward", "kda_scan"]["count"] == 1
    assert costs["recompute", "kda_scan"]["count"] == 1
    assert costs["backward", "prototypes"]["count"] == 1
    assert costs["unknown", "unknown"]["gpu_duration_sum_us"] == 5
    assert sum(r["count"] for r in result["costs"]) == 5
    assert result["autograd_to_os_thread_ids"] == {"1002": [2], "1003": [3]}
    assert result["gpu_activity_union_us"] == 21  # 5+6+5+5
    assert result["synchronization"]["cudaDeviceSynchronize"]["cpu_api_duration_sum_us"] == 5


def test_cpu_mirrors_do_not_change_capture_and_multiple_processes_fail(module):
    rows = [
        event("user_annotation", "gradpert/capture_window", 0, 100),
        event("gpu_user_annotation", "gradpert/capture_window", 10, 50),
    ]
    result = module["attribute_update"](lambda: iter(rows))
    assert result["window_us"] == 100
    assert result["costs"] == []
    rows.append(dict(cat="cpu_op", pid=2))
    with pytest.raises(ValueError, match="one CPU process"):
        module["attribute_update"](lambda: iter(rows))


def test_real_pytorch_flow_maps_os_and_autograd_threads(module, tmp_path):
    import json

    import torch

    layer = torch.nn.Linear(4, 3)
    x = torch.randn(2, 4, requires_grad=True)
    with (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile,
        torch.profiler.record_function("gradpert/capture_window"),
    ):
        with torch.profiler.record_function("gradpert/component/student/head/prototypes"):
            y = layer(x)
        with torch.profiler.record_function("gradpert/backward"):
            y.square().sum().backward()
    path = tmp_path / "tiny.json"
    profile.export_chrome_trace(str(path))
    rows = json.loads(path.read_text())["traceEvents"]
    result = module["attribute_update"](lambda: iter(rows))
    assert result["linked_backward_cpu_scopes"] > 0
    assert result["forward_backward_flow_count"] > 0
    assert result["autograd_to_os_thread_ids"]
