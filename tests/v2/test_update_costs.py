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


def test_module_groups_keep_graph_sequence_and_recompute_disjoint(module):
    rows = [
        event("user_annotation", "gradpert/capture_window", 0, 100),
        event("user_annotation", "gradpert/component/student/graph.layers.0/kda_scan", 5, 20),
        event("cuda_runtime", "cudaLaunchKernel", 10, 1, correlation=1),
        event("kernel", "graph", 12, 4, correlation=1),
        event("user_annotation", "gradpert/checkpoint/recompute", 30, 20),
        event("user_annotation", "gradpert/component/student/cell.layers.0/kda_scan", 32, 15),
        event("cuda_runtime", "cudaLaunchKernel", 35, 1, correlation=2),
        event("kernel", "cell", 37, 6, correlation=2),
        event("kernel", "unlinked", 60, 2, correlation=99),
    ]
    by_component = module["attribute_update"](lambda: iter(rows))
    by_module = module["attribute_update"](lambda: iter(rows), group_by="module")
    costs = {(r["phase"], r["component"]): r for r in by_module["costs"]}
    assert costs["forward", "component/student/graph.layers.0/kda_scan"]["gpu_duration_sum_us"] == 4
    assert (
        costs["recompute", "component/student/cell.layers.0/kda_scan"]["gpu_duration_sum_us"] == 6
    )
    assert costs["unknown", "unknown"]["gpu_duration_sum_us"] == 2
    assert by_component["gpu_activity_union_us"] == by_module["gpu_activity_union_us"] == 12
    assert sum(r["count"] for r in by_module["costs"]) == 3
    assert sum(r["cpu_kernel_launch_count"] for r in by_module["costs"]) == 2
    with pytest.raises(ValueError, match="group_by"):
        module["attribute_update"](lambda: iter(rows), group_by="guessed")


def test_allocator_traffic_is_not_live_saved_memory(module):
    def memory(ts, size, total, device=1):
        return dict(
            name="[memory]",
            ts=ts,
            tid=2,
            pid=1,
            ph="i",
            args={"Device Type": device, "Bytes": size, "Total Allocated": total},
        )

    rows = [
        event("user_annotation", "gradpert/capture_window", 0, 100),
        event("user_annotation", "gradpert/component/student/graph/kda_scan", 10, 20),
        event("user_annotation", "gradpert/checkpoint/recompute", 40, 20),
        event("user_annotation", "gradpert/component/student/cell/kda_scan", 42, 15),
        memory(-1, 128, 1024),  # allocated before the captured window
        memory(12, 64, 1088),
        memory(15, 10000, 11088, device=0),  # host memory is not CUDA
        memory(45, -128, 960),  # released here, not allocated by this scope
        memory(50, 16, 976),
        memory(100, 1000, 1976),  # outside half-open capture
    ]
    random.Random(21).shuffle(rows)
    result = module["attribute_update"](lambda: iter(rows), group_by="module")
    stats = result["cuda_allocator_events"]
    assert stats["observed_peak_allocated_bytes"] == 1088
    assert stats["by_host_scope"] == [
        dict(
            phase="forward",
            component="component/student/graph/kda_scan",
            allocated_bytes=64,
            freed_bytes=0,
            events=1,
        ),
        dict(
            phase="recompute",
            component="component/student/cell/kda_scan",
            allocated_bytes=16,
            freed_bytes=128,
            events=2,
        ),
    ]
    assert not result["cuda_activity_present"]
    assert result["gpu_activity_union_us"] is None
    empty = module["attribute_update"](
        lambda: iter([event("user_annotation", "gradpert/capture_window", 0, 100)])
    )
    assert empty["cuda_allocator_events"]["observed_peak_allocated_bytes"] is None


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
