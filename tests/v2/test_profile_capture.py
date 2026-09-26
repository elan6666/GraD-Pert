"""Diagnostic instrumentation must not change the model update or count overlaps twice."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.optimizer import V2Optimizer

ROOT = Path(__file__).resolve().parents[2]


def load_script(relative):
    spec = importlib.util.spec_from_file_location("profile_test_helper", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROFILE = load_script("scripts/v2/profile_capture.py")
FIXTURE = load_script("tests/v2/test_relay_method.py")


def test_trace_union_counts_parallel_kernels_once_and_missing_cuda_is_unknown():
    trace = {
        "traceEvents": [
            {"ph": "X", "name": "gradpert/capture_window", "ts": 0, "dur": 100},
            {"ph": "X", "cat": "kernel", "ts": 10, "dur": 40},
            {"ph": "X", "cat": "kernel", "ts": 30, "dur": 40},
            {"ph": "X", "cat": "gpu_memcpy", "ts": 80, "dur": 30},
            {"ph": "X", "cat": "cpu_op", "ts": 0, "dur": 100},
        ]
    }
    summary = PROFILE.trace_summary(trace)
    assert summary["gpu_busy_union_us"] == 80
    assert summary["gpu_busy_fraction"] == 0.8
    assert summary["kernel_count"] == 2 and summary["memcpy_count"] == 1
    assert summary["largest_gpu_gaps_start_and_duration_us"] == [(0, 10), (70, 10)]
    trace["traceEvents"] = trace["traceEvents"][:1]
    assert PROFILE.trace_summary(trace)["gpu_busy_fraction"] is None
    with pytest.raises(ValueError, match="exactly one"):
        PROFILE.trace_summary({"traceEvents": []})


def test_profile_preserves_complete_update_rng_and_restores_wrapped_methods(tmp_path):
    objective, batch = FIXTURE.relay_training_fixture()
    snapshots = []
    for capture_enabled in (False, True):
        current = copy.deepcopy(objective)
        optimizer = V2Optimizer(current.student, 0.001, 0)
        runtime = SimpleNamespace(objective=current, optimizer=optimizer, data=SimpleNamespace())
        capture = PROFILE.ProfileCapture(runtime, tmp_path, 0)
        torch.manual_seed(311)
        if capture_enabled:
            capture.start(cuda=False)
        terms = optimizer_step(
            current, optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False
        )
        if capture_enabled:
            summary = capture.close()
            assert summary["summary"]["cuda_activity_present"] is False
            assert "gradpert/student/graph" in summary["summary"]["regions"]
            assert (tmp_path / "rank-0-trace.json").is_file()
        assert "forward" not in vars(current.student.graph)
        assert "step" not in vars(optimizer)
        snapshots.append(
            (terms, current.state_dict(), optimizer.state_dict(), torch.get_rng_state())
        )
    FIXTURE.assert_tree_close(snapshots[0], snapshots[1])


def test_cuda_mirrored_annotations_are_not_cpu_windows_or_cpu_regions():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "gradpert/capture_window",
                "ts": 0,
                "dur": 100,
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "gradpert/capture_window",
                "ts": 10,
                "dur": 80,
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "gradpert/student/graph",
                "ts": 5,
                "dur": 50,
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "gradpert/student/graph",
                "ts": 10,
                "dur": 40,
            },
            {"ph": "X", "cat": "kernel", "ts": 10, "dur": 40},
        ]
    }
    summary = PROFILE.trace_summary(trace)
    assert summary["window_us"] == 100
    assert summary["gpu_busy_union_us"] == 40
    assert summary["regions"]["gradpert/student/graph"] == {"calls": 1, "cpu_inclusive_us": 50}


def test_region_cleanup_on_failure():
    objective, _ = FIXTURE.relay_training_fixture()
    optimizer = V2Optimizer(objective.student, 0.001, 0)
    runtime = SimpleNamespace(objective=objective, optimizer=optimizer, data=SimpleNamespace())
    original = torch.autograd.backward
    with pytest.raises(RuntimeError, match="injected"), PROFILE.diagnostic_regions(runtime):
        raise RuntimeError("injected")
    assert torch.autograd.backward is original
    assert "forward" not in vars(objective.student.graph)
