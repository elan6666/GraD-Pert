import runpy
import sys
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/v2"
    monkeypatch.syspath_prepend(str(scripts))
    # run_path loads the sibling streaming parser through the temporary path.
    result = runpy.run_path(str(scripts / "trace_attribution.py"))
    yield result
    sys.modules.pop("trace_costs", None)


def test_attribution_uses_thread_interval_and_correlation_not_event_order(module):
    rows = [
        dict(cat="cpu_op", ph="X", name="aten::logsumexp", pid=1, tid=2, ts=10, dur=5),
        dict(cat="cpu_op", ph="X", name="aten::logsumexp", pid=1, tid=2, ts=11, dur=2),
        # Same label on a GPU annotation must not become a CPU scope.
        dict(
            cat="gpu_user_annotation", ph="X", name="aten::logsumexp", pid=0, tid=2, ts=0, dur=100
        ),
        dict(cat="kernel", ph="X", name="x", pid=0, tid=7, ts=30, dur=8, args={"correlation": 3}),
        dict(cat="kernel", ph="X", name="x", pid=0, tid=7, ts=40, dur=50, args={"correlation": 4}),
        dict(
            cat="cuda_runtime",
            ph="X",
            name="cudaLaunchKernel",
            pid=1,
            tid=2,
            ts=12,
            dur=1,
            args={"correlation": 3},
        ),
        dict(
            cat="cuda_runtime",
            ph="X",
            name="cudaLaunchKernel",
            pid=1,
            tid=3,
            ts=12,
            dur=1,
            args={"correlation": 4},
        ),
    ]
    result = module["attribute"](lambda: iter(rows))
    assert result["total_kernel_count"] == 2
    assert result["scopes"]["aten::logsumexp"] == {
        "cpu_scope_union_us_summed_across_threads": 5,
        "launch_api_duration_sum_us": 1,
        "gpu_kernel_count": 1,
        "gpu_kernel_duration_sum_us": 8,
    }


def test_multiple_cpu_processes_rejected(module):
    rows = [dict(cat="cpu_op", pid=i) for i in (1, 2)]
    with pytest.raises(ValueError, match="one CPU process"):
        module["attribute"](lambda: iter(rows))
