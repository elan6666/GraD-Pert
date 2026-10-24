"""A read-only metric candidate cannot escape the isolated engineering runner."""

import argparse
import runpy
from pathlib import Path

import pytest

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/v2/capacity_probe.py"))[
    "validate_metric_scalar_mode"
]
FIELDS = (
    "integration_only",
    "shared_queue",
    "profile_last_update",
    "profile_pt1_stages",
    "profile_memory",
    "profile_operator_table",
    "sync_phase_timing",
    "cpu_prefetch",
    "no_sequence_checkpoint",
    "fused_sinkhorn",
    "save_no_grad_sinkhorn",
    "fused_gram",
    "gram_io",
    "aten_kda",
    "reference_unused_ssl_decoder",
    "joint_validation_check",
)


def options(**overrides):
    values = dict.fromkeys(FIELDS, False)
    values.update(benchmark_only=False, metric_scalar_batch=False, record_update_state=False)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("flag", ["metric_scalar_batch", "record_update_state"])
@pytest.mark.parametrize("other", FIELDS)
def test_rejects_mixed_candidate_or_profile_even_in_benchmark(flag, other):
    with pytest.raises(ValueError, match="isolated unprofiled benchmark-only"):
        CHECK(options(benchmark_only=True, **{flag: True, other: True}))


@pytest.mark.parametrize("flag", ["metric_scalar_batch", "record_update_state"])
def test_rejects_formal_and_capacity_modes(flag):
    with pytest.raises(ValueError, match="benchmark-only"):
        CHECK(options(**{flag: True}))


def test_allows_base_candidate_and_parity_as_separate_bounded_diagnostics():
    for candidate, state in [(False, True), (True, False), (True, True)]:
        CHECK(
            options(benchmark_only=True, metric_scalar_batch=candidate, record_update_state=state)
        )


def test_default_does_not_change_existing_probe_modes():
    for other in FIELDS:
        CHECK(options(**{other: True}))


def test_parity_snapshot_includes_objective_buffers_and_unregistered_temperature_clock(monkeypatch):
    import types

    import torch

    functions = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/v2/capacity_probe.py")
    )
    capture = functions["update_parity_state"]
    monkeypatch.setitem(
        capture.__globals__, "validation_state", lambda runtime: {"student": "same"}
    )
    buffers = {"temperature_update_index": torch.tensor(2)}
    objective = types.SimpleNamespace(
        named_buffers=lambda: iter(buffers.items()), _temperature_updates=2
    )
    runtime = types.SimpleNamespace(objective=objective)
    before = capture(runtime)
    objective._temperature_updates = 3
    changed = capture(runtime)
    assert before["objective_buffers"] == changed["objective_buffers"]
    assert before["temperature_updates"] != changed["temperature_updates"]
    buffers["temperature_update_index"].add_(1)
    assert capture(runtime)["objective_buffers"] != changed["objective_buffers"]
