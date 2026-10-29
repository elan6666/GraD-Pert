"""CUDA-labeled policy tests; no CUDA execution or scientific calibration."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Match the native script entrypoint's import directory when collected alone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/v2"))

from scripts.v2.calibrate_response_anchor import (
    _calibration_failure_payload,
    compare_head_audit_gradients,
    head_route_deterministic_context,
)
from scripts.v2.run_population_calibration import require_deterministic_route_audit


@pytest.mark.parametrize("enabled,warn", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("fail", [False, True])
def test_strict_context_restores(monkeypatch, enabled, warn, fail):
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(enabled, warn_only=warn)
    try:
        error = RuntimeError("first CUDA operation unavailable")
        try:
            with head_route_deterministic_context(SimpleNamespace(type="cuda")) as metadata:
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                if fail:
                    raise error
        except RuntimeError as caught:
            assert caught is error
            payload = _calibration_failure_payload(caught, "synthetic")
            assert (
                "first CUDA operation unavailable"
                in payload["population_head_route_deterministic_audit"]["failure"][
                    "original_traceback"
                ]
            )
        assert metadata["before"] == metadata["restored"] == {"enabled": enabled, "warn_only": warn}
        assert metadata["settings_restored"]
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])


def test_cpu_no_policy_mutation(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with head_route_deterministic_context(SimpleNamespace(type="cpu")) as metadata:
        assert metadata["inside"] == metadata["before"]
    assert metadata["settings_restored"]


@pytest.mark.parametrize("cbc", ["", ":1:1", "4096:8"])
def test_invalid_workspace_fails_restored(monkeypatch, cbc):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", cbc)
    with (
        pytest.raises(ValueError) as caught,
        head_route_deterministic_context(SimpleNamespace(type="cuda")),
    ):
        pytest.fail("invalid workspace entered audit")
    assert caught.value.population_head_route_deterministic_audit["settings_restored"]


def valid_audit():
    report = compare_head_audit_gradients((torch.ones(1),), (torch.ones(1),), atol=2e-6)
    return dict(
        deterministic_audit=dict(
            scope="route_audit_only_not_norms_or_fit",
            cuda=True,
            cublas_workspace_config=":16:8",
            before=dict(enabled=False, warn_only=True),
            inside=dict(enabled=True, warn_only=False),
            restored=dict(enabled=False, warn_only=True),
            settings_restored=True,
        ),
        group_replays=5,
        additional_audit_replays=2,
        replay_norms_included_in_first_eight=False,
        repeatability_comparisons={
            r: {s: copy.deepcopy(report) for s in ("input", "response")}
            for r in ("true_true", "false_false")
        },
    )


def test_guardian_strict_metadata_and_original_gate():
    audit = valid_audit()
    require_deterministic_route_audit(audit)
    for route in ("true_true", "false_false"):
        bad = copy.deepcopy(audit)
        bad["repeatability_comparisons"][route]["response"]["max_abs"] = 2.01e-6
        with pytest.raises(ValueError):
            require_deterministic_route_audit(bad)
    for field, value in (
        ("settings_restored", False),
        ("cuda", False),
        ("inside", dict(enabled=True, warn_only=True)),
    ):
        bad = copy.deepcopy(audit)
        bad["deterministic_audit"][field] = value
        with pytest.raises(ValueError):
            require_deterministic_route_audit(bad)
    with pytest.raises(ValueError):
        compare_head_audit_gradients((torch.tensor([3e-6]),), (torch.zeros(1),), atol=2e-6)


def test_wrapper_covers_entire_native_audit_and_restores_before_return(monkeypatch):
    import scripts.v2.calibrate_response_anchor as module

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def native(runtime, *args, **kwargs):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        return {"complete": True}

    monkeypatch.setattr(module, "_audit_population_head_route", native)
    result = module.audit_population_head_route(
        SimpleNamespace(device=SimpleNamespace(type="cuda"))
    )
    assert result["deterministic_audit"]["settings_restored"]
    assert result["deterministic_audit"]["restored"]["enabled"] == (
        torch.are_deterministic_algorithms_enabled()
    )
