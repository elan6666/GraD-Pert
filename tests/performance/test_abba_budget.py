from types import SimpleNamespace

import pytest

from scripts.performance.profile_native_a0 import ProfileGateError, _step_budget


@pytest.mark.parametrize(
    "phase,budget", [("capacity", (3, 3)), ("profile", (2, 3)), ("timing", (2, 10))]
)
def test_legacy_budget_unchanged(phase, budget):
    assert _step_budget(SimpleNamespace(phase=phase)) == budget


def test_abba_budget_is_exactly_25():
    args = SimpleNamespace(phase="timing", timing_protocol="abba_5_20", capture_exact_state=False)
    assert _step_budget(args) == (5, 20)


@pytest.mark.parametrize(
    "phase,capture", [("capacity", False), ("profile", False), ("timing", True)]
)
def test_abba_rejects_other_work(phase, capture):
    with pytest.raises(ProfileGateError):
        _step_budget(
            SimpleNamespace(phase=phase, timing_protocol="abba_5_20", capture_exact_state=capture)
        )
