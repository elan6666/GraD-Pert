"""Resource caps affect admission only; truncated probes never count as tests."""

import pytest

from gradpert.execution.v2_checkpoint_eval import (
    diagnostic_conditions,
    evaluation_memory_fraction,
    execute_evaluation_plan,
)


@pytest.mark.parametrize("value", [0, -0.1, 1.1, True, "0.2", float("nan")])
def test_reject_invalid_allocation_budget(value):
    with pytest.raises(ValueError):
        evaluation_memory_fraction({"cuda_memory_fraction": value})


def test_default_has_no_cap_and_no_subset():
    assert evaluation_memory_fraction({}) is None
    assert evaluation_memory_fraction({"cuda_memory_fraction": 0.19}) == 0.19
    assert diagnostic_conditions({}, ("a", "b")) == ("a", "b")


def test_probe_is_explicit_and_cannot_write_scientific_receipt():
    plan = {"diagnostic_only": True, "diagnostic_condition_count": 1}
    assert diagnostic_conditions(plan, ("a", "b")) == ("a",)
    with pytest.raises(ValueError, match="COMPLETE"):
        execute_evaluation_plan(plan)
    with pytest.raises(ValueError):
        diagnostic_conditions({"diagnostic_condition_count": 1}, ("a", "b"))
