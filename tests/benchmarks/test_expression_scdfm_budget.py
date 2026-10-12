"""scDFM equivalent-epoch budgets do not pretend to traverse every training row."""

from types import SimpleNamespace

import pytest

from benchmarks.expression_holdout.scdfm import (
    OFFICIAL_RECIPE,
    budget_progress,
    resolve_training_budget,
)


def prepared(n):
    return SimpleNamespace(train_row_ids=tuple(f"target-{i}" for i in range(n)))


def test_four_epochs_resolve_ceil_real_train_perturbed_rows_and_official_batch():
    training, budget = resolve_training_budget(prepared(97), {"epochs": 4})
    assert training["batch_size"] == 48
    assert "steps" not in training
    assert budget["n_train_perturbed_rows"] == 97
    assert budget["steps_per_epoch"] == 3
    assert budget["total_optimizer_updates"] == budget["cosine_t_max"] == 12
    assert budget["epochs_requested"] == 4
    assert budget["official_optimizer_updates"] == 200000
    assert budget["budget_deviation"]
    assert "not exactly one traversal" in budget["epoch_semantics"]
    assert training["lr"] == OFFICIAL_RECIPE["lr"]
    assert training["eta_min"] == OFFICIAL_RECIPE["eta_min"]
    assert training["gamma"] == OFFICIAL_RECIPE["gamma"]


def test_ten_update_preflight_receipt_does_not_claim_four_epochs_complete():
    _, budget = resolve_training_budget(prepared(481), {"epochs": 4})
    receipt = budget_progress(budget, completed_updates=10)
    assert receipt["steps_per_epoch"] == 11
    assert receipt["epochs_completed"] == 0
    assert receipt["epoch_equivalent_progress"] == pytest.approx(10 / 11)
    assert receipt["completed_updates"] == 10
    assert receipt["sampled_target_cell_draws"] == 480
    assert receipt["training_complete"] is False
    assert receipt["cosine_t_max"] == 44
    assert receipt["epochs_requested"] == 4


def test_completed_receipt_and_partial_fourth_epoch_are_distinct():
    _, budget = resolve_training_budget(prepared(97), {"epochs": 4})
    partial = budget_progress(budget, 10)
    complete = budget_progress(budget, 12)
    assert partial["epochs_completed"] == 3
    assert partial["training_complete"] is False
    assert complete["epochs_completed"] == 4
    assert complete["epoch_equivalent_progress"] == 4
    assert complete["training_complete"] is True
    assert complete["sampled_target_cell_draws"] == 576


def test_existing_official_and_explicit_steps_budgets_are_preserved():
    default_training, default = resolve_training_budget(prepared(100), {})
    assert default_training["steps"] == default["cosine_t_max"] == 200000
    assert default["epochs_requested"] is None and default["budget_deviation"] is None
    training, custom = resolve_training_budget(prepared(100), {"steps": 22})
    assert training["steps"] == custom["total_optimizer_updates"] == 22
    assert custom["cosine_t_max"] == 22
    assert custom["budget_mode"] == "steps"
    assert custom["budget_deviation"]


def test_cpu_diagnostic_batch_uses_actual_batch_size_without_architecture_changes():
    training, budget = resolve_training_budget(prepared(24), {"epochs": 4, "batch_size": 4})
    assert training["batch_size"] == 4
    assert budget["steps_per_epoch"] == 6 and budget["total_optimizer_updates"] == 24
    assert budget_progress(budget, 10)["epochs_completed"] == 1


@pytest.mark.parametrize(
    "training",
    [
        {"epochs": 4, "steps": 200000},
        {"epochs": 0},
        {"epochs": -1},
        {"epochs": 4.0},
        {"epochs": True},
        {"steps": 0},
        {"steps": False},
        {"batch_size": 0},
    ],
)
def test_invalid_or_ambiguous_budgets_fail_closed(training):
    with pytest.raises(ValueError):
        resolve_training_budget(prepared(100), training)


def test_duplicate_or_empty_train_rows_and_impossible_progress_fail_closed():
    with pytest.raises(ValueError, match="distinct"):
        resolve_training_budget(prepared(0), {"epochs": 4})
    with pytest.raises(ValueError, match="distinct"):
        resolve_training_budget(SimpleNamespace(train_row_ids=("t", "t")), {"epochs": 4})
    _, budget = resolve_training_budget(prepared(100), {"epochs": 4})
    with pytest.raises(ValueError, match="exceed"):
        budget_progress(budget, 13)
