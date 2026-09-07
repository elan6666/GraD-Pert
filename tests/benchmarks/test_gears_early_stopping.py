from copy import deepcopy

import pytest

from benchmarks.gears.early_stopping import train_with_patience


def compute_metrics(value):
    return {"mse_de": value}, {}


class OfficialShape:
    """Tiny stand-in for the audited official loop's control flow, not its model."""

    def __init__(self, values):
        self.values = values
        self.dataloader = {}
        self.model = [0]
        self.optimizer_creations = 0
        self.optimizer_steps = 0

    def train(self, epochs=20, lr=0.001, weight_decay=0.0005):
        self.optimizer_creations += 1
        best_model = deepcopy(self.model)
        min_val = float("inf")
        for epoch in range(epochs):
            self.optimizer_steps += 1
            self.model[0] = epoch
            compute_metrics(0)
            val, _ = compute_metrics(self.values[epoch])
            if val["mse_de"] < min_val:
                min_val = val["mse_de"]
                best_model = deepcopy(self.model)
        self.best_model = best_model


def test_patience_preserves_optimizer_and_final_best_assignment():
    model = OfficialShape([3, 2, 2, 2, 2])
    original_metric = compute_metrics
    events = []
    result = train_with_patience(
        model, epochs=5, lr=0.001, weight_decay=0.0005, patience=2, on_epoch=events.append
    )
    assert len(result) == 4
    assert model.optimizer_creations == 1
    assert model.optimizer_steps == 4
    assert model.best_model == [1]
    assert compute_metrics is original_metric
    assert len(events) == 4


def test_nonfinite_validation_and_test_loader_fail_closed():
    model = OfficialShape([float("nan")])
    with pytest.raises(RuntimeError, match="nonfinite"):
        train_with_patience(
            model, epochs=1, lr=0.001, weight_decay=0.0005, patience=10, on_epoch=lambda _: None
        )
    model.dataloader["test_loader"] = object()
    with pytest.raises(ValueError, match="test loader"):
        train_with_patience(
            model, epochs=1, lr=0.001, weight_decay=0.0005, patience=10, on_epoch=lambda _: None
        )
