from __future__ import annotations

import pytest

from gradpert.training.controls import TrainingControlPairer
from gradpert.training.selection import EarlyStoppingState


def test_training_control_pairing_is_epoch_named_and_context_restricted() -> None:
    pairer = TrainingControlPairer(run_seed=4)
    kwargs = {
        "perturbed_row_ids": ["pert-1", "pert-2"],
        "context_ids": ["K562::b1", "K562::b2"],
        "control_pools": {
            "K562::b1": ["c1", "c2", "c3"],
            "K562::b2": ["c4", "c5", "c6"],
        },
    }
    first = pairer.pair_epoch(epoch=7, **kwargs)
    repeated = pairer.pair_epoch(epoch=7, **kwargs)
    assert first == repeated
    assert first.control_row_ids[0] in kwargs["control_pools"]["K562::b1"]
    assert first.control_row_ids[1] in kwargs["control_pools"]["K562::b2"]
    assert pairer.pair_epoch(epoch=8, **kwargs).epoch == 8


def test_training_control_pairing_fails_without_compatible_control() -> None:
    with pytest.raises(ValueError, match="no valid compatible"):
        TrainingControlPairer(run_seed=1).pair_epoch(
            epoch=0,
            perturbed_row_ids=["pert-1"],
            context_ids=["missing"],
            control_pools={},
        )


def test_early_stopping_requires_ten_consecutive_non_improvements() -> None:
    state = EarlyStoppingState()
    improved, stopped = state.update(epoch=0, validation_metric=0.1)
    assert improved and not stopped
    for epoch in range(1, 10):
        improved, stopped = state.update(epoch=epoch, validation_metric=0.1)
        assert not improved and not stopped
    improved, stopped = state.update(epoch=10, validation_metric=0.1)
    assert not improved and stopped


def test_strict_improvement_resets_patience() -> None:
    state = EarlyStoppingState()
    state.update(epoch=0, validation_metric=0.1)
    for epoch in range(1, 6):
        state.update(epoch=epoch, validation_metric=0.09)
    improved, stopped = state.update(epoch=6, validation_metric=0.10001)
    assert improved and not stopped
    assert state.consecutive_non_improvements == 0
