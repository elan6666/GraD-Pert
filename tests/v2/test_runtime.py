from types import SimpleNamespace

import numpy as np
import torch

from gradpert.training.v2.runtime import Runtime


def test_condition_cap_is_bounded_by_actual_probe_batch():
    calls = []

    def steps_per_epoch(**kwargs):
        calls.append(kwargs)
        return 7

    runtime = Runtime(
        data=SimpleNamespace(steps_per_epoch=steps_per_epoch),
        options=SimpleNamespace(max_conditions=8),
        index=None,
        objective=None,
        optimizer=None,
        generator=None,
        device=None,
        batch_size=2,
        identity={},
    )
    assert runtime.steps_per_epoch == 7
    assert runtime.steps_per_epoch == 7
    assert calls == [{"batch_size": 2, "max_unique_conditions": 2}]


def test_evaluation_can_load_distributed_training_but_cannot_train():
    import pytest

    from gradpert.training.v2.runtime import validate_world

    validate_world(2, 1, "evaluation")
    validate_world(2, 2, "training")
    with pytest.raises(ValueError, match="world size"):
        validate_world(2, 1, "training")
    with pytest.raises(ValueError, match="one process"):
        validate_world(2, 2, "evaluation")
    runtime = Runtime(
        data=None,
        options=None,
        index=None,
        objective=None,
        optimizer=None,
        generator=None,
        device=None,
        batch_size=128,
        identity={},
        purpose="evaluation",
    )
    with pytest.raises(RuntimeError, match="cannot enter training"):
        next(runtime.batches(1))
    with pytest.raises(RuntimeError, match="cannot enter training"):
        _ = runtime.steps_per_epoch


def test_validation_views_are_fixed_and_do_not_consume_training_rng(monkeypatch):
    from gradpert.training.v2 import runtime as runtime_module

    raw = SimpleNamespace(
        perturbed_row_ids=("val-a", "val-b"),
        control_row_ids=("ctrl-1", "ctrl-2"),
        condition_ids=("A", "B"),
    )
    data = SimpleNamespace(iter_validation_epoch=lambda **kwargs: iter((raw,)))

    def assemble(_raw, _index, _options, generator, **kwargs):
        return float(generator.random())

    monkeypatch.setattr(runtime_module, "assemble_batch", assemble)
    training_rng = np.random.default_rng(7)
    runtime = Runtime(
        data=data,
        options=SimpleNamespace(max_conditions=0),
        index=None,
        objective=None,
        optimizer=None,
        generator=training_rng,
        device=torch.device("cpu"),
        batch_size=2,
        identity={"run_seed": 7},
    )
    assert list(runtime.validation_batches(batch_size=2)) == list(
        runtime.validation_batches(batch_size=2)
    )
    assert training_rng.random() == np.random.default_rng(7).random()
