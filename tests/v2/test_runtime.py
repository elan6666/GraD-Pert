from types import SimpleNamespace

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
