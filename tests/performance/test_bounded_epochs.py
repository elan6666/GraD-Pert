"""Stopping diagnostics must not materialize an extra batch or alter schedule."""

import pytest

from scripts.performance.bounded_epochs import (
    BoundedEpochComplete,
    bounded_epoch_factory,
    state_content_sha256,
    validate_boundary_records,
)


def test_checkpoint_hash_covers_tensor_rng_and_structure() -> None:
    import numpy as np
    import torch

    original = {"tensor": torch.tensor([1.0, 2.0]), "rng": np.array([1, 3], dtype=np.uint32)}
    copied = {"rng": original["rng"].copy(), "tensor": original["tensor"].clone()}
    assert state_content_sha256(original) == state_content_sha256(copied)
    copied["tensor"][0] = 0
    assert state_content_sha256(original) != state_content_sha256(copied)
    assert state_content_sha256([1, 2]) != state_content_sha256((1, 2))
    assert state_content_sha256(torch.tensor([1.0])) != state_content_sha256(torch.tensor([[1.0]]))


@pytest.mark.parametrize("fault", [None, "schedule", "steps", "validation", "last", "test"])
def test_boundary_records_fail_closed(fault) -> None:
    progress = dict(completed_epochs=3, global_step=6, test_evaluations=0)
    args = dict(
        epoch=3,
        steps_per_epoch=2,
        schedule_steps=100,
        progress=progress,
        steps=[dict(epoch=i // 2, global_step=i) for i in range(6)],
        validations=[dict(epoch=i, global_step=(i + 1) * 2) for i in range(3)],
        last_progress=dict(progress),
        best_progress=dict(progress),
    )
    if fault == "schedule":
        args["schedule_steps"] = 6
    if fault == "steps":
        args["steps"][2]["global_step"] = 1
    if fault == "validation":
        args["validations"].pop()
    if fault == "last":
        args["last_progress"]["completed_epochs"] = 2
    if fault == "test":
        args["best_progress"]["test_evaluations"] = 1
    if fault:
        with pytest.raises(ValueError):
            validate_boundary_records(**args)
    else:
        validate_boundary_records(**args)


def test_three_epoch_boundary_never_requests_fourth_epoch() -> None:
    events = []

    def data(epoch):
        events.append(("data", epoch))
        return (epoch,)

    bounded = bounded_epoch_factory(
        data, stop_before_epoch=3, on_boundary=lambda e: events.append(("boundary", e))
    )
    for epoch in range(3):
        assert tuple(bounded(epoch)) == (epoch,)
        events.append(("validated_saved", epoch))
    with pytest.raises(BoundedEpochComplete):
        bounded(3)
    assert events[-2:] == [("validated_saved", 2), ("boundary", 3)]
    assert ("data", 3) not in events


def test_resume_starts_at_supplied_epoch_without_replaying_data() -> None:
    seen = []
    bounded = bounded_epoch_factory(
        lambda e: seen.append(e) or [e], stop_before_epoch=3, on_boundary=lambda _: None
    )
    assert list(bounded(1)) == [1]
    assert list(bounded(2)) == [2]
    with pytest.raises(BoundedEpochComplete):
        bounded(3)
    assert seen == [1, 2]


def test_boundary_validation_error_is_not_relabelled_as_success() -> None:
    def reject(epoch):
        raise ValueError("checkpoint missing")

    bounded = bounded_epoch_factory(lambda _: (), stop_before_epoch=1, on_boundary=reject)
    with pytest.raises(ValueError, match="checkpoint missing"):
        bounded(1)


@pytest.mark.parametrize("stop", [0, 2, 4, 50])
def test_unsupported_boundary_fails_closed(stop) -> None:
    with pytest.raises(ValueError):
        bounded_epoch_factory(lambda _: (), stop_before_epoch=stop, on_boundary=lambda _: None)
