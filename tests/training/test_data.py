from __future__ import annotations

from collections import Counter

import pytest

from gradpert.training.data import condition_limited_epoch_batches


def test_condition_limited_batches_are_deterministic_full_and_bounded() -> None:
    conditions = tuple(
        condition
        for condition, count in (("A", 81), ("B", 73), ("C", 61), ("D", 49), ("E", 37))
        for _ in range(count)
    )
    first = condition_limited_epoch_batches(
        condition_ids=conditions,
        run_seed=7,
        epoch=2,
        batch_size=64,
        max_unique_conditions=3,
    )
    second = condition_limited_epoch_batches(
        condition_ids=conditions,
        run_seed=7,
        epoch=2,
        batch_size=64,
        max_unique_conditions=3,
    )
    assert first == second
    flattened = [index for batch in first for index in batch]
    assert len(flattened) == len(set(flattened))
    assert len(conditions) - len(flattened) in {0, 1}
    assert all(2 <= len(batch) <= 64 for batch in first)
    assert all(len({conditions[index] for index in batch}) <= 3 for batch in first)
    assert Counter(conditions[index] for index in flattened) <= Counter(conditions)


def test_condition_limited_batches_change_with_epoch() -> None:
    conditions = ("A",) * 80 + ("B",) * 80
    epoch_zero = condition_limited_epoch_batches(
        condition_ids=conditions,
        run_seed=11,
        epoch=0,
    )
    epoch_one = condition_limited_epoch_batches(
        condition_ids=conditions,
        run_seed=11,
        epoch=1,
    )
    assert epoch_zero != epoch_one
    assert len(epoch_zero) == len(epoch_one)


@pytest.mark.parametrize(
    ("batch_size", "max_unique"),
    [(1, 1), (64, 0), (64, 65)],
)
def test_condition_limited_batches_reject_invalid_shape(
    batch_size: int,
    max_unique: int,
) -> None:
    with pytest.raises(ValueError):
        condition_limited_epoch_batches(
            condition_ids=("A", "B"),
            run_seed=1,
            epoch=0,
            batch_size=batch_size,
            max_unique_conditions=max_unique,
        )
