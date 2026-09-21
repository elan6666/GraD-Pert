"""Shared batch execution for native training adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

Batch = TypeVar("Batch")


def execute_epoch(
    batches: Iterable[Batch],
    update: Callable[[Batch, int], None],
    *,
    maximum_steps: int | None = None,
) -> int:
    """Count completed updates; propagate failures without advancing past them.

    A sealed adapter can reject excess batches before an update. Legacy adapters
    may omit the bound and retain their existing end-of-epoch count validation.
    Logging, model state and recovery remain owned by the supplied update adapter.
    """
    if maximum_steps is not None and maximum_steps < 1:
        raise ValueError("epoch step bound must be positive")
    completed = 0
    for batch in batches:
        if maximum_steps is not None and completed >= maximum_steps:
            raise ValueError("epoch iterator exceeds sealed step budget")
        update(batch, completed)
        completed += 1
    return completed
