"""Validation-only early stopping with strict-improvement semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    patience: int = 10
    min_delta: float = 0.0
    best_metric: float | None = None
    best_epoch: int | None = None
    consecutive_non_improvements: int = 0

    def __post_init__(self) -> None:
        if self.patience != 10 or self.min_delta != 0.0:
            raise ValueError("v1 early stopping is frozen to patience=10 and min_delta=0")

    def update(self, *, epoch: int, validation_metric: float) -> tuple[bool, bool]:
        """Return ``(improved, should_stop)`` after one validation epoch."""

        if epoch < 0 or not math.isfinite(validation_metric):
            raise ValueError("validation epoch/metric must be nonnegative and finite")
        improved = self.best_metric is None or validation_metric > self.best_metric + self.min_delta
        if improved:
            self.best_metric = validation_metric
            self.best_epoch = epoch
            self.consecutive_non_improvements = 0
        else:
            self.consecutive_non_improvements += 1
        return improved, self.consecutive_non_improvements >= self.patience
