"""Epoch-indexed warmup/cosine restart schedule with explicit provenance.

Matches the sequential step() path audited in scLong utils.py at
41b72021540918e4386c4f2264351d7a6cbeed99. No upstream runtime import.
The completed epoch count is the entire schedule state; checkpoint identity
binds its configuration, and optimizer state retains the next epoch's rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpochWarmupCosineRestarts:
    first_cycle_steps: int
    cycle_mult: int
    max_lr: float
    min_lr: float
    warmup_steps: int
    gamma: float

    @classmethod
    def from_config(cls, value: Any) -> EpochWarmupCosineRestarts | None:
        if value == "none":
            return None
        required = {
            "name",
            "interval",
            "first_cycle_steps",
            "cycle_mult",
            "max_lr",
            "min_lr",
            "warmup_steps",
            "gamma",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("native scheduler requires an explicit complete configuration")
        if value["name"] != "warmup_cosine_restarts" or value["interval"] != "epoch":
            raise ValueError("native warmup/cosine restarts must use epoch intervals")
        for key in ("first_cycle_steps", "cycle_mult", "warmup_steps"):
            if type(value[key]) is not int:
                raise ValueError("scheduler interval lengths must be integers")
        result = cls(**{k: value[k] for k in required - {"name", "interval"}})
        if not 0 < result.warmup_steps < result.first_cycle_steps or result.cycle_mult < 1:
            raise ValueError("invalid scheduler cycle lengths")
        if not all(math.isfinite(x) for x in (result.min_lr, result.max_lr, result.gamma)):
            raise ValueError("scheduler rates must be finite")
        if not 0 < result.min_lr <= result.max_lr or not 0 < result.gamma <= 1:
            raise ValueError("invalid scheduler rate bounds")
        return result

    def at_epoch(self, epoch: int) -> dict[str, int | float]:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("scheduler epoch must be a nonnegative integer")
        position, length, cycle = epoch, self.first_cycle_steps, 0
        while position >= length:
            position -= length
            length = (length - self.warmup_steps) * self.cycle_mult + self.warmup_steps
            cycle += 1
        peak = self.max_lr * self.gamma**cycle
        if position < self.warmup_steps:
            lr = (peak - self.min_lr) * position / self.warmup_steps + self.min_lr
        else:
            lr = (
                self.min_lr
                + (peak - self.min_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (position - self.warmup_steps) / (length - self.warmup_steps)
                    )
                )
                / 2
            )
        return {
            "epoch": epoch,
            "cycle": cycle,
            "cycle_length": length,
            "position": position,
            "peak_lr": peak,
            "learning_rate": lr,
        }
