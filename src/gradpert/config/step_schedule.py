"""Native step schedules aligned to frozen reference arithmetic, without imports.

Reference: facebookresearch/dinov2 7764ea0f912e53c92e82eb78a2a1631e92725fc8,
utils/config.py apply_scaling_rules_to_cfg and utils/utils.py CosineScheduler.
The warmup fraction is a project adaptation, not an official dataset recipe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from gradpert.config.lr_schedule import EpochWarmupCosineRestarts


@dataclass(frozen=True)
class StepWarmupCosine:
    base_lr: float
    min_lr: float
    global_batch_size: int
    warmup_fraction: float
    teacher_start: float
    teacher_end: float

    @property
    def max_lr(self) -> float:
        return self.base_lr * math.sqrt(self.global_batch_size / 1024)

    def at_step(self, step: int, total_steps: int) -> dict[str, float]:
        if type(step) is not int or type(total_steps) is not int or step < 0 or total_steps < 1:
            raise ValueError("invalid schedule step budget")
        warmup = int(total_steps * self.warmup_fraction)
        if step >= total_steps:
            lr, momentum = self.min_lr, self.teacher_end
        else:
            if step < warmup:
                lr = self.max_lr * step / max(1, warmup - 1)
            else:
                progress = (step - warmup) / (total_steps - warmup)
                lr = (
                    self.min_lr
                    + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                )
            momentum = (
                self.teacher_end
                + (self.teacher_start - self.teacher_end)
                * (1 + math.cos(math.pi * step / total_steps))
                / 2
            )
        return {"learning_rate": lr, "teacher_momentum": momentum}


@dataclass(frozen=True)
class LRWarmupCosine:
    """R50 LR-only schedule; teacher EMA remains on the original engine path."""

    max_lr: float
    min_lr: float
    warmup_fraction: float

    def at_step(self, step: int, total_steps: int) -> dict[str, float]:
        if type(step) is not int or type(total_steps) is not int or step < 0 or total_steps < 1:
            raise ValueError("invalid schedule step budget")
        warmup = int(total_steps * self.warmup_fraction)
        if step >= total_steps:
            lr = self.min_lr
        elif step < warmup:
            lr = self.max_lr * step / max(1, warmup - 1)
        else:
            progress = (step - warmup) / (total_steps - warmup)
            lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        return {"learning_rate": lr}


@dataclass(frozen=True)
class EndpointLRWarmupCosine(LRWarmupCosine):
    """R50 schedule-only variant with an exact final-step LR endpoint."""

    def at_step(self, step: int, total_steps: int) -> dict[str, float]:
        if type(step) is not int or type(total_steps) is not int or step < 0:
            raise ValueError("invalid schedule step budget")
        warmup = int(total_steps * self.warmup_fraction)
        if warmup < 1 or total_steps - 1 <= warmup:
            raise ValueError("schedule requires positive warmup and decay intervals")
        if step >= total_steps - 1:
            lr = self.min_lr
        elif step < warmup:
            lr = self.max_lr * step / warmup
        else:
            progress = (step - warmup) / (total_steps - 1 - warmup)
            lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        return {"learning_rate": lr}


def load_training_schedule(
    value: Any,
) -> EpochWarmupCosineRestarts | StepWarmupCosine | LRWarmupCosine | None:
    if isinstance(value, dict) and value.get("name") in {
        "lr_warmup_cosine",
        "lr_warmup_cosine_endpoint",
    }:
        fields = {"max_lr", "min_lr", "warmup_fraction"}
        if set(value) != fields | {"name", "interval"} or value["interval"] != "step":
            raise ValueError("LR-only schedule requires complete explicit fields")
        if any(type(value[k]) not in (int, float) or not math.isfinite(value[k]) for k in fields):
            raise ValueError("LR-only schedule parameters must be finite")
        schedule_type = (
            EndpointLRWarmupCosine
            if value["name"] == "lr_warmup_cosine_endpoint"
            else LRWarmupCosine
        )
        result_lr = schedule_type(**{k: value[k] for k in fields})
        if not 0 < result_lr.min_lr <= result_lr.max_lr or not 0 < result_lr.warmup_fraction < 1:
            raise ValueError("invalid LR-only schedule bounds")
        return result_lr
    if not isinstance(value, dict) or value.get("name") != "step_warmup_cosine":
        return EpochWarmupCosineRestarts.from_config(value)
    fields = {
        "base_lr",
        "min_lr",
        "global_batch_size",
        "warmup_fraction",
        "teacher_start",
        "teacher_end",
    }
    if set(value) != fields | {"name", "interval", "scaling_rule"}:
        raise ValueError("step schedule requires complete explicit fields")
    if value["interval"] != "step" or value["scaling_rule"] != "sqrt_wrt_1024":
        raise ValueError("unsupported step schedule or LR scaling rule")
    if type(value["global_batch_size"]) is not int or value["global_batch_size"] < 2:
        raise ValueError("invalid global batch size")
    for key in fields - {"global_batch_size"}:
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]):
            raise ValueError("schedule parameters must be finite numbers")
    result = StepWarmupCosine(**{key: value[key] for key in fields})
    if not 0 < result.min_lr <= result.max_lr or not 0 < result.warmup_fraction < 1:
        raise ValueError("invalid LR or warmup bounds")
    if not 0 < result.teacher_start < result.teacher_end <= 1:
        raise ValueError("invalid teacher momentum bounds")
    return result
