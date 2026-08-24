"""Teacher EMA, centers, and frozen step-level schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class CenterState:
    condition: Tensor
    masked_node: Tensor

    @classmethod
    def zeros(cls, *, prototype_count: int, device: torch.device) -> CenterState:
        if prototype_count <= 0:
            raise ValueError("prototype_count must be positive")
        return cls(
            condition=torch.zeros(1, prototype_count, device=device),
            masked_node=torch.zeros(1, prototype_count, device=device),
        )


def initialize_teacher_from_student(student: nn.Module, teacher: nn.Module) -> None:
    teacher.load_state_dict(student.state_dict(), strict=True)
    teacher.requires_grad_(False)


@torch.no_grad()
def update_teacher_ema(student: nn.Module, teacher: nn.Module, *, momentum: float) -> None:
    if not 0 <= momentum <= 1:
        raise ValueError("teacher momentum must be in [0, 1]")
    student_parameters = dict(student.named_parameters())
    teacher_parameters = dict(teacher.named_parameters())
    if student_parameters.keys() != teacher_parameters.keys():
        raise ValueError("student and teacher parameter structures differ")
    for name, teacher_parameter in teacher_parameters.items():
        teacher_parameter.mul_(momentum).add_(student_parameters[name].detach(), alpha=1 - momentum)
    student_buffers = dict(student.named_buffers())
    teacher_buffers = dict(teacher.named_buffers())
    if student_buffers.keys() != teacher_buffers.keys():
        raise ValueError("student and teacher buffer structures differ")
    for name, teacher_buffer in teacher_buffers.items():
        teacher_buffer.copy_(student_buffers[name])


@torch.no_grad()
def update_center(center: Tensor, teacher_logits: Tensor, *, momentum: float = 0.9) -> None:
    if momentum != 0.9:
        raise ValueError("center momentum is frozen at 0.9")
    if teacher_logits.ndim != 2 or center.shape != (1, teacher_logits.shape[-1]):
        raise ValueError("center/logit shape mismatch")
    batch_center = teacher_logits.detach().mean(dim=0, keepdim=True)
    center.mul_(momentum).add_(batch_center, alpha=1 - momentum)


def cosine_teacher_momentum(
    *,
    global_step: int,
    total_steps: int,
    start: float = 0.996,
) -> float:
    if start != 0.996:
        raise ValueError("teacher momentum start is frozen at 0.996")
    if total_steps <= 0 or not 0 <= global_step <= total_steps:
        raise ValueError("global_step must be within the positive schedule")
    progress = global_step / total_steps
    return 1.0 - (1.0 - start) * (math.cos(math.pi * progress) + 1.0) / 2.0
