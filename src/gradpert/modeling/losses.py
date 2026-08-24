"""Centered graph-view consistency, masked-node, and spread objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ConsistencyLossBreakdown:
    condition: Tensor
    masked_node: Tensor
    spread: Tensor
    spread_available: bool

    @property
    def ssl(self) -> Tensor:
        return self.condition + self.masked_node + 0.1 * self.spread


def centered_teacher_probabilities(
    logits: Tensor,
    center: Tensor,
    *,
    temperature: float = 0.04,
) -> Tensor:
    if temperature != 0.04:
        raise ValueError("teacher temperature is frozen at 0.04")
    return F.softmax((logits - center) / temperature, dim=-1).detach()


def _soft_target_cross_entropy(
    student_logits: Tensor,
    teacher_probabilities: Tensor,
    *,
    student_temperature: float = 0.1,
) -> Tensor:
    if student_temperature != 0.1:
        raise ValueError("student temperature is frozen at 0.1")
    log_probabilities = F.log_softmax(student_logits / student_temperature, dim=-1)
    return -(teacher_probabilities * log_probabilities).sum(dim=-1)


def condition_consistency_loss(
    *,
    student_view_logits: tuple[Tensor, ...],
    teacher_global_logits: tuple[Tensor, Tensor],
    center: Tensor,
) -> Tensor:
    """Average the exact 9+9 cross-view terms."""

    if len(student_view_logits) != 10:
        raise ValueError("condition consistency requires two globals plus eight locals")
    terms: list[Tensor] = []
    for teacher_index, teacher_logits in enumerate(teacher_global_logits):
        targets = centered_teacher_probabilities(teacher_logits, center)
        for student_index, student_logits in enumerate(student_view_logits):
            if student_index == teacher_index:
                continue
            terms.append(_soft_target_cross_entropy(student_logits, targets).mean())
    if len(terms) != 18:
        raise AssertionError("condition consistency pairing must contain exactly 18 terms")
    return torch.stack(terms).mean()


def masked_node_consistency_loss(
    *,
    student_logits: Tensor,
    teacher_logits: Tensor,
    center: Tensor,
) -> Tensor:
    """Average over nodes masked in the single batch-shared global view."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher masked-node shapes differ")
    if student_logits.ndim != 2:
        raise ValueError("masked-node logits must be [masked_nodes, prototypes]")
    if student_logits.shape[0] == 0:
        return student_logits.sum() * 0
    targets = centered_teacher_probabilities(teacher_logits, center)
    return _soft_target_cross_entropy(student_logits, targets).mean()


def embedding_spread_loss(unique_condition_states: Tensor) -> tuple[Tensor, bool]:
    if unique_condition_states.ndim != 2:
        raise ValueError("spread input must be [unique_conditions, latent_dim]")
    if unique_condition_states.shape[0] < 2:
        return unique_condition_states.sum() * 0, False
    normalized = F.normalize(unique_condition_states, p=2, dim=-1)
    similarities = normalized @ normalized.T
    similarities.fill_diagonal_(-torch.inf)
    nearest_indices = similarities.argmax(dim=1)
    nearest = normalized.index_select(0, nearest_indices)
    distances = torch.linalg.vector_norm(normalized - nearest, dim=-1)
    return -torch.log(distances.clamp_min(1e-8)).mean(), True
