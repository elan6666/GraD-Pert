"""Explicit condition versus cell sampling of the same graph condition states."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from gradpert.modeling.losses import embedding_spread_loss


def cell_spread_pool(
    states: Tensor, condition_order: Sequence[str], cells: Sequence[str]
) -> Tensor:
    """Repeat condition states in real batch order, without noise or cell features."""
    if states.ndim != 2 or len(condition_order) != states.shape[0]:
        raise ValueError("condition state axis mismatch")
    if len(set(condition_order)) != len(condition_order):
        raise ValueError("condition state axis must be unique")
    lookup = {condition: i for i, condition in enumerate(condition_order)}
    if any(condition not in lookup for condition in cells):
        raise ValueError("batch condition missing from graph state axis")
    indices = torch.tensor([lookup[c] for c in cells], device=states.device, dtype=torch.long)
    return states.index_select(0, indices)


def diagnose_cell_spread(
    states: Tensor, condition_order: Sequence[str], cells: Sequence[str]
) -> dict[str, Any]:
    """Diagnostic-only probe: gradient norm is w.r.t. unique p, not model weights.

    A detached copy avoids touching training gradients/RNG. Loss is unweighted;
    the configured spread weight is applied by the unchanged training objective.
    """
    with torch.enable_grad():
        unique = states.detach().clone().requires_grad_(True)
        pool = cell_spread_pool(unique, condition_order, cells)
        loss, available = embedding_spread_loss(pool)
        gradient = torch.autograd.grad(loss, unique)[0]
        result: dict[str, Any] = {
            "cell_count": len(cells),
            "unique_condition_count": len(condition_order),
            "available": available,
            "unweighted_loss": float(loss.detach()),
            "unweighted_unique_p_gradient_norm": float(gradient.norm().detach()),
            "same_condition_nearest_fraction": None,
            "zero_distance_nearest_fraction": None,
            "repeated_condition_rows": len(cells) - len(set(cells)),
            "repeated_p_exactly_equal": True,
        }
        first: dict[str, int] = {}
        for i, condition in enumerate(cells):
            if condition in first:
                result["repeated_p_exactly_equal"] &= torch.equal(pool[i], pool[first[condition]])
            else:
                first[condition] = i
        if available:
            normalized = F.normalize(pool, p=2, dim=-1)
            similarities = normalized @ normalized.T
            similarities.fill_diagonal_(-torch.inf)
            nearest = similarities.argmax(dim=1)
            distance = torch.linalg.vector_norm(normalized - normalized[nearest], dim=-1)
            result["zero_distance_nearest_fraction"] = float((distance == 0).float().mean())
            result["same_condition_nearest_fraction"] = sum(
                cells[i] == cells[j] for i, j in enumerate(nearest.tolist())
            ) / len(cells)
        return result
