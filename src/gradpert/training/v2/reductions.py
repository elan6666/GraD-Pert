"""Global-population weights and differentiable cross-rank neighborhoods."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import functional as F


def population_weights(conditions: Tensor, valid: Tensor, strategy: str) -> Tensor:
    """Use the complete global row population; invalid rows have zero weight."""
    if conditions.ndim != 1 or valid.shape != conditions.shape or valid.dtype != torch.bool:
        raise ValueError("aligned condition IDs and boolean validity are required")
    if strategy not in ("row_mean", "condition_mean"):
        raise ValueError("unknown unified reduction")
    weights = torch.zeros_like(conditions, dtype=torch.float32)
    if not valid.any():
        return weights
    if strategy == "row_mean":
        weights[valid] = 1.0 / valid.sum()
    else:
        _, inverse, counts = torch.unique(
            conditions[valid], return_inverse=True, return_counts=True
        )
        weights[valid] = 1.0 / (len(counts) * counts[inverse].float())
    return weights


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: Tensor) -> Tensor:
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        count = torch.tensor([len(values)], device=values.device, dtype=torch.long)
        sizes = [torch.zeros_like(count) for _ in range(world)]
        torch.distributed.all_gather(sizes, count)
        lengths = [int(n.item()) for n in sizes]
        padded = values.new_zeros((max(1, max(lengths)), *values.shape[1:]))
        padded[: len(values)] = values
        gathered = [torch.empty_like(padded) for _ in range(world)]
        torch.distributed.all_gather(gathered, padded)
        ctx.start, ctx.length = sum(lengths[:rank]), lengths[rank]
        return torch.cat([v[:n] for v, n in zip(gathered, lengths, strict=True)])

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tensor:
        # A remote row can be selected as a neighbor. Sum its contributions
        # from every rank before routing gradients back to the owning rank.
        gradient = gradient.contiguous().clone()
        if gradient.numel():
            torch.distributed.all_reduce(gradient)
        return gradient[ctx.start : ctx.start + ctx.length]


def gather_rows(values: Tensor) -> Tensor:
    if not torch.distributed.is_initialized():
        return values
    return cast(Tensor, _GatherRows.apply(values))


def nearest_neighbor_terms(values: Tensor) -> Tensor:
    """One loss per row over the supplied full population, excluding self."""
    if len(values) < 2:
        return values.sum(dim=-1) * 0
    normalized = F.normalize(values.float(), dim=-1)
    similarity = normalized.detach() @ normalized.detach().T
    similarity.fill_diagonal_(float("-inf"))
    neighbors = similarity.argmax(-1)
    return cast(
        Tensor, -(torch.linalg.vector_norm(normalized - normalized[neighbors], dim=-1) + 1e-8).log()
    )
