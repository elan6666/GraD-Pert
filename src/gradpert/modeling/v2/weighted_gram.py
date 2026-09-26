"""Isolated KDA weighted-Gram candidate; not connected to any model default.

Only the strict lower triangle affects the delta block solve. The CUDA candidate
will fuse pairwise channel products into that triangle. Its initial backward
deliberately uses the original PyTorch pointwise/reduction ordering, recomputing
decay rather than saving the five-dimensional forward temporary. This trades
backward work for forward launches/storage and must be measured end to end.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor


def _decay(keys: Tensor, gates: Tensor) -> tuple[Tensor, Tensor]:
    length = keys.shape[-2]
    legal = torch.ones(length, length, device=keys.device, dtype=torch.bool).tril(-1)
    difference = gates.unsqueeze(-2) - gates.unsqueeze(-3)
    return difference.masked_fill(~legal[..., None], 0).exp(), legal


def weighted_gram_reference(keys: Tensor, gates: Tensor) -> Tensor:
    """keys/gates: [batch, head, length, channel], preserving input dtype."""
    decay, legal = _decay(keys, gates)
    product = keys.unsqueeze(-2) * keys.unsqueeze(-3)
    return (product * decay).sum(-1).masked_fill(~legal, 0)


def weighted_gram_backward(keys: Tensor, gates: Tensor, upstream: Tensor) -> tuple[Tensor, Tensor]:
    decay, legal = _decay(keys, gates)
    u = upstream.masked_fill(~legal, 0).unsqueeze(-1)
    product_gradient = u * decay
    key_gradient = (product_gradient * keys.unsqueeze(-3)).sum(-2)
    key_gradient = key_gradient + (product_gradient * keys.unsqueeze(-2)).sum(-3)
    del product_gradient
    difference_gradient = (u * (keys.unsqueeze(-2) * keys.unsqueeze(-3))) * decay
    difference_gradient = difference_gradient.masked_fill(~legal[..., None], 0)
    gate_gradient = difference_gradient.sum(-2) + (-difference_gradient).sum(-3)
    return key_gradient, gate_gradient


class _WeightedGram(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, keys: Tensor, gates: Tensor) -> Tensor:
        from ._weighted_gram_cuda import forward

        # Preserve original strides for the eager backward's reduction ordering.
        ctx.save_for_backward(keys, gates)
        return forward(keys, gates)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, upstream: Tensor) -> tuple[Tensor, Tensor]:
        keys, gates = ctx.saved_tensors
        return weighted_gram_backward(keys, gates, upstream)


def fused_weighted_gram(keys: Tensor, gates: Tensor) -> Tensor:
    if (
        keys.ndim != 4
        or keys.shape != gates.shape
        or keys.device != gates.device
        or not keys.is_cuda
        or keys.dtype != torch.float32
        or gates.dtype != torch.float32
        or keys.shape[-1] != 64
        or not 1 <= keys.shape[-2] <= 32
    ):
        raise ValueError("weighted-Gram candidate requires matching CUDA FP32 [B,H,1..32,64]")
    apply = cast(Callable[..., Tensor], _WeightedGram.apply)
    return apply(keys, gates)
