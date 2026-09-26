"""Experimental fused four-stream Sinkhorn; not selected by model defaults.

The twenty alternating log normalizations are unchanged. A CUDA forward stores
the normalized probabilities for each normalization; the backward applies
u - softmax(z) * sum(u) in reverse order. This reduces launch count without
reducing iterations or detaching any gradient. Higher derivatives are unsupported.
"""

from __future__ import annotations

import torch
from torch import Tensor


def reference_with_saved(logits: Tensor, iterations: int = 20) -> tuple[Tensor, Tensor]:
    """FP32 equation oracle for the saved intermediates, also usable on CPU."""
    z = logits.float()
    probabilities = []
    for _ in range(iterations):
        for axis in (-2, -1):
            z = z - z.logsumexp(axis, keepdim=True)
            probabilities.append(z.exp())
    return z.exp(), torch.stack(probabilities, dim=0)


def reference_backward(upstream: Tensor, output: Tensor, probabilities: Tensor) -> Tensor:
    gradient = upstream.float() * output
    for index in range(len(probabilities) - 1, -1, -1):
        axis = -2 if index % 2 == 0 else -1
        gradient = gradient - probabilities[index] * gradient.sum(axis, keepdim=True)
    return gradient


class _Sinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, iterations):
        from ._sinkhorn_cuda import forward

        output, probabilities = forward(logits, iterations)
        ctx.save_for_backward(output, probabilities)
        ctx.input_dtype = logits.dtype
        ctx.iterations = iterations
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        from ._sinkhorn_cuda import backward

        output, probabilities = ctx.saved_tensors
        gradient = backward(upstream, output, probabilities, ctx.iterations)
        return gradient.to(ctx.input_dtype), None


def fused_sinkhorn(logits: Tensor, iterations: int = 20) -> Tensor:
    if not logits.is_cuda or logits.shape[-2:] != (4, 4):
        raise ValueError("experimental fused Sinkhorn requires CUDA and four streams")
    if not 1 <= iterations <= 32:
        raise ValueError("iterations must be between 1 and 32")
    return _Sinkhorn.apply(logits.contiguous(), iterations)
