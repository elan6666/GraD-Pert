"""Isolated forward+backward Gram IO candidate; native defaults are untouched."""

from contextlib import contextmanager

import torch


def backward_tiled_reference(keys, gates, upstream):
    """CPU oracle for the two incident-edge reductions, checked against autograd.

    This is a numerical oracle, never a CUDA performance fallback. Large
    channel-expanded tensors are limited to one target token at a time.
    """
    gradients = []
    for token in range(keys.shape[-2]):
        ki, gi = keys[..., token : token + 1, :], gates[..., token : token + 1, :]
        earlier = keys[..., :token, :]
        later = keys[..., token + 1 :, :]
        dl = (gi - gates[..., :token, :]).exp()
        dr = (gates[..., token + 1 :, :] - gi).exp()
        ul = upstream[..., token, :token, None]
        ur = upstream[..., token + 1 :, token, None]
        dk = ((ul * dl) * earlier).sum(-2) + ((ur * dr) * later).sum(-2)
        dg = ((ul * (ki * earlier)) * dl).sum(-2) - ((ur * (later * ki)) * dr).sum(-2)
        gradients.append((dk, dg))
    return tuple(torch.stack([pair[i] for pair in gradients], -2) for i in (0, 1))


class _GramIO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, keys, gates):
        from gradpert.modeling.v2._weighted_gram_cuda import forward

        # Reuse the original forward candidate; the new hypothesis is backward IO.
        # Save original views: the backward reads strides without new copies.
        ctx.save_for_backward(keys, gates)
        return forward(keys, gates)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        from scripts.performance.gram_io_cuda import backward

        return backward(*ctx.saved_tensors, upstream)


def fused_gram_io(keys, gates):
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
        raise ValueError("Gram IO requires matching CUDA FP32 [B,H,1..32,64]")
    return _GramIO.apply(keys, gates)


@contextmanager
def candidate_scope():
    """Restore the old diagnostic implementation even if the probe fails."""
    from gradpert.modeling.v2 import weighted_gram

    original = weighted_gram.fused_weighted_gram
    weighted_gram.fused_weighted_gram = fused_gram_io
    try:
        yield
    finally:
        weighted_gram.fused_weighted_gram = original
