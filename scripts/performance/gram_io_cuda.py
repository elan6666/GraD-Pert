"""Diagnostic KDA Gram backward without a channel-expanded HBM tensor.

Each program owns one token and head and reduces both incident edge directions.
No atomic addition, random draw, gate clipping or change to the triangular solve.
This is a local IO-aware adaptation, not a reproduction of an upstream kernel.
"""

import torch
import triton  # type: ignore[import-not-found]
import triton.language as tl  # type: ignore[import-not-found]
from triton.language.extra import cuda as cuda_extra  # type: ignore[import-not-found]


@triton.jit(do_not_specialize=["N"])
def _backward(K, G, U, DK, DG, N, H: tl.constexpr, UB, UH, UI, UJ):
    token = tl.program_id(0)
    batch_head = tl.program_id(1)
    neighbor = tl.arange(0, 32)
    channel = tl.arange(0, 64)
    base = batch_head * N * 64
    ki = tl.load(K + base + token * 64 + channel)
    gi = tl.load(G + base + token * 64 + channel)
    pointers = base + neighbor[:, None] * 64 + channel[None, :]
    kj = tl.load(K + pointers, neighbor[:, None] < N, 0.0)
    gj = tl.load(G + pointers, neighbor[:, None] < N, 0.0)
    ubase = (batch_head // H) * UB + (batch_head % H) * UH

    earlier = neighbor < token
    later = (neighbor > token) & (neighbor < N)
    u_left = tl.load(U + ubase + token * UI + neighbor * UJ, earlier, 0.0)
    u_right = tl.load(U + ubase + neighbor * UI + token * UJ, later, 0.0)
    # Mask before exp: unused upper-triangle exponents may overflow otherwise.
    left_decay = cuda_extra.libdevice.exp(tl.where(earlier[:, None], gi[None, :] - gj, 0.0))
    right_decay = cuda_extra.libdevice.exp(tl.where(later[:, None], gj - gi[None, :], 0.0))
    left_key = (u_left[:, None] * left_decay) * kj
    right_key = (u_right[:, None] * right_decay) * kj
    dk = tl.sum(left_key, axis=0) + tl.sum(right_key, axis=0)
    left_gate = (u_left[:, None] * (ki[None, :] * kj)) * left_decay
    right_gate = (u_right[:, None] * (kj * ki[None, :])) * right_decay
    dg = tl.sum(left_gate, axis=0) - tl.sum(right_gate, axis=0)
    destination = base + token * 64 + channel
    tl.store(DK + destination, dk)
    tl.store(DG + destination, dg)


def backward(keys, gates, upstream):
    batch, heads, length, _ = keys.shape
    assert keys.is_contiguous() and gates.is_contiguous()
    key_gradient, gate_gradient = torch.empty_like(keys), torch.empty_like(gates)
    if batch * heads:
        _backward[(length, batch * heads)](
            keys,
            gates,
            upstream,
            key_gradient,
            gate_gradient,
            length,
            heads,
            *upstream.stride(),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return key_gradient, gate_gradient
