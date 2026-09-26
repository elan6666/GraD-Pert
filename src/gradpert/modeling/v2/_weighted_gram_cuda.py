"""Forward-only fused KDA Gram candidate, with no channel-expanded allocation."""

import torch
import triton  # type: ignore[import-not-found]
import triton.language as tl  # type: ignore[import-not-found]
from torch import Tensor
from triton.language.extra import cuda as cuda_extra  # type: ignore[import-not-found]


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["N"])
def _gram(K, G, P, N):  # type: ignore[no-untyped-def]
    query = tl.program_id(0)
    batch_head = tl.program_id(1)
    neighbor = tl.arange(0, 32)
    channel = tl.arange(0, 64)
    base = batch_head * N * 64
    ki = tl.load(K + base + query * 64 + channel)
    gi = tl.load(G + base + query * 64 + channel)
    indices = base + neighbor[:, None] * 64 + channel[None, :]
    kj = tl.load(K + indices, neighbor[:, None] < N, 0)
    gj = tl.load(G + indices, neighbor[:, None] < N, 0)
    legal = neighbor < query
    difference = tl.where(legal[:, None], gi[None, :] - gj, 0.0)
    # Preserve multiply order; do not rewrite decay as exp(g_i)/exp(g_j).
    product = ki[None, :] * kj
    weighted = product * cuda_extra.libdevice.exp(difference)
    # Match ATen's descending-offset 64-channel reduction: combine halves
    # before the warp-width tree, rather than adjacent per-thread elements.
    half = tl.arange(0, 32)
    paired = tl.gather(weighted, tl.broadcast_to(half[None, :], (32, 32)), axis=1)
    paired += tl.gather(weighted, tl.broadcast_to((half + 32)[None, :], (32, 32)), axis=1)
    indices_16 = tl.broadcast_to(tl.arange(0, 16)[None, :], (32, 16))
    paired = tl.gather(paired, indices_16, axis=1) + tl.gather(paired, indices_16 + 16, axis=1)
    indices_8 = tl.broadcast_to(tl.arange(0, 8)[None, :], (32, 8))
    paired = tl.gather(paired, indices_8, axis=1) + tl.gather(paired, indices_8 + 8, axis=1)
    indices_4 = tl.broadcast_to(tl.arange(0, 4)[None, :], (32, 4))
    paired = tl.gather(paired, indices_4, axis=1) + tl.gather(paired, indices_4 + 4, axis=1)
    indices_2 = tl.broadcast_to(tl.arange(0, 2)[None, :], (32, 2))
    paired = tl.gather(paired, indices_2, axis=1) + tl.gather(paired, indices_2 + 2, axis=1)
    indices_1 = tl.broadcast_to(tl.arange(0, 1)[None, :], (32, 1))
    paired = tl.gather(paired, indices_1, axis=1) + tl.gather(paired, indices_1 + 1, axis=1)
    value = tl.sum(paired, axis=1)
    tl.store(P + (batch_head * N + query) * N + neighbor, tl.where(legal, value, 0.0), neighbor < N)


def forward(keys: Tensor, gates: Tensor) -> Tensor:
    batch, heads, length, _ = keys.shape
    output = torch.empty((batch, heads, length, length), device=keys.device, dtype=torch.float32)
    if batch * heads:
        _gram[(length, batch * heads)](
            keys.contiguous(),
            gates.contiguous(),
            output,
            length,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
