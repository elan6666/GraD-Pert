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
    value = tl.sum(product * cuda_extra.libdevice.exp(difference), axis=1)
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
