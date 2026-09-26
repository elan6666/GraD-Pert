"""Native Triton kernels for the opt-in four-stream Sinkhorn experiment."""

import torch
import triton
import triton.language as tl
from triton.language.extra import cuda as cuda_extra


@triton.jit
def _forward(X, Y, P, N: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    tokens = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.arange(0, 4)
    col = tl.arange(0, 4)
    index = tokens[:, None, None] * 16 + row[None, :, None] * 4 + col[None, None, :]
    z = tl.load(X + index, tokens[:, None, None] < N, 0).to(tl.float32)
    for step in range(STEPS):
        maximum = tl.max(z, axis=1)
        lse = maximum + cuda_extra.libdevice.log(
            tl.sum(cuda_extra.libdevice.exp(z - maximum[:, None, :]), axis=1)
        )
        z = z - lse[:, None, :]
        probability = cuda_extra.libdevice.exp(z)
        saved = tokens[:, None, None] * (STEPS * 2 * 16) + step * 32
        offsets = row[None, :, None] * 4 + col[None, None, :]
        tl.store(P + saved + offsets, probability, tokens[:, None, None] < N)
        maximum = tl.max(z, axis=2)
        lse = maximum + cuda_extra.libdevice.log(
            tl.sum(cuda_extra.libdevice.exp(z - maximum[:, :, None]), axis=2)
        )
        z = z - lse[:, :, None]
        probability = cuda_extra.libdevice.exp(z)
        tl.store(P + saved + 16 + offsets, probability, tokens[:, None, None] < N)
    tl.store(Y + index, cuda_extra.libdevice.exp(z), tokens[:, None, None] < N)


@triton.jit
def _backward(U, Y, P, DX, N: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    tokens = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.arange(0, 4)
    col = tl.arange(0, 4)
    offsets = row[None, :, None] * 4 + col[None, None, :]
    index = tokens[:, None, None] * 16 + offsets
    mask = tokens[:, None, None] < N
    gradient = tl.load(U + index, mask, 0).to(tl.float32) * tl.load(Y + index, mask, 0)
    for reverse_step in range(STEPS):
        step = STEPS - 1 - reverse_step
        saved = tokens[:, None, None] * (STEPS * 2 * 16) + step * 32 + offsets
        probability = tl.load(P + saved + 16, mask, 0)
        gradient = gradient - probability * tl.sum(gradient, axis=2)[:, :, None]
        probability = tl.load(P + saved, mask, 0)
        gradient = gradient - probability * tl.sum(gradient, axis=1)[:, None, :]
    tl.store(DX + index, gradient, mask)


def forward(logits, iterations):
    count = logits.numel() // 16
    output = torch.empty_like(logits, dtype=torch.float32)
    probabilities = torch.empty((count, iterations * 2, 4, 4), device=logits.device)
    if count:
        _forward[(triton.cdiv(count, 8),)](
            logits, output, probabilities, count, iterations, 8, num_warps=4, enable_fp_fusion=False
        )
    return output, probabilities


def backward(upstream, output, probabilities, iterations):
    count = output.numel() // 16
    gradient = torch.empty_like(output)
    if count:
        _backward[(triton.cdiv(count, 8),)](
            upstream.contiguous(),
            output,
            probabilities,
            gradient,
            count,
            iterations,
            8,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return gradient
