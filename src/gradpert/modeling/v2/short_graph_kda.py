"""Fused final-state recurrence for independent short graph neighborhoods.

Only the write-only KDA state is computed; graph target reads stay in the
existing model. This native kernel follows the project recurrence rather than
importing an upstream attention package. The analytic reverse recomputes
per-token states within one bounded graph chunk.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"], do_not_specialize_on_alignment=["T"])
def _short_forward(
    K,
    V,
    D,
    BETA,
    OUT,
    STATES,
    T,
    H: tl.constexpr,
    W: tl.constexpr,
    SAVE_STATES: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    i = tl.arange(0, W)
    j = tl.arange(0, W)
    state = tl.full((W, W), 0, tl.float32)
    for t in range(T):
        offset = ((row * T + t) * H + head) * W
        key = tl.load(K + offset + i).to(tl.float32)
        value = tl.load(V + offset + j).to(tl.float32)
        decay = tl.exp(tl.load(D + offset + i).to(tl.float32))
        beta = tl.load(BETA + (row * T + t) * H + head).to(tl.float32)
        if SAVE_STATES:
            state_offset = ((row * H + head) * T + t) * W * W
            tl.store(STATES + state_offset + i[:, None] * W + j[None, :], state)
        old = decay[:, None] * state
        error = value - tl.sum(key[:, None] * old, axis=0)
        state = old + (beta * key)[:, None] * error[None, :]
    out_offset = (row * H + head) * W * W
    tl.store(OUT + out_offset + i[:, None] * W + j[None, :], state)


@triton.jit(do_not_specialize=["T"], do_not_specialize_on_alignment=["T"])
def _short_backward(
    K,
    V,
    D,
    BETA,
    STATES,
    DOUT,
    DK,
    DV,
    DD,
    DBETA,
    T,
    H: tl.constexpr,
    W: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    i = tl.arange(0, W)
    j = tl.arange(0, W)
    out_offset = (row * H + head) * W * W
    grad = tl.load(DOUT + out_offset + i[:, None] * W + j[None, :]).to(tl.float32)
    for reverse in range(T):
        t = T - 1 - reverse
        offset = ((row * T + t) * H + head) * W
        key = tl.load(K + offset + i).to(tl.float32)
        value = tl.load(V + offset + j).to(tl.float32)
        decay = tl.exp(tl.load(D + offset + i).to(tl.float32))
        beta = tl.load(BETA + (row * T + t) * H + head).to(tl.float32)
        state_offset = ((row * H + head) * T + t) * W * W
        before = tl.load(STATES + state_offset + i[:, None] * W + j[None, :])
        old = decay[:, None] * before
        error = value - tl.sum(key[:, None] * old, axis=0)
        d_error = beta * tl.sum(grad * key[:, None], axis=0)
        d_key = beta * tl.sum(grad * error[None, :], axis=1)
        d_key -= tl.sum(old * d_error[None, :], axis=1)
        d_old = grad - key[:, None] * d_error[None, :]
        d_decay_log = tl.sum(d_old * old, axis=1)
        d_beta = tl.sum(tl.sum(grad * key[:, None] * error[None, :], axis=0), axis=0)
        tl.store(DK + offset + i, d_key)
        tl.store(DV + offset + j, d_error)
        tl.store(DD + offset + i, d_decay_log)
        tl.store(DBETA + (row * T + t) * H + head, d_beta)
        grad = d_old * decay[:, None]


class ShortFinalState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, key, value, log_decay, beta):
        if not (key.is_cuda and key.shape == value.shape == log_decay.shape):
            raise ValueError("short KDA requires aligned CUDA writes")
        if key.shape[-1] != 64 or key.shape[1] > 64 or beta.shape != key.shape[:-1]:
            raise ValueError("probe supports W64 and at most 64 neighbors")
        key, value, log_decay, beta = (x.contiguous() for x in (key, value, log_decay, beta))
        n, length, heads, width = key.shape
        out = torch.empty((n, heads, width, width), device=key.device, dtype=torch.float32)
        _short_forward[(n, heads)](
            key, value, log_decay, beta, out, out, length, heads, width, False, num_warps=4
        )
        ctx.save_for_backward(key, value, log_decay, beta)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        key, value, log_decay, beta = ctx.saved_tensors
        n, length, heads, width = key.shape
        before = torch.empty(
            (n, heads, length, width, width), device=key.device, dtype=torch.float32
        )
        final = torch.empty((n, heads, width, width), device=key.device, dtype=torch.float32)
        _short_forward[(n, heads)](
            key,
            value,
            log_decay,
            beta,
            final,
            before,
            length,
            heads,
            width,
            True,
            num_warps=4,
        )
        dk, dv, dd, dbeta = (torch.empty_like(x) for x in (key, value, log_decay, beta))
        _short_backward[(n, heads)](
            key,
            value,
            log_decay,
            beta,
            before,
            grad_out.contiguous(),
            dk,
            dv,
            dd,
            dbeta,
            length,
            heads,
            width,
            num_warps=4,
        )
        if value.dtype == torch.bfloat16:
            # The project reference uses a block triangular solve. A few
            # float32 recurrent-adjoint values straddle BF16 rounding ties;
            # retain the reference path for dV until a matching fused adjoint
            # has been verified. This is an execution-only diagnostic and
            # must earn its end-to-end speedup despite the extra work.
            from .operators import chunk_delta_final_state

            with torch.enable_grad():
                reference_value = value.detach().requires_grad_(True)
                reference_state = chunk_delta_final_state(
                    key.detach(), reference_value, log_decay.detach(), beta.detach()
                )
                dv = torch.autograd.grad(reference_state, reference_value, grad_out)[0]
        return dk, dv, dd, dbeta


def short_graph_final_state(
    key: torch.Tensor, value: torch.Tensor, log_decay: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Apply one independent short scan per graph target/head."""
    return ShortFinalState.apply(key, value, log_decay, beta)
