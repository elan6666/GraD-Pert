"""Stateless native FP8 FFN GEMMs; HYBRID adaptation, not DS-V3 scaling.

Master parameters stay in their original native Linear modules. This uses
PyTorch's private FP8-v1 ``_scaled_mm`` API, which must pass the separate
version/device/backend launch probe. There is deliberately no fallback.
"""

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import functional as F

SUPPORTED_CUDA_BUILDS = ("12.8", "12.9", "13.0")


def _aligned(size: int) -> int:
    return (size + 15) // 16 * 16


def _quantize(value: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Return saturated FP8 data and its positive FP32 dequantization scale."""
    value = value.float()
    # Device-side assertion avoids a per-GEMM CPU synchronization. Nonfinite
    # source data is an error, never silently converted into a finite tensor.
    torch._assert_async(torch.isfinite(value).all(), "FP8 input must be finite")
    limit = torch.finfo(dtype).max
    scale = (value.abs().amax() / limit).clamp_min(torch.finfo(torch.float32).tiny)
    return (value / scale).clamp(-limit, limit).to(dtype), scale


def _gemm(
    a: Tensor,
    b: Tensor,
    sa: Tensor,
    sb: Tensor,
    dtype: torch.dtype,
    *,
    operation: str,
    parameter_id: int,
) -> Tensor:
    # The FP8-v1 API requires row-major A and column-major B. Do not let
    # autocast narrow the FP32 gradient result requested by backward.
    with torch.autocast(device_type=a.device.type, enabled=False):
        return torch._scaled_mm(
            a.contiguous(),
            b.t().contiguous().t(),
            scale_a=sa,
            scale_b=sb,
            out_dtype=dtype,
            use_fast_accum=False,
        )


class _HybridLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, weight: Tensor) -> Tensor:
        if x.ndim < 1 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
            raise ValueError("FP8 linear input/weight shape mismatch")
        if x.numel() == 0 or 0 in weight.shape:
            raise ValueError("FP8 linear does not accept empty dimensions")
        rows, inner = x.numel() // x.shape[-1], x.shape[-1]
        outer = weight.shape[0]
        m, k, n = _aligned(rows), _aligned(inner), _aligned(outer)
        xp = F.pad(x.reshape(rows, inner), (0, k - inner, 0, m - rows))
        wp = F.pad(weight, (0, k - inner, 0, n - outer))
        x8, sx = _quantize(xp, torch.float8_e4m3fn)
        w8, sw = _quantize(wp, torch.float8_e4m3fn)
        result = _gemm(
            x8, w8.t(), sx, sw, torch.bfloat16, operation="forward", parameter_id=id(weight)
        )
        ctx.save_for_backward(x8, w8, sx, sw)
        ctx.input_shape = x.shape
        ctx.input_dtype, ctx.weight_dtype = x.dtype, weight.dtype
        ctx.logical_shape = rows, inner, outer
        ctx.parameter_id = id(weight)
        return result[:rows, :outer].reshape(*x.shape[:-1], outer)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor | None, Tensor | None]:
        x8, w8, sx, sw = ctx.saved_tensors
        rows, inner, outer = ctx.logical_shape
        gp = F.pad(
            grad.reshape(rows, outer),
            (0, w8.shape[0] - outer, 0, x8.shape[0] - rows),
        )
        g8, sg = _quantize(gp, torch.float8_e5m2)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = _gemm(
                g8, w8, sg, sw, torch.float32, operation="dgrad", parameter_id=ctx.parameter_id
            )
            dx = dx[:rows, :inner].reshape(ctx.input_shape).to(ctx.input_dtype)
        if ctx.needs_input_grad[1]:
            dw = _gemm(
                g8.t(), x8, sg, sx, torch.float32, operation="wgrad", parameter_id=ctx.parameter_id
            )
            dw = dw[:outer, :inner].to(ctx.weight_dtype)
        return dx, dw


def fp8_linear(x: Tensor, weight: Tensor) -> Tensor:
    """CUDA-only real FP8 GEMM with native autograd and no persistent state."""
    if x.device.type != "cuda" or weight.device != x.device:
        raise RuntimeError("FP8 FFN requires input and weight on the same CUDA device")
    if torch.__version__.split("+")[0] != "2.13.0":
        raise RuntimeError("FP8 FFN private API is pinned to torch 2.13.0")
    if torch.version.cuda not in SUPPORTED_CUDA_BUILDS:
        raise RuntimeError("FP8 FFN requires an explicitly supported CUDA build")
    if torch.cuda.get_device_capability(x.device) != (12, 0):
        raise RuntimeError("this FP8 FFN candidate requires the approved SM12.0 devices")
    if not callable(getattr(torch, "_scaled_mm", None)):
        raise RuntimeError("FP8 FFN requires the verified torch._scaled_mm FP8-v1 API")
    if x.dtype not in (torch.float32, torch.bfloat16) or weight.dtype != torch.float32:
        raise RuntimeError("FP8 FFN requires FP32 master weights and FP32/BF16 input")
    apply = cast(Callable[..., Tensor], _HybridLinear.apply)
    return apply(x, weight)
