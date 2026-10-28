"""Scalar summaries of two gradients at the same fixed model parameters.

Inputs are in-memory autograd results, never persisted by this module. Missing
gradients describe disconnected parameters; numerical zero gradients describe
connected parameters. A cosine needs two nonzero norms on their shared support.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor


def _cpu_chunk(value: Tensor, start: int, stop: int) -> Tensor:
    chunk = value[start:stop].to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(chunk).all():
        raise ValueError("finite floating-point gradients required")
    return chunk


def summarize_gradient_pair(
    prediction: Mapping[str, Tensor | None],
    auxiliary: Mapping[str, Tensor | None],
    *,
    auxiliary_scale: float,
) -> dict:
    """Compare scalar derivatives, preserving both support and scaling semantics.

    Both maps must cover exactly the same ordered model parameter names. The
    caller defines the scalar losses and scale; this is neither a replay of an
    optimizer update nor a measurement of a full joint objective with SSL.
    """
    if not prediction or tuple(prediction) != tuple(auxiliary):
        raise ValueError("same nonempty ordered parameter population required")
    if not math.isfinite(auxiliary_scale) or auxiliary_scale < 0:
        raise ValueError("finite nonnegative auxiliary scale required")
    counts = dict(
        parameters=len(prediction),
        prediction_connected=0,
        auxiliary_connected=0,
        shared_connected=0,
        prediction_only=0,
        auxiliary_only=0,
        neither_connected=0,
        prediction_numerically_zero=0,
        auxiliary_numerically_zero=0,
        shared_elements=0,
    )
    totals = {
        name: []
        for name in ("pred", "aux", "pred_shared", "aux_shared", "dot", "sum", "pred_dot_sum")
    }
    for name, first in prediction.items():
        second = auxiliary[name]
        a, b = first is not None, second is not None
        counts["prediction_connected"] += int(a)
        counts["auxiliary_connected"] += int(b)
        counts["shared_connected"] += int(a and b)
        counts["prediction_only"] += int(a and not b)
        counts["auxiliary_only"] += int(b and not a)
        counts["neither_connected"] += int(not a and not b)
        for value in (first, second):
            if value is not None and (
                not value.is_floating_point() or value.layout != torch.strided
            ):
                raise ValueError("dense floating-point gradients required")
        if a and b:
            if first.shape != second.shape:
                raise ValueError("shared parameter gradient shape differs: " + name)
            counts["shared_elements"] += first.numel()
        if not a and not b:
            continue
        x = first.detach().reshape(-1) if a else None
        y = second.detach().reshape(-1) if b else None
        size = x.numel() if a else y.numel()
        x_nonzero = y_nonzero = False
        for start in range(0, size, 262144):
            stop = min(start + 262144, size)
            xx = _cpu_chunk(x, start, stop) if a else None
            yy = _cpu_chunk(y, start, stop) if b else None
            px = float(xx.square().sum()) if a else 0.0
            ay = float(yy.square().sum()) if b else 0.0
            x_nonzero |= px > 0
            y_nonzero |= ay > 0
            totals["pred"].append(px)
            totals["aux"].append(ay)
            if a and b:
                totals["pred_shared"].append(px)
                totals["aux_shared"].append(ay)
                totals["dot"].append(float((xx * yy).sum()))
            combined = xx + auxiliary_scale * yy if a and b else xx if a else auxiliary_scale * yy
            # Direct sums avoid cancellation in ||p||² + scale²||a||² + 2 scale p·a.
            totals["sum"].append(float(combined.square().sum()))
            if a:
                totals["pred_dot_sum"].append(float((xx * combined).sum()))
        counts["prediction_numerically_zero"] += int(a and not x_nonzero)
        counts["auxiliary_numerically_zero"] += int(b and not y_nonzero)
    energies = {key: math.fsum(values) for key, values in totals.items()}
    if not all(math.isfinite(x) for x in energies.values()):
        raise FloatingPointError("gradient summary overflow")
    p, a = math.sqrt(energies["pred_shared"]), math.sqrt(energies["aux_shared"])
    dot = energies["dot"]
    cosine = max(-1.0, min(1.0, dot / (p * a))) if p > 0 and a > 0 else None
    status = (
        "ok"
        if cosine is not None
        else "no_shared_connected_parameters"
        if not counts["shared_connected"]
        else "zero_norm_on_shared_support"
    )
    return dict(
        status=status,
        counts=counts,
        prediction_l2=math.sqrt(energies["pred"]),
        auxiliary_l2=math.sqrt(energies["aux"]),
        prediction_shared_l2=p,
        auxiliary_shared_l2=a,
        shared_dot=dot,
        shared_cosine=cosine,
        auxiliary_scale=auxiliary_scale,
        scaled_shared_auxiliary_to_prediction_l2=auxiliary_scale * a / p if p > 0 else None,
        prediction_dot_two_branch_sum=energies["pred_dot_sum"],
        two_branch_sum_l2=math.sqrt(energies["sum"]),
        interpretation="Fixed-point scalar-loss gradients; no optimizer or SSL update inferred.",
    )
