"""Auxiliary alignment and spherical uniformity on a full effective population.

The caller deduplicates (control, condition) records before gathering both
Global views. Each view has N rows; the views must never become a 2N-row
population. Prediction tensors are not modified. No Teacher or stop-gradient
branch is introduced here.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class AUResult:
    alignment: Tensor
    uniformity: Tensor
    pair_count: int
    input_count: int
    context_count: int


def _validate_ids(condition_ids: Tensor, context_ids: Tensor, n: int | None = None) -> int:
    integral = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    for name, values in (("condition_ids", condition_ids), ("context_ids", context_ids)):
        if not isinstance(values, Tensor) or values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional integer tensor")
        if values.dtype not in integral:
            raise ValueError(f"{name} must have an integer dtype (not bool or floating point)")
    count = condition_ids.numel()
    if context_ids.numel() != count or (n is not None and count != n):
        raise ValueError("condition_ids and context_ids must match the N representation rows")
    return count


def pair_census(condition_ids: Tensor, context_ids: Tensor) -> dict[str, Any]:
    """Return metadata-only counts; zero-pair populations are valid for diagnosis.

    Context weights are the fractions of all eligible unordered pairs. The
    loss averages pairs, not contexts. This helper does not deduplicate inputs
    because it has no control IDs; deduplication belongs to the caller.
    """
    n = _validate_ids(condition_ids, context_ids)
    grouped: dict[int, Counter[int]] = {}
    conditions = condition_ids.detach().cpu().tolist()
    contexts = context_ids.detach().cpu().tolist()
    for condition, context in zip(conditions, contexts, strict=True):
        grouped.setdefault(context, Counter())[condition] += 1
    rows = []
    for context, counts in sorted(grouped.items()):
        context_n = sum(counts.values())
        pairs = context_n * (context_n - 1) // 2
        pairs -= sum(count * (count - 1) // 2 for count in counts.values())
        rows.append(
            {
                "context_id": context,
                "input_count": context_n,
                "condition_count": len(counts),
                "pair_count": pairs,
            }
        )
    total = sum(row["pair_count"] for row in rows)
    return {
        "input_count": n,
        "pair_count": total,
        "context_count": len(rows),
        "contexts": [
            {**row, "pair_weight": row["pair_count"] / total if total else 0.0} for row in rows
        ],
    }


def alignment_uniformity(
    z1: Tensor,
    z2: Tensor,
    condition_ids: Tensor,
    context_ids: Tensor,
    temperature: float = 2.0,
    *,
    compute_uniformity: bool = True,
) -> AUResult:
    """Compute A and U without weights, margins, rank targets, or detach.

    A = mean_i(1 - cos(z1_i, z2_i)). For each view, U is log-mean-exp
    of 2*t*(cos(i,j)-1) over i<j in the same context and different conditions;
    the returned U averages the two views. ``temperature`` is the positive
    concentration t from the spherical-distance kernel, not a divisor.

    Both Global inputs retain gradients through auxiliary FP32 L2 normalization.
    Exactly zero (or FP32-underflowed) row norms have no defined direction and
    are rejected, rather than replaced with artificial unit vectors. Empty
    eligible pair sets also fail explicitly. Call on the full effective
    population: averaging microbatch log-mean-exp is a different objective.
    """
    if not isinstance(z1, Tensor) or not isinstance(z2, Tensor):
        raise ValueError("both Global representations must be tensors")
    if z1.ndim != 2 or z2.shape != z1.shape:
        raise ValueError("both Global representations must have identical shape [N, D]")
    n, width = z1.shape
    if n < 2 or width < 1:
        raise ValueError("alignment_uniformity requires N >= 2 and D >= 1")
    if z1.device != z2.device:
        raise ValueError("both Global representations must be on the same device")
    if not z1.is_floating_point() or not z2.is_floating_point():
        raise ValueError("both Global representations must have real floating-point dtype")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite positive number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")
    if temperature > torch.finfo(torch.float32).max / 4:
        raise ValueError("temperature overflows the FP32 spherical-distance kernel")
    _validate_ids(condition_ids, context_ids, n)
    conditions = condition_ids.to(device=z1.device)
    contexts = context_ids.to(device=z1.device)
    pairs = torch.triu(
        contexts[:, None].eq(contexts[None, :]) & conditions[:, None].ne(conditions[None, :]),
        diagonal=1,
    )
    pair_count = int(pairs.count_nonzero().item())
    if compute_uniformity and pair_count == 0:
        raise ValueError("no eligible same-context, different-condition unordered pairs")

    # Explicitly disable outer mixed precision: .float() alone does not stop
    # autocast from reducing the precision of the Gram matrix multiplication.
    with torch.autocast(device_type=z1.device.type, enabled=False):
        normalized = []
        for name, values in (("z1", z1), ("z2", z2)):
            values = values.float()
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"{name} contains non-finite values after FP32 conversion")
            norms = torch.linalg.vector_norm(values, dim=1)
            if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
                raise ValueError(f"{name} has zero or non-finite FP32 row norm")
            normalized.append(F.normalize(values, p=2, dim=1, eps=0.0))
        u1, u2 = normalized
        alignment = (1.0 - (u1 * u2).sum(dim=1)).mean()
        uniformities = (
            [
                torch.logsumexp(2.0 * temperature * ((view @ view.T)[pairs] - 1.0), dim=0)
                - math.log(pair_count)
                for view in normalized
            ]
            if compute_uniformity
            else []
        )
        uniformity = (
            (uniformities[0] + uniformities[1]) * 0.5 if compute_uniformity else alignment * 0
        )

    return AUResult(alignment, uniformity, pair_count, n, torch.unique(contexts).numel())


def negative_coding_rate(global0: Tensor, global1: Tensor, epsilon_squared: float = 0.05) -> Tensor:
    """Shared native training/diagnostic CR; normalize separate Globals without centering."""
    if (
        not isinstance(global0, Tensor)
        or not isinstance(global1, Tensor)
        or global0.ndim != 2
        or global1.shape != global0.shape
        or global0.device != global1.device
        or not global0.is_floating_point()
        or not global1.is_floating_point()
    ):
        raise ValueError("coding rate requires aligned floating-point Global matrices")
    rates = []
    for values in (global0, global1):
        if values.ndim != 2 or not all(values.shape):
            raise ValueError("coding rate requires nonempty feature matrices")
        x = values.double()
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        if not torch.isfinite(x).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("coding rate requires finite nonzero unit norms")
        if epsilon_squared != 0.05:
            raise ValueError("coding-rate epsilon_squared is fixed at 0.05")
        unit = x / norms
        n, d = unit.shape
        gram = unit.T @ unit
        matrix = torch.eye(d, dtype=x.dtype, device=x.device) + d / (n * epsilon_squared) * gram
        matrix = (matrix + matrix.T) / 2
        factor, info = torch.linalg.cholesky_ex(matrix)
        if info.any() or not torch.isfinite(factor).all():
            raise ValueError("coding rate matrix is not finite positive definite")
        rates.append(-factor.diagonal().log().sum())
    return (rates[0] + rates[1]) / 2
