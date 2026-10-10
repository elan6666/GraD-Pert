"""Population response supervision and bounded, context-matched epoch sampling.

The native MMD estimator follows the multiscale RBF formula verified in
scDFM's ``src/script/run.py`` at commit
``8de47e7d3d443939f93f768f7902c66e0e82eda2``. There is no upstream import.
The explicit degeneracy skip and once-per-epoch truth-row budget are project
choices; this module does not implement flow matching or official sampling.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class PopulationResponseLoss:
    """Unweighted scalar components; the caller owns the MMD coefficient.

    ``bandwidth_squared`` is the detached median squared truth distance, so
    each kernel variance is ``factor * bandwidth_squared``. A skipped MMD
    is a differentiable zero and is identified explicitly, not counted as a
    valid observation by downstream logging.
    """

    mean_mse: Tensor
    mmd_unbiased: Tensor
    mmd_valid: bool
    skip_reason: str | None
    bandwidth_squared: Tensor | None
    prediction_rows: int
    truth_rows: int


def _squared_distances(left: Tensor, right: Tensor) -> Tensor:
    """Use O(m*n) intermediates, rather than materializing [m,n,genes]."""
    return (
        left.square().sum(-1)[:, None] + right.square().sum(-1)[None, :] - 2 * (left @ right.T)
    ).clamp_min(0)


def population_response_loss(
    prediction: Tensor,
    truth: Tensor,
    *,
    bandwidth_factors: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> PopulationResponseLoss:
    r"""Compare two same-condition populations on one already-allowed gene axis.

    For predicted rows X and training-truth rows Y, the mean term is
    ``mean_gene((mean_row(X) - mean_row(Y))**2)``. For each factor a,
    ``k_a(x,y) = exp(-||x-y||**2 / (2*a*median_truth_distance + 1e-12))``.
    The MMD component averages four unbiased estimates by default:

    sum(i!=j) k(X_i,X_j)/(m*(m-1))
    + sum(i!=j) k(Y_i,Y_j)/(n*(n-1)) - 2*mean(i,j) k(X_i,Y_j).

    Within-population diagonals are excluded. Finite negative estimates are
    retained, and duplicated prediction vectors remain legitimate samples.
    A detached training-truth median is the only bandwidth source. The caller
    must enforce training membership, condition/context grouping, and gene
    holdout restrictions; neither labels nor excluded genes are added here.
    """
    if prediction.ndim != 2 or truth.ndim != 2:
        raise ValueError("population inputs must be row-by-gene matrices")
    if prediction.shape[1] != truth.shape[1] or prediction.shape[1] == 0:
        raise ValueError("population inputs require the same nonempty gene axis")
    if len(prediction) == 0 or len(truth) == 0:
        raise ValueError("population inputs require at least one row each")
    if prediction.device != truth.device:
        raise ValueError("population inputs must share a device")
    if not prediction.is_floating_point() or not truth.is_floating_point():
        raise ValueError("population inputs must be floating point")
    factors = tuple(float(factor) for factor in bandwidth_factors)
    if not factors or any(not math.isfinite(factor) or factor <= 0 for factor in factors):
        raise ValueError("bandwidth factors must be finite and positive")

    dtype = (
        torch.float64
        if prediction.dtype == torch.float64 or truth.dtype == torch.float64
        else torch.float32
    )
    # Explicitly disable mixed-precision matmul here: a Gram-matrix distance
    # should not silently revert to bf16 after its inputs are promoted.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        x, y = prediction.to(dtype), truth.to(dtype)
        mean_mse = (x.mean(0) - y.mean(0)).square().mean()
        zero = x.sum() * 0
        m, n = len(x), len(y)

        def skipped(reason: str, bandwidth: Tensor | None = None) -> PopulationResponseLoss:
            return PopulationResponseLoss(mean_mse, zero, False, reason, bandwidth, m, n)

        if m < 2:
            return skipped("prediction_rows_lt_two")
        if n < 2:
            return skipped("truth_rows_lt_two")
        if not bool((y.detach() != y.detach()[0]).any()):
            return skipped("truth_has_no_distinct_vectors")

        # A common translation reduces cancellation when a population has a
        # large shared offset. It changes no real-valued distance or gradient.
        origin = y.detach().mean(0)
        x, y = x - origin, y - origin
        dyy = _squared_distances(y, y)
        truth_off_diagonal = ~torch.eye(n, dtype=torch.bool, device=y.device)
        bandwidth = dyy.detach()[truth_off_diagonal].median()
        if not bool(torch.isfinite(bandwidth)):
            return skipped("truth_bandwidth_nonfinite", bandwidth)
        if not bool(bandwidth > 0):
            return skipped("truth_bandwidth_nonpositive", bandwidth)
        dxx = _squared_distances(x, x)
        dxy = _squared_distances(x, y)
        values = []
        for factor in factors:
            denominator = 2 * factor * bandwidth + 1e-12
            kxx, kyy, kxy = ((-distance / denominator).exp() for distance in (dxx, dyy, dxy))
            xx = (kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
            yy = (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
            values.append(xx + yy - 2 * kxy.mean())
        return PopulationResponseLoss(
            mean_mse, torch.stack(values).mean(), True, None, bandwidth, m, n
        )


@dataclass(frozen=True)
class PopulationRows:
    """One condition's actual truth rows and replacement-sampled controls.

    Controls match the truth rows' contexts. Index alignment specifies only
    context composition; it is not an observed before/after cellular pair.
    """

    condition: Hashable
    truth_rows: tuple[int, ...]
    control_rows: tuple[int, ...]
    contexts: tuple[Hashable, ...]


def population_epoch_batches(
    truth_row_ids: Sequence[int],
    condition_keys: Sequence[Hashable],
    context_keys: Sequence[Hashable],
    control_pools: Mapping[Hashable, Sequence[int]],
    *,
    batch_size: int,
    generator: np.random.Generator,
) -> Iterator[PopulationRows]:
    """Consume each eligible truth row once, in same-condition chunks.

    The caller supplies only eligible training truth IDs and matching control
    pools, already filtered to its canonical cell-type/batch contexts. No
    fixed artificial dataset length, truth resampling, or drop-last is used:
    the epoch truth-row budget is exactly ``len(truth_row_ids)``. Tail groups
    may contain one row, in which case the loss retains mean MSE and skips
    MMD. The explicit NumPy generator can be checkpointed by the caller.
    """
    if batch_size < 1:
        raise ValueError("population batch size must be positive")
    if not (len(truth_row_ids) == len(condition_keys) == len(context_keys)):
        raise ValueError("truth IDs, conditions, and contexts must align")
    truth_ids = tuple(int(row) for row in truth_row_ids)
    if len(set(truth_ids)) != len(truth_ids) or any(row < 0 for row in truth_ids):
        raise ValueError("eligible truth row IDs must be unique and nonnegative")
    pools: dict[Hashable, tuple[int, ...]] = {}
    for context in dict.fromkeys(context_keys):
        if context not in control_pools:
            raise ValueError(f"missing compatible control pool for context {context!r}")
        pool = tuple(int(row) for row in control_pools[context])
        if not pool or any(row < 0 for row in pool):
            raise ValueError(f"invalid compatible control pool for context {context!r}")
        pools[context] = pool
    groups: dict[Hashable, list[int]] = {}
    for position in generator.permutation(len(truth_ids)):
        groups.setdefault(condition_keys[position], []).append(int(position))
    for condition, positions in groups.items():
        for start in range(0, len(positions), batch_size):
            selected = positions[start : start + batch_size]
            contexts = tuple(context_keys[position] for position in selected)
            controls = tuple(int(generator.choice(pools[context])) for context in contexts)
            yield PopulationRows(
                condition,
                tuple(truth_ids[position] for position in selected),
                controls,
                contexts,
            )
