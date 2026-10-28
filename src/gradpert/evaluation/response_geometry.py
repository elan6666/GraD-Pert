"""Pure NumPy condition-balanced response geometry; arrays remain server-private."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _pearson(x, y):
    """Exact constant detection before centering; staged division avoids product overflow."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return None, "fewer_than_two_genes"
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None, "nonfinite_values"
    if x.min() == x.max() or y.min() == y.max():
        return None, "constant_vector"
    # Scaling first keeps finite extremes from overflowing centered sums.
    x, y = x / np.max(np.abs(x)), y / np.max(np.abs(y))
    x, y = x - x.mean(), y - y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return None, "zero_centered_norm"
    value = float(np.dot(x / nx, y / ny))
    return float(np.clip(value, -1, 1)), None


@dataclass
class WeightedRanks:
    """One immutable sort/tie partition; each draw reranks its replicated observations."""

    values: np.ndarray

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim != 1 or not np.isfinite(self.values).all():
            raise ValueError("finite rank vector required")
        self.order = np.argsort(self.values, kind="stable")
        ordered = self.values[self.order]
        self.starts = (
            np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
            if len(ordered)
            else np.array([], dtype=int)
        )
        self.lengths = np.diff(np.r_[self.starts, len(ordered)])

    def ranks(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        if (
            weights.shape != self.values.shape
            or not np.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("finite nonnegative aligned multiplicities required")
        if not len(weights):
            return weights.copy()
        mass = np.add.reduceat(weights[self.order], self.starts)
        mid = np.cumsum(mass) - mass / 2 + 0.5
        result = np.empty(len(weights), dtype=np.float64)
        result[self.order] = np.repeat(mid, self.lengths)
        return result


def weighted_spearman(x, y, weights=None):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("aligned one-dimensional distances required")
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    return _weighted_rank_correlation(WeightedRanks(x), WeightedRanks(y), weights)


def _weighted_rank_correlation(x, y, weights):
    weights = np.asarray(weights, dtype=float)
    rx, ry = x.ranks(weights), y.ranks(weights)
    positive = weights > 0
    if positive.sum() < 2:
        return None
    w = weights[positive]
    rx, ry = rx[positive], ry[positive]
    if rx.min() == rx.max() or ry.min() == ry.max():
        return None
    total = w.sum()
    rx, ry = rx - np.dot(w, rx) / total, ry - np.dot(w, ry) / total
    nx, ny = np.sqrt(np.dot(w, rx * rx)), np.sqrt(np.dot(w, ry * ry))
    if nx == 0 or ny == 0 or not np.isfinite([nx, ny]).all():
        return None
    return float(np.clip(np.dot(w, (rx / nx) * (ry / ny)), -1, 1))


@dataclass
class Geometry:
    ids: tuple[str, ...]
    allowed: np.ndarray
    targets: tuple[tuple[int, ...], ...]
    truth: np.ndarray
    pearson: np.ndarray
    mse: np.ndarray
    reasons: list[str | None]
    genes: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    response_distance: np.ndarray
    cls_distance: np.ndarray
    pair_genes: np.ndarray


def _geometry(
    *,
    condition_ids,
    allowed_gene_indices,
    own_target_indices,
    prediction_delta,
    truth_delta,
    cls_means,
):
    ids = tuple(str(x) for x in condition_ids)
    allowed = np.asarray(allowed_gene_indices)
    own = np.asarray(own_target_indices)
    pred, truth, z = [
        np.asarray(a, dtype=np.float64) for a in (prediction_delta, truth_delta, cls_means)
    ]
    p = len(ids)
    if (
        p < 2
        or len(set(ids)) != p
        or allowed.ndim != 1
        or allowed.dtype.kind not in "iu"
        or not len(allowed)
        or (allowed < 0).any()
        or len(np.unique(allowed)) != len(allowed)
    ):
        raise ValueError("unique condition axis and nonempty unique global gene axis required")
    if own.ndim != 2 or own.shape[0] != p or own.dtype.kind not in "iu" or (own < -1).any():
        raise ValueError("P by K integral global target IDs with -1 padding required")
    if (
        pred.shape != (p, len(allowed))
        or truth.shape != pred.shape
        or z.ndim != 2
        or z.shape[0] != p
        or z.shape[1] == 0
    ):
        raise ValueError("response/CLS shapes differ from condition and allowed axes")
    targets = tuple(tuple(sorted(set(int(i) for i in row if i >= 0))) for row in own)
    pearson, mse, genes = np.full(p, np.nan), np.full(p, np.nan), np.zeros(p, dtype=int)
    reasons = []
    for i in range(p):
        keep = ~np.isin(allowed, targets[i])
        genes[i] = keep.sum()
        value, reason = _pearson(pred[i, keep], truth[i, keep])
        if value is not None:
            pearson[i] = value
        reasons.append(reason)
        if keep.any() and np.isfinite(pred[i, keep]).all() and np.isfinite(truth[i, keep]).all():
            with np.errstate(over="ignore", invalid="ignore"):
                error = float(np.mean((pred[i, keep] - truth[i, keep]) ** 2))
            if np.isfinite(error):
                mse[i] = error
    ii, jj = np.triu_indices(p, 1)
    rd, zd = np.full(len(ii), np.nan), np.full(len(ii), np.nan)
    ng = np.zeros(len(ii), dtype=int)
    # Per-row blocks avoid P*P*A temporary memory and preserve pair-union masks.
    for i in range(p - 1):
        positions = np.flatnonzero(ii == i)
        js = jj[positions]
        with np.errstate(over="ignore", invalid="ignore"):
            for pos, j in zip(positions, js, strict=True):
                keep = ~np.isin(allowed, (*targets[i], *targets[j]))
                ng[pos] = keep.sum()
                if keep.any():
                    value = float(np.sum((truth[i, keep] - truth[j, keep]) ** 2))
                    if np.isfinite(value):
                        rd[pos] = value
            values = np.sum((z[js] - z[i]) ** 2, axis=1)
        zd[positions] = np.where(np.isfinite(values), values, np.nan)
    return Geometry(ids, allowed, targets, truth, pearson, mse, reasons, genes, ii, jj, rd, zd, ng)


def _mean(values):
    finite = np.isfinite(values)
    return dict(
        mean=float(np.sum(values[finite] / finite.sum())) if finite.any() else None,
        finite_N=int(finite.sum()),
        NA_N=int((~finite).sum()),
    )


def _rsa(g):
    finite = np.isfinite(g.response_distance) & np.isfinite(g.cls_distance)
    value = (
        weighted_spearman(g.response_distance[finite], g.cls_distance[finite])
        if finite.any()
        else None
    )
    return dict(
        Spearman=value,
        finite_pairs=int(finite.sum()),
        NA_pairs=int((~finite).sum()),
        original_condition_N=len(g.ids),
        remaining_genes_min=int(g.pair_genes.min()),
        remaining_genes_max=int(g.pair_genes.max()),
        NA_reason="constant_or_insufficient_finite_RDM" if value is None else None,
    )


def summarize_endpoint(**arrays):
    """Return only small condition-balanced aggregates and per-condition scalars."""
    g = _geometry(**arrays)
    return dict(
        condition_N=len(g.ids),
        allowed_gene_N=len(g.allowed),
        ownmask_input_delta_Pearson=_mean(g.pearson),
        ownmask_prediction_delta_MSE=_mean(g.mse),
        response_CLS_RSA=_rsa(g),
        per_condition=[
            dict(
                index=i,
                condition_digest=hashlib.sha256(p.encode()).hexdigest(),
                remaining_genes=int(g.genes[i]),
                ownmask_input_delta_Pearson=float(g.pearson[i])
                if np.isfinite(g.pearson[i])
                else None,
                ownmask_prediction_delta_MSE=float(g.mse[i]) if np.isfinite(g.mse[i]) else None,
                Pearson_NA_reason=g.reasons[i],
                MSE_NA_reason=None if np.isfinite(g.mse[i]) else "empty_or_nonfinite_error",
            )
            for i, p in enumerate(g.ids)
        ],
    )


def _interval(values, point):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    return dict(
        candidate_minus_reference=point,
        CI95=np.quantile(values[finite], [0.025, 0.975]).tolist() if finite.any() else None,
        valid_draws=int(finite.sum()),
        invalid_draws=int((~finite).sum()),
    )


def compare_endpoints(reference, candidate, *, draws=10000, seed=20261028):
    """Paired condition bootstrap; same original-ID pairs are excluded, never rerolled."""
    if type(draws) is not int or draws < 1:
        raise ValueError("positive integer bootstrap draws required")
    r, c = _geometry(**reference), _geometry(**candidate)
    if (
        r.ids != c.ids
        or not np.array_equal(r.allowed, c.allowed)
        or r.targets != c.targets
        or not np.array_equal(r.truth, c.truth, equal_nan=True)
    ):
        raise ValueError("paired condition/allowed/target/truth identity differs")
    # No finite-set shrinking: all original conditions/pairs required for primary support.
    complete_p = bool(np.isfinite(r.pearson).all() and np.isfinite(c.pearson).all())
    complete_m = bool(np.isfinite(r.mse).all() and np.isfinite(c.mse).all())
    finite = (
        np.isfinite(r.response_distance) & np.isfinite(r.cls_distance) & np.isfinite(c.cls_distance)
    )
    complete_r = bool(finite.all())
    ranks = [
        WeightedRanks(v[finite]) for v in (r.response_distance, r.cls_distance, c.cls_distance)
    ]
    ii, jj = r.pair_i[finite], r.pair_j[finite]
    distribution = np.full((draws, 3), np.nan)
    rng = np.random.Generator(np.random.PCG64(seed))
    for b in range(draws):
        counts = np.bincount(rng.integers(len(r.ids), size=len(r.ids)), minlength=len(r.ids))
        if complete_p:
            distribution[b, 0] = np.dot(counts, c.pearson - r.pearson) / len(r.ids)
        if complete_m:
            distribution[b, 1] = np.dot(counts, c.mse - r.mse) / len(r.ids)
        weights = counts[ii] * counts[jj]
        if complete_r:
            rs = _weighted_rank_correlation(ranks[0], ranks[1], weights)
            cs = _weighted_rank_correlation(ranks[0], ranks[2], weights)
            if rs is not None and cs is not None:
                distribution[b, 2] = cs - rs
    rp, cp = _rsa(r)["Spearman"], _rsa(c)["Spearman"]
    return dict(
        schema="paired-condition-response-geometry-v1",
        draws=draws,
        seed=seed,
        condition_N=len(r.ids),
        resampling_unit=(
            "original condition; pair multiplicity ci*cj; same-original-ID pairs excluded"
        ),
        ownmask_Pearson_delta=_interval(
            distribution[:, 0], float(np.mean(c.pearson - r.pearson)) if complete_p else None
        ),
        ownmask_MSE_delta=_interval(
            distribution[:, 1], float(np.mean(c.mse - r.mse)) if complete_m else None
        ),
        RSA_delta=_interval(
            distribution[:, 2],
            cp - rp if complete_r and cp is not None and rp is not None else None,
        ),
        full_axis_finite=dict(Pearson=complete_p, MSE=complete_m, RSA=complete_r),
        candidate_new_Pearson_NA=int(np.sum(np.isfinite(r.pearson) & ~np.isfinite(c.pearson))),
        conditioned_on_fixed_truth_control_bank=True,
        scientific_complete=False,
    )
