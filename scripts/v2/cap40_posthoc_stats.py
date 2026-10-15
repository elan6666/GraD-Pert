"""Streaming, grouping-aware statistics for immutable-checkpoint diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def seeded_int(value: Any, seed: int) -> int:
    return int(digest_json([seed, value])[:15], 16)


class Moments:
    """Mergeable float64 centered covariance; never retain sample vectors."""

    def __init__(self, width: int) -> None:
        self.n = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros((width, width), dtype=np.float64)
        self.norm_sum = 0.0

    def add(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64).reshape(-1, len(self.mean))
        if not np.isfinite(x).all():
            raise ValueError("nonfinite feature")
        if not len(x):
            return
        mean = x.mean(0)
        centered = x - mean
        self.merge(len(x), mean, centered.T @ centered, float(np.linalg.norm(x, axis=1).sum()))

    def merge(self, n: int, mean: np.ndarray, m2: np.ndarray, norm_sum: float) -> None:
        if not n:
            return
        total = self.n + n
        delta = mean - self.mean
        self.m2 += m2 + np.outer(delta, delta) * (self.n * n / total)
        self.mean += delta * (n / total)
        self.n = total
        self.norm_sum += norm_sum

    def result(self) -> dict:
        if self.n < 2:
            return {
                "rows": self.n,
                "effective_rank": None,
                "relative_variance": None,
                "status": "undefined:fewer_than_two_rows",
            }
        cov = self.m2 / self.n
        eig = np.linalg.eigvalsh((cov + cov.T) * 0.5).clip(0)
        trace = float(eig.sum())
        energy = trace + float(self.mean @ self.mean)
        p = eig / trace if trace > max(energy, 1e-30) * 1e-12 else np.zeros_like(eig)
        nz = p[p > 0]
        rank = float(np.exp(-(nz * np.log(nz)).sum())) if len(nz) else 0.0
        return {
            "rows": self.n,
            "width": len(self.mean),
            "effective_rank": rank,
            "rank_ceiling": min(len(self.mean), self.n - 1),
            "relative_variance": trace / max(energy, 1e-30),
            "variance": trace,
            "mean_norm": self.norm_sum / self.n,
            "status": "ok",
        }


class Distribution:
    """Accumulate a generalized JS, not a mean of chunk-level JS values."""

    def __init__(self, width: int) -> None:
        self.n = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.hits = np.zeros(width, dtype=np.int64)
        self.entropy_sum = self.top1_sum = self.logit_std_sum = 0.0

    def add(self, logits: np.ndarray) -> None:
        x = np.asarray(logits, dtype=np.float64).reshape(-1, len(self.sum))
        if not np.isfinite(x).all():
            raise ValueError("nonfinite logits")
        if not len(x):
            return
        a = x - x.max(1, keepdims=True)
        p = np.exp(a)
        p /= p.sum(1, keepdims=True)
        self.n += len(x)
        self.sum += p.sum(0)
        self.entropy_sum += float(-(p * np.log(p.clip(1e-300))).sum())
        self.hits += np.bincount(p.argmax(1), minlength=len(self.sum))
        self.top1_sum += float(p.max(1).sum())
        self.logit_std_sum += float(x.std(1).sum())

    def result(self) -> dict:
        if not self.n:
            return {"rows": 0, "status": "undefined:no_valid_rows"}
        p = self.sum / self.n
        hm = float(-(p * np.log(p.clip(1e-300))).sum())
        h = self.entropy_sum / self.n
        return {
            "rows": self.n,
            "H_sample": h,
            "H_marginal": hm,
            "JS": max(0.0, hm - h),
            "N_eff": math.exp(hm),
            "uniform_entropy": math.log(len(p)),
            "mean_top1": self.top1_sum / self.n,
            "argmax_used": int((self.hits > 0).sum()),
            "largest_argmax_fraction": float(self.hits.max() / self.n),
            "scaled_logit_std": self.logit_std_sum / self.n,
            "status": "ok",
        }


def condition_summary(rows: list[dict], field: str, *, seed: int = 42) -> dict:
    """Bootstrap conditions, never treat tokens/views as independent cells."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            grouped[str(row["condition_id"])].append(float(value))
    means = np.array([np.mean(grouped[k]) for k in sorted(grouped)])
    if not len(means):
        return {"conditions": 0, "condition_mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    ci = None
    if len(means) > 1:
        boot = np.array([rng.choice(means, len(means), replace=True).mean() for _ in range(1000)])
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
    return {
        "conditions": len(means),
        "condition_mean": float(means.mean()),
        "row_mean": float(np.mean([v for vs in grouped.values() for v in vs])),
        "ci95": ci,
        "bootstrap_unit": "perturbation_condition",
    }


def select_balanced(records: list[dict], limit: int, seed: int) -> list[dict]:
    """Round-robin contexts, hash-randomized within each context, no replacement."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[row["context_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: seeded_int(r["truth_row_id"], seed))
    selected = []
    for rank in range(max(map(len, groups.values()), default=0)):
        for context in sorted(groups):
            if rank < len(groups[context]):
                selected.append(groups[context][rank])
                if len(selected) == limit:
                    return selected
    return selected


def query_panels(allowed: Iterable[int], count: int, seed: int) -> list[list[int]]:
    """A common anchor plus rotating panels that cover every allowed identity."""
    pool = np.array(sorted(set(allowed)), dtype=np.int64)
    if len(pool) < count or count < 2:
        raise ValueError("invalid query budget")
    shuffled = np.random.default_rng(seed).permutation(pool)
    anchor_count = min(256, count // 2)
    anchor, remaining = shuffled[:anchor_count], shuffled[anchor_count:]
    width = count - anchor_count
    result = []
    for start in range(0, len(remaining), width):
        panel = remaining[(np.arange(width) + start) % len(remaining)]
        result.append(sorted(map(int, np.concatenate((anchor, panel)))))
    if set().union(*(set(p) for p in result)) != set(pool):
        raise AssertionError("query coverage incomplete")
    return result


def cell_balanced_terms(values: np.ndarray, row_ids: np.ndarray, cells: int) -> np.ndarray:
    """Mean tokens inside each valid cell; omit cells with no masked tokens."""
    values = np.asarray(values, dtype=np.float64)
    ids = np.asarray(row_ids, dtype=np.int64)
    if len(values) != len(ids) or (ids < 0).any() or (ids >= cells).any():
        raise ValueError("invalid token-to-cell alignment")
    sums = np.zeros((cells, values.shape[1]), dtype=np.float64)
    np.add.at(sums, ids, values)
    counts = np.bincount(ids, minlength=cells)
    valid = counts > 0
    return sums[valid] / counts[valid, None]
