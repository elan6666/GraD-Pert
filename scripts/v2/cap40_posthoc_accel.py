"""Exact-statistic execution alternatives; no model or sample changes.

The reference observer uses NumPy float64 probabilities. Keep that precision
here, reducing each bounded CUDA block before returning statistics to the CPU.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
from cap40_posthoc_stats import Distribution, Moments


def sample_moments(values):
    """Same centered spectrum using the smaller Gram matrix when rows < width."""
    x = np.asarray(values, dtype=np.float64)
    n, d = x.shape
    if n < 2 or n >= d:
        m = Moments(d)
        m.add(x)
        return m.result()
    if not np.isfinite(x).all():
        raise ValueError("nonfinite feature")
    mean = x.mean(0)
    centered = x - mean
    gram = centered @ centered.T / n
    eig = np.linalg.eigvalsh((gram + gram.T) * 0.5).clip(0)
    trace = float(eig.sum())
    energy = trace + float(mean @ mean)
    p = eig / trace if trace > max(energy, 1e-30) * 1e-12 else np.zeros_like(eig)
    nz = p[p > 0]
    return {
        "rows": n,
        "width": d,
        "effective_rank": float(np.exp(-(nz * np.log(nz)).sum())) if len(nz) else 0.0,
        "rank_ceiling": min(d, n - 1),
        "relative_variance": trace / max(energy, 1e-30),
        "variance": trace,
        "mean_norm": float(np.linalg.norm(x, axis=1).mean()),
        "status": "ok",
    }


def reduced_distribution(logits, chunk_rows=128):
    """Return the original Distribution schema without copying row logits.

    Only O(prototypes) sufficient statistics cross the device boundary, once
    per invocation. Inputs are the same already-scaled FP32 logits as reference.
    No RNG, model update, prototype sampling, or approximate probability.
    """
    import torch

    if chunk_rows < 1:
        raise ValueError("positive statistics chunk required")
    width = logits.shape[-1]
    result = Distribution(width)
    flat = logits.detach().reshape(-1, width)
    if not len(flat):
        return result
    prob_sum = torch.zeros(width, device=flat.device, dtype=torch.float64)
    hits = torch.zeros(width, device=flat.device, dtype=torch.int64)
    totals = torch.zeros(3, device=flat.device, dtype=torch.float64)
    finite = torch.ones((), device=flat.device, dtype=torch.bool)
    for block in flat.split(chunk_rows):
        x = block.to(torch.float64)
        finite.logical_and_(torch.isfinite(x).all())
        shifted = x - x.amax(1, keepdim=True)
        p = shifted.exp()
        p = p / p.sum(1, keepdim=True)
        prob_sum.add_(p.sum(0))
        hits.add_(torch.bincount(p.argmax(1), minlength=width))
        totals.add_(
            torch.stack(
                (
                    -(p * p.clamp_min(1e-300).log()).sum(),
                    p.amax(1).sum(),
                    x.std(1, correction=0).sum(),
                )
            )
        )
    packed = torch.cat((prob_sum, hits.to(torch.float64), totals, finite.reshape(1))).cpu()
    values = packed.numpy()
    if not values[-1] or not np.isfinite(values).all():
        raise ValueError("nonfinite logits")
    result.n = len(flat)
    result.sum = values[:width].copy()
    result.hits = values[width : 2 * width].astype(np.int64)
    result.entropy_sum, result.top1_sum, result.logit_std_sum = values[-4:-1]
    return result


class ObserverTimings:
    """CPU wall regions plus profiler labels; nested regions are not additive."""

    def __init__(self):
        self.seconds = {}
        self.calls = {}

    @contextmanager
    def region(self, name):
        import torch

        started = time.perf_counter()
        with torch.profiler.record_function("medium." + name):
            yield
        self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started
        self.calls[name] = self.calls.get(name, 0) + 1

    def result(self):
        return {"cpu_wall_seconds": self.seconds, "calls": self.calls, "nested_not_additive": True}
