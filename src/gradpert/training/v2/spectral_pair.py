"""Soft effective-rank and positive cosine on two complete Student Globals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class SpectralPairResult:
    rank: Tensor
    cosine: Tensor
    rho: Tensor
    traces: Tensor
    input_count: int
    feature_count: int
    max_rank: int


def _spectral_entropy(eigenvalues: Tensor) -> tuple[Tensor, Tensor]:
    """Ignore only FP64 PSD roundoff; use the continuous zero-mass entropy limit."""
    if not torch.isfinite(eigenvalues).all():
        raise ValueError("nonfinite covariance spectrum")
    scale = eigenvalues.abs().max()
    tolerance = 64 * len(eigenvalues) * torch.finfo(torch.float64).eps * scale
    if (eigenvalues < -tolerance).any():
        raise ValueError("covariance has a materially negative eigenvalue")
    eigenvalues = eigenvalues.clamp_min(0)
    trace = eigenvalues.sum()
    if not torch.isfinite(trace) or trace <= 1e-24:
        raise ValueError("near-zero or nonfinite centered covariance trace")
    probabilities = eigenvalues / trace
    # clamp only the logarithm's argument: exact zero eigenvalues stay zero,
    # never acquire mass or artificial rank, and backward remains finite.
    entropy = -(
        probabilities * probabilities.clamp_min(torch.finfo(torch.float64).tiny).log()
    ).sum()
    return entropy, trace


def spectral_pair(
    first: Tensor, second: Tensor, *, target: float = 0.20, softness: float = 0.25
) -> SpectralPairResult:
    """Compute the preregistered losses, without weights or an update clock.

    Caller supplies one unique (control, condition) population gathered across
    all microbatches and ranks. Covariance centers raw post-RMSNorm Response CLS
    separately for G0/G1, with no row normalization or diagonal jitter. Eigensolver
    uses FP64 eigenvalues only, so repeated eigenvalues need no eigenvector
    derivative. Cosine alone normalizes rows and differentiates both Globals.
    """
    if (
        not isinstance(first, Tensor)
        or not isinstance(second, Tensor)
        or first.ndim != 2
        or first.shape != second.shape
        or first.device != second.device
        or not first.is_floating_point()
        or not second.is_floating_point()
    ):
        raise ValueError("spectral pair requires aligned floating-point [N,D] Globals")
    n, d = first.shape
    if n < 2 or d < 1:
        raise ValueError("spectral pair requires N >= 2 and D >= 1")
    if (
        isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not math.isfinite(target)
        or not 0 < target <= 1
        or isinstance(softness, bool)
        or softness != 0.25
    ):
        raise ValueError("spectral target must be in (0,1] and softness fixed at .25")
    max_rank = min(d, n - 1)
    rhos, traces, units = [], [], []
    with torch.autocast(device_type=first.device.type, enabled=False):
        for values in (first, second):
            x = values.double()
            if not torch.isfinite(x).all():
                raise ValueError("spectral pair requires finite Globals")
            centered = x - x.mean(dim=0, keepdim=True)
            covariance = centered.T @ centered / n
            covariance = (covariance + covariance.T) * 0.5
            entropy, trace = _spectral_entropy(torch.linalg.eigvalsh(covariance))
            rhos.append(entropy.exp() / max_rank)
            traces.append(trace)
            norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
            if not torch.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("positive cosine requires finite nonzero row norms")
            units.append(x / norms)
        rho = torch.stack(rhos)
        rank = (softness * F.softplus((math.log(target) - rho.log()) / softness)).mean()
        cosine = (1 - (units[0] * units[1]).sum(-1)).mean()
    return SpectralPairResult(rank, cosine, rho, torch.stack(traces), n, d, max_rank)
