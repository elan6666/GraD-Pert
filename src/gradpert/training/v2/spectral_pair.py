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


@dataclass(frozen=True)
class SpectralAmplitudePairResult(SpectralPairResult):
    amplitude: Tensor
    q: Tensor
    total_energies: Tensor
    mean_energies: Tensor


def _relative_variance(x: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Raw centered/total energy ratio, with both denominator gradients retained."""
    trace = covariance.diagonal().sum()
    total_energy = x.square().sum(-1).mean()
    mean_energy = x.mean(0).square().sum()
    if not torch.isfinite(total_energy) or total_energy <= 1e-24:
        raise ValueError("near-zero or nonfinite total raw energy")
    q = trace / total_energy
    tolerance = 64 * x.shape[-1] * torch.finfo(torch.float64).eps
    if not torch.isfinite(q) or q < -tolerance or q > 1 + tolerance:
        raise ValueError("relative centered variance outside [0,1] beyond FP64 roundoff")
    return q.clamp(0, 1), total_energy, mean_energy


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
    first: Tensor,
    second: Tensor,
    *,
    target: float = 0.20,
    softness: float = 0.25,
    amplitude_target: float | None = None,
) -> SpectralPairResult:
    """Compute the preregistered losses, without weights or an update clock.

    Caller supplies one unique (control, condition) population gathered across
    all microbatches and ranks. Covariance centers raw post-RMSNorm Response CLS
    separately for G0/G1, with no row normalization or diagonal jitter. Eigensolver
    uses FP64 eigenvalues only, so repeated eigenvalues need no eigenvector
    derivative. Cosine alone normalizes rows and differentiates both Globals.
    Optional amplitude is a linear hinge on sqrt(centered_trace/total_energy +
    1e-8); it reuses these raw Globals and keeps total-energy gradients. None
    preserves the historical computation and result fields exactly. Rank still
    rejects a near-zero centered trace, including an exactly constant population.
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
    if amplitude_target is not None and (
        isinstance(amplitude_target, bool)
        or not isinstance(amplitude_target, (int, float))
        or not math.isfinite(amplitude_target)
        or not 0 < amplitude_target <= 1
    ):
        raise ValueError("spectral amplitude target must be finite and in (0,1]")
    max_rank = min(d, n - 1)
    rhos, traces, units = [], [], []
    ratios, total_energies, mean_energies = [], [], []
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
            if amplitude_target is not None:
                q, total_energy, mean_energy = _relative_variance(x, covariance)
                ratios.append(q)
                total_energies.append(total_energy)
                mean_energies.append(mean_energy)
            norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
            if not torch.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("positive cosine requires finite nonzero row norms")
            units.append(x / norms)
        rho = torch.stack(rhos)
        rank = (softness * F.softplus((math.log(target) - rho.log()) / softness)).mean()
        cosine = (1 - (units[0] * units[1]).sum(-1)).mean()
        if amplitude_target is not None:
            relative_variances = torch.stack(ratios)
            amplitude = F.relu(
                math.sqrt(amplitude_target) - torch.sqrt(relative_variances + 1e-8)
            ).mean()
            return SpectralAmplitudePairResult(
                rank,
                cosine,
                rho,
                torch.stack(traces),
                n,
                d,
                max_rank,
                amplitude,
                relative_variances,
                torch.stack(total_energies),
                torch.stack(mean_energies),
            )
    return SpectralPairResult(rank, cosine, rho, torch.stack(traces), n, d, max_rank)
