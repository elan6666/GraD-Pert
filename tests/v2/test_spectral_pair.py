"""Analytic spectra, invariants and derivatives of the preregistered pair."""

import math

import pytest
import torch
from torch.nn import functional as F

from gradpert.training.v2.spectral_pair import (
    SpectralAmplitudePairResult,
    SpectralPairResult,
    _relative_variance,
    _spectral_entropy,
    spectral_pair,
)


def test_population_covariance_repeated_full_rank_analytic_value():
    x = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.float64)
    result = spectral_pair(x, x, target=0.35)
    torch.testing.assert_close(result.rho, torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(result.traces, torch.ones(2, dtype=torch.float64))
    assert result.rank.item() == pytest.approx(0.25 * math.log1p(0.35**4))
    assert result.cosine.item() == 0
    assert (result.input_count, result.feature_count, result.max_rank) == (4, 2, 2)


def test_rank_one_zero_eigenvalues_have_no_artificial_mass_and_finite_gradients():
    x = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    y = (x.detach() + torch.tensor([0.0, 1.0, 0.0])).requires_grad_()
    result = spectral_pair(x, y, target=0.35)
    torch.testing.assert_close(result.rho, torch.full((2,), 1 / 3, dtype=torch.float64))
    grads = torch.autograd.grad(result.rank + result.cosine, (x, y))
    assert all(torch.isfinite(g).all() for g in grads)


def test_sample_rank_bound_uses_nminus1_not_feature_width():
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    result = spectral_pair(x, x)
    assert result.max_rank == 1
    torch.testing.assert_close(result.rho, torch.ones(2, dtype=torch.float64))


def test_rank_centers_raw_values_and_is_not_row_normalized_or_view_pooled():
    torch.manual_seed(92)
    x, y = torch.randn(7, 4, dtype=torch.float64), torch.randn(7, 4, dtype=torch.float64)
    result = spectral_pair(x, y)
    expected = []
    for z in (x, y):
        centered = z - z.mean(0)
        eigen = torch.linalg.svdvals(centered).square() / len(z)
        p = eigen / eigen.sum()
        expected.append((-(p * p.log()).sum()).exp() / 4)
    torch.testing.assert_close(result.rho, torch.stack(expected))
    shifted = spectral_pair(x + 25, 3 * y - 13)
    torch.testing.assert_close(shifted.rho, result.rho)
    torch.testing.assert_close(shifted.traces[1], result.traces[1] * 9)
    row_scaled = spectral_pair(x * torch.arange(1.0, 8.0).unsqueeze(1), y)
    assert abs(float(row_scaled.rho[0] - result.rho[0])) > 0.01
    torch.testing.assert_close(result.cosine, (1 - F.cosine_similarity(x, y)).mean())
    micro = (spectral_pair(x[:3], y[:3]).rank + spectral_pair(x[3:], y[3:]).rank) / 2
    assert abs(float(micro - result.rank)) > 1e-4


def test_both_branch_gradients_match_finite_differences_and_repeated_spectrum():
    torch.manual_seed(8)
    x = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: spectral_pair(a, b, target=0.35).rank, (x, y))
    assert torch.autograd.gradcheck(lambda a, b: spectral_pair(a, b).cosine, (x, y))
    repeated = torch.cat((torch.eye(3), -torch.eye(3))).double().requires_grad_()
    result = spectral_pair(repeated, y)
    for term in (result.rank, result.cosine):
        grads = torch.autograd.grad(term, (repeated, y), retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grads)
        assert grads[1].norm() > 0
    # Repeated positive eigenvalues need no eigenvector derivative.
    assert torch.autograd.gradcheck(lambda a: spectral_pair(a, a).rank, (repeated,))


def test_negative_roundoff_is_clamped_but_material_indefiniteness_fails():
    entropy, trace = _spectral_entropy(torch.tensor([-1e-15, 0.0, 1.0], dtype=torch.float64))
    assert entropy.item() == 0 and trace.item() == 1
    with pytest.raises(ValueError, match="materially negative"):
        _spectral_entropy(torch.tensor([-0.01, 0.0, 1.0], dtype=torch.float64))


@pytest.mark.parametrize("x", [torch.ones(4, 3), torch.ones(4, 3) * 1e-20])
def test_zero_or_near_zero_centered_trace_rejected(x):
    with pytest.raises(ValueError, match="trace"):
        spectral_pair(x, x)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_rejected(bad):
    x = torch.eye(3)
    x[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        spectral_pair(x, x)


def test_zero_row_norm_rejected_and_outer_autocast_cannot_narrow_eigensolve():
    x = torch.eye(3, dtype=torch.bfloat16).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = spectral_pair(x, x)
    assert result.rho.dtype == torch.float64
    result.rank.backward()
    assert torch.isfinite(x.grad).all()
    zero_row = torch.eye(3)
    zero_row[0] = 0
    with pytest.raises(ValueError, match="row norms"):
        spectral_pair(zero_row, zero_row)


def test_amplitude_analytic_relative_energy_and_fixed_epsilon():
    x = torch.tensor([[3.0, 1], [3, -1], [4, 1], [4, -1]], dtype=torch.float64)
    result = spectral_pair(x, 2 * x, amplitude_target=0.20)
    assert isinstance(result, SpectralAmplitudePairResult)
    torch.testing.assert_close(result.q, torch.full((2,), 1.25 / 13.5, dtype=torch.float64))
    torch.testing.assert_close(result.total_energies, torch.tensor([13.5, 54.0]).double())
    torch.testing.assert_close(result.mean_energies, torch.tensor([12.25, 49.0]).double())
    torch.testing.assert_close(result.traces, result.total_energies - result.mean_energies)
    assert result.amplitude.item() == pytest.approx(math.sqrt(0.20) - math.sqrt(1.25 / 13.5 + 1e-8))
    legacy = spectral_pair(x, 2 * x)
    assert type(legacy) is SpectralPairResult
    assert not hasattr(legacy, "amplitude")


def test_amplitude_gradcheck_both_globals_and_analytic_denominator_derivative():
    torch.manual_seed(18)
    x = (torch.randn(6, 3, dtype=torch.float64) + 8).requires_grad_()
    y = (torch.randn(6, 3, dtype=torch.float64) - 9).requires_grad_()

    def term(a, b):
        result = spectral_pair(a, b, amplitude_target=0.0225)
        assert isinstance(result, SpectralAmplitudePairResult)
        return result.amplitude

    assert torch.autograd.gradcheck(term, (x, y))
    result = spectral_pair(x, y, amplitude_target=0.0225)
    assert isinstance(result, SpectralAmplitudePairResult)
    gradients = torch.autograd.grad(result.amplitude, (x, y))
    for i, (z, gradient) in enumerate(zip((x, y), gradients, strict=True)):
        centered = z - z.mean(0)
        energy = z.square().sum(-1).mean()
        variance = centered.square().sum(-1).mean()
        dq = 2 * (centered * energy - variance * z) / (len(z) * energy.square())
        expected = -dq / (4 * (result.q[i] + 1e-8).sqrt())
        torch.testing.assert_close(gradient, expected, atol=2e-15, rtol=2e-12)
        assert gradient.norm() > 0
        # Detaching E would remove this exact scale-invariance orthogonality.
        assert abs(float((gradient * z).sum().detach())) < 1e-14


def test_amplitude_uniform_scale_invariance_and_shared_mean_sensitivity():
    torch.manual_seed(19)
    x, y = (torch.randn(7, 4, dtype=torch.float64) + 5 for _ in range(2))
    original = spectral_pair(x, y, amplitude_target=0.20)
    scaled = spectral_pair(0.03 * x, -7 * y, amplitude_target=0.20)
    shifted = spectral_pair(x + 20, y + 20, amplitude_target=0.20)
    assert isinstance(original, SpectralAmplitudePairResult)
    assert isinstance(scaled, SpectralAmplitudePairResult)
    assert isinstance(shifted, SpectralAmplitudePairResult)
    torch.testing.assert_close(original.q, scaled.q, rtol=2e-13, atol=2e-15)
    torch.testing.assert_close(original.amplitude, scaled.amplitude, rtol=2e-13, atol=2e-15)
    torch.testing.assert_close(original.traces, shifted.traces, rtol=2e-13, atol=2e-13)
    assert (shifted.q < original.q).all()
    assert shifted.amplitude > original.amplitude


def test_tiny_variance_epsilon_limit_does_not_invent_amplitude_or_rank_recovery():
    x = (torch.ones(4, 2, dtype=torch.float64) + 1e-7 * torch.eye(4, 2)).requires_grad_()
    result = spectral_pair(x, x, amplitude_target=0.0225)
    assert isinstance(result, SpectralAmplitudePairResult)
    assert (result.q < 1e-8).all()
    expected = (math.sqrt(0.0225) - (result.q + 1e-8).sqrt()).mean()
    torch.testing.assert_close(result.amplitude, expected, rtol=0, atol=0)
    gradient = torch.autograd.grad(result.amplitude, x)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    with pytest.raises(ValueError, match="trace"):
        spectral_pair(torch.ones_like(x), torch.ones_like(x), amplitude_target=0.0225)


def test_inactive_amplitude_hinge_exact_zero_and_zero_gradient():
    x = torch.cat((torch.eye(3), -torch.eye(3))).double().requires_grad_()
    result = spectral_pair(x, x, amplitude_target=0.0225)
    assert isinstance(result, SpectralAmplitudePairResult)
    assert result.amplitude.item() == 0
    torch.testing.assert_close(torch.autograd.grad(result.amplitude, x)[0], torch.zeros_like(x))


@pytest.mark.parametrize("target", [True, -0.1, 0.0, 1.01, float("nan"), float("inf")])
def test_amplitude_rejects_invalid_target(target):
    with pytest.raises(ValueError, match="amplitude target"):
        spectral_pair(torch.eye(3), torch.eye(3), amplitude_target=target)


def test_relative_variance_rejects_invalid_energy_or_ratio_except_fp64_roundoff():
    x = torch.ones(4, 3, dtype=torch.float64)
    for tiny in (0.0, 1e-13, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="total raw energy"):
            _relative_variance(x * tiny, torch.eye(3, dtype=torch.float64))
    for value in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="outside"):
            _relative_variance(x, value * torch.eye(3, dtype=torch.float64))
    q, _, _ = _relative_variance(x, -1e-16 * torch.eye(3, dtype=torch.float64))
    assert q.item() == 0
    q, _, _ = _relative_variance(x, (1 + 1e-15) * torch.eye(3, dtype=torch.float64))
    assert q.item() == 1


def test_active_amplitude_outer_autocast_remains_fp64_and_short_population_bound():
    x = (torch.eye(3, 5) + 3).bfloat16().requires_grad_()
    y = (torch.eye(3, 5) - 4).bfloat16().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = spectral_pair(x, y, amplitude_target=0.20)
    assert isinstance(result, SpectralAmplitudePairResult)
    assert result.amplitude.dtype == result.q.dtype == torch.float64
    assert result.max_rank == 2
    gradients = torch.autograd.grad(result.amplitude, (x, y))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
