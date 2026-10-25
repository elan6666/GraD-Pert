"""Analytic spectra, invariants and derivatives of the preregistered pair."""

import math

import pytest
import torch
from torch.nn import functional as F

from gradpert.training.v2.spectral_pair import _spectral_entropy, spectral_pair


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
