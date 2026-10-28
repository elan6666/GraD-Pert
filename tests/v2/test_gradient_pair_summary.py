"""Independent support and analytic-gradient checks for scalar diagnostics."""

import importlib.util
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "gradient_pair_summary",
    Path(__file__).parents[2] / "scripts/v2/gradient_pair_summary.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
summarize = _MODULE.summarize_gradient_pair


def test_analytic_scalar_losses_preserve_parameter_grads():
    x = torch.tensor([2.0, -1.0], requires_grad=True)
    left = (x.square().sum(), (x * torch.tensor([-1.0, 3.0])).sum())
    first = torch.autograd.grad(left[0], x)[0]
    second = torch.autograd.grad(left[1], x)[0]
    r = summarize({"x": first}, {"x": second}, auxiliary_scale=0.5)
    assert r["shared_dot"] == pytest.approx(-10.0)
    assert r["prediction_dot_two_branch_sum"] == pytest.approx(15.0)
    assert r["two_branch_sum_l2"] == pytest.approx((12.5) ** 0.5)
    assert x.grad is None


def test_exclusive_heads_do_not_pollute_shared_cosine():
    p = {"shared": torch.tensor([3.0, 4.0]), "decoder": torch.tensor([12.0]), "head": None}
    a = {"shared": torch.tensor([-6.0, -8.0]), "decoder": None, "head": torch.tensor([24.0])}
    r = summarize(p, a, auxiliary_scale=0.1)
    assert r["shared_cosine"] == pytest.approx(-1.0)
    assert r["prediction_l2"] == pytest.approx(13.0)
    assert r["auxiliary_l2"] == pytest.approx(26.0)
    assert r["counts"]["shared_connected"] == 1
    assert r["counts"]["prediction_only"] == r["counts"]["auxiliary_only"] == 1
    assert r["two_branch_sum_l2"] == pytest.approx((165.76) ** 0.5)


def test_zero_is_connected_but_cosine_undefined():
    r = summarize(
        {"x": torch.zeros(2), "y": None}, {"x": torch.ones(2), "y": None}, auxiliary_scale=1
    )
    assert r["shared_cosine"] is None
    assert r["status"] == "zero_norm_on_shared_support"
    assert r["counts"]["shared_connected"] == 1
    assert r["counts"]["prediction_numerically_zero"] == 1
    assert r["counts"]["neither_connected"] == 1


def test_disjoint_support_is_not_a_zero_cosine():
    r = summarize(
        {"x": torch.ones(2), "y": None}, {"x": None, "y": torch.ones(2)}, auxiliary_scale=1
    )
    assert r["status"] == "no_shared_connected_parameters"
    assert r["shared_cosine"] is None
    assert r["two_branch_sum_l2"] == pytest.approx(2.0)


@pytest.mark.parametrize("scale", [-1, float("nan"), float("inf")])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError):
        summarize({"x": torch.ones(1)}, {"x": torch.ones(1)}, auxiliary_scale=scale)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_derivative_rejected(value):
    with pytest.raises(ValueError):
        summarize({"x": torch.tensor([value])}, {"x": torch.ones(1)}, auxiliary_scale=1)


def test_mismatched_population_or_shape_rejected():
    with pytest.raises(ValueError):
        summarize({"x": torch.ones(2)}, {"y": torch.ones(2)}, auxiliary_scale=1)
    with pytest.raises(ValueError):
        summarize({"x": torch.ones(2)}, {"x": torch.ones(3)}, auxiliary_scale=1)


def test_nearly_cancelling_gradients_use_direct_vector_sum():
    p = torch.tensor([1e8, 1], dtype=torch.float64)
    a = torch.tensor([-1e8, 1], dtype=torch.float64)
    result = summarize({"x": p}, {"x": a}, auxiliary_scale=1)
    assert result["two_branch_sum_l2"] == pytest.approx(2.0)
    assert result["prediction_dot_two_branch_sum"] == pytest.approx(2.0)


def test_noncontiguous_and_mixed_dtype_gradients():
    p = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
    a = -p.double()
    assert not p.is_contiguous()
    result = summarize({"x": p}, {"x": a}, auxiliary_scale=1)
    assert result["shared_cosine"] == pytest.approx(-1.0)
    assert result["two_branch_sum_l2"] == 0


def test_finite_inputs_with_overflowing_combination_fail():
    with pytest.raises(FloatingPointError):
        summarize({"x": torch.ones(1)}, {"x": torch.ones(1)}, auxiliary_scale=1e308)


def test_large_parameter_chunk_boundaries():
    p = torch.ones(262145)
    a = p.clone()
    a[-1] = -1
    result = summarize({"x": p}, {"x": a}, auxiliary_scale=1)
    assert result["two_branch_sum_l2"] == pytest.approx(1024.0)
    assert result["prediction_dot_two_branch_sum"] == pytest.approx(524288.0)
