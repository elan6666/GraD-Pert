"""Distinguish uniform outputs, one-prototype collapse, and sample diversity."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

path = Path(__file__).parents[2] / "scripts/v2/prototype_diagnostics.py"
spec = importlib.util.spec_from_file_location("prototype_diagnostics", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_uniform_is_high_entropy_but_not_sample_dependent():
    stats = module.Distribution(8)
    stats.add(torch.zeros(6, 8))
    result = stats.result()
    assert result["mean_sample_entropy"] == pytest.approx(math.log(8))
    assert result["effective_marginal_prototypes"] == pytest.approx(8)
    assert result["sample_dependence_js"] == pytest.approx(0, abs=1e-12)


def test_one_prototype_collapse_has_low_marginal_and_sample_entropy():
    stats = module.Distribution(8)
    logits = torch.full((6, 8), -100.0)
    logits[:, 3] = 100
    stats.add(logits)
    result = stats.result()
    assert result["mean_sample_entropy"] < 1e-6
    assert result["effective_marginal_prototypes"] == pytest.approx(1)
    assert result["largest_argmax_fraction"] == 1
    assert result["sample_dependence_js"] < 1e-6


def test_sample_specific_sharp_outputs_have_nonzero_js_and_streaming_parity():
    x = torch.eye(8) * 200
    full, streamed = module.Distribution(8), module.Distribution(8)
    full.add(x)
    streamed.add(x[:3])
    streamed.add(x[3:])
    result = full.result()
    assert result["sample_dependence_js"] == pytest.approx(math.log(8))
    assert result["argmax_prototypes_used"] == 8
    for name, value in result.items():
        if isinstance(value, float):
            assert value == pytest.approx(streamed.result()[name], abs=1e-12)
        else:
            assert value == streamed.result()[name]


def test_constant_features_have_zero_variance_and_unit_cosine():
    stats = module.Features()
    stats.add(torch.ones(10, 4))
    result = stats.result()
    assert result["total_centered_variance"] == 0
    assert result["mean_pair_cosine"] == pytest.approx(1)


def test_feature_covariance_streaming_and_effective_rank():
    x = torch.cat([torch.eye(4), -torch.eye(4)])
    stats = module.Features()
    stats.add(x[:3])
    stats.add(x[3:])
    result = stats.result()
    assert result["covariance_effective_rank"] == pytest.approx(4)
    assert result["relative_centered_variance"] == pytest.approx(1)
