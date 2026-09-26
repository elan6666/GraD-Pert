"""Portable backend selection preserves historical configuration and CPU math."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Architecture, V2Options
from gradpert.modeling.v2.operators import ManifoldResidual
from gradpert.modeling.v2.sinkhorn_fused import resolve_sinkhorn_backend


def test_native_identity_and_unsupported_auto_fallback():
    assert "sinkhorn_backend" not in V2Architecture().payload()
    for device in ("cpu", "mps"):
        assert resolve_sinkhorn_backend("auto", device, 4) == "native"
        with pytest.raises(ValueError):
            resolve_sinkhorn_backend("triton", device, 4)
    assert resolve_sinkhorn_backend("auto", "cuda", 3) == "native"
    with pytest.raises(ValueError):
        V2Architecture(sinkhorn_backend="unknown")


def test_cpu_auto_matches_native_maps_and_gradients():
    torch.manual_seed(12)
    layer = ManifoldResidual(8, 4, torch.nn.Identity())
    x = torch.randn(2, 7, 4, 8, requires_grad=True)
    original = layer.maps(x)
    layer.sinkhorn_backend = "auto"
    actual = layer.maps(x)
    assert layer.resolved_sinkhorn_backend == "native"
    for a, b in zip(original, actual, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    inputs = (x, *layer.parameters())
    a = torch.autograd.grad(sum(t.square().sum() for t in original), inputs, allow_unused=True)
    b = torch.autograd.grad(sum(t.square().sum() for t in actual), inputs, allow_unused=True)
    for left, right in zip(a, b, strict=True):
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_new_configs_only_select_execution_backend():
    root = Path(__file__).resolve().parents[2] / "configs/v2"
    for path in (root / "optimized_single_pass_jurkat").rglob("*.yaml"):
        old = load_experiment_config(
            root / "single_pass_jurkat" / path.relative_to(root / "optimized_single_pass_jurkat")
        )
        new = load_experiment_config(path)
        a, oa = V2Options.parse_parameters(old.model.parameters)
        b, ob = V2Options.parse_parameters(new.model.parameters)
        assert b.sinkhorn_backend == "auto" and replace(b, sinkhorn_backend="native") == a
        assert oa == ob
        left, right = old.model_dump(), new.model_dump()
        right["model"]["parameters"].pop("sinkhorn_backend")
        assert left == right
