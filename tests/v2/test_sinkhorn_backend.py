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
        parent = (
            root / "single_pass_jurkat" / path.relative_to(root / "optimized_single_pass_jurkat")
        )
        if not parent.exists():
            assert path.relative_to(root / "optimized_single_pass_jurkat").parts[0] in {
                "capacity_m66_a2",
                "capacity_m68_a2",
                "one_epoch_m66_a2",
            }
            # Later capacity/one-epoch recipes have explicit changes beyond
            # the execution backend; audited separately below.
            continue
        old = load_experiment_config(parent)
        new = load_experiment_config(path)
        a, oa = V2Options.parse_parameters(old.model.parameters)
        b, ob = V2Options.parse_parameters(new.model.parameters)
        assert b.sinkhorn_backend == "auto" and replace(b, sinkhorn_backend="native") == a
        assert oa == ob
        left, right = old.model_dump(), new.model_dump()
        right["model"]["parameters"].pop("sinkhorn_backend")
        assert left == right


def test_later_capacity_and_one_epoch_recipes_have_only_declared_overrides():
    root = Path(__file__).resolve().parents[2] / "configs/v2"
    base = load_experiment_config(
        root / "single_pass_jurkat/gradpert_v2/nadig_jurkat.yaml"
    ).model_dump()
    for name, micro, epochs in (
        ("capacity_m66_a2", 66, 5),
        ("capacity_m68_a2", 68, 5),
        ("one_epoch_m66_a2", 66, 1),
    ):
        new = load_experiment_config(
            root / f"optimized_single_pass_jurkat/{name}/gradpert_v2/nadig_jurkat.yaml"
        ).model_dump()
        assert new["model"]["parameters"].pop("sinkhorn_backend")["value"] == "auto"
        assert new["model"]["parameters"]["microbatch"]["value"] == micro
        assert new["training"]["train_batch_size"]["value"] == micro * 4
        assert new["training"]["max_epochs"]["value"] == epochs
        new["model"]["parameters"]["microbatch"] = base["model"]["parameters"]["microbatch"]
        new["training"]["train_batch_size"] = base["training"]["train_batch_size"]
        if epochs == 1:
            assert new["model"]["parameters"]["lambda2"]["value"] == 1
            assert new["training"]["formal_run_policy"] == "v2_fixed_1"
            new["model"]["parameters"]["lambda2"] = base["model"]["parameters"]["lambda2"]
            new["training"]["max_epochs"] = base["training"]["max_epochs"]
            new["training"]["formal_run_policy"] = base["training"]["formal_run_policy"]
        assert new == base
