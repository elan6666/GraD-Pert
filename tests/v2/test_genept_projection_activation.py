"""Nonlinear shared projection and historical linear checkpoint compatibility."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from test_relay_method import small_architecture, small_index

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Architecture, V2Options
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "configs/v2/cap40_unseen_u4_20261009/U4/gradpert_v2/nadig_jurkat.yaml"
NEW = ROOT / "configs/v2/cap40_genept_gelu_20261010/gradpert_v2/nadig_jurkat.yaml"


def make_model(raw, activation):
    torch.manual_seed(291)
    return GraDPertV2(
        raw,
        replace(
            small_architecture(),
            relay_passes=1,
            learned_genept_projection=True,
            genept_projection_activation=activation,
        ),
    )


def test_graph_receives_gelu_after_shared_projection_before_layernorm():
    raw = torch.randn(9, 16)
    linear = make_model(raw, "none").eval()
    rng = torch.get_rng_state().clone()
    nonlinear = make_model(raw, "gelu").eval()
    assert torch.equal(torch.get_rng_state(), rng)
    assert linear.state_dict().keys() == nonlinear.state_dict().keys()
    for name, value in linear.state_dict().items():
        torch.testing.assert_close(value, nonlinear.state_dict()[name], atol=0, rtol=0)
    assert sum(p.numel() for p in linear.parameters()) == sum(
        p.numel() for p in nonlinear.parameters()
    )
    observed = []
    handle = nonlinear.graph.norm.register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach().clone())
    )
    view = small_index().view(
        np.array([7, 2, 5]),
        [(7,)],
        rng=np.random.default_rng(3),
        device=torch.device("cpu"),
        induced=False,
    )
    graph = nonlinear.graph(
        view.ids, view.neighbors, view.valid, view.sources, context=view.context
    )
    handle.remove()
    expected = F.gelu(raw @ nonlinear.graph.adapter.weight.T + nonlinear.graph.adapter.bias)
    torch.testing.assert_close(observed[0], expected)
    assert not torch.allclose(observed[0], linear.graph.adapter(raw))
    graph.square().mean().backward()
    grad = nonlinear.graph.adapter.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert nonlinear.graph.embedding.weight.grad is None


def test_nonlinear_projection_has_explicit_identity_and_rejects_linear_resume(tmp_path):
    raw = torch.randn(9, 16)
    linear = JointObjective(make_model(raw, "none"))
    nonlinear = JointObjective(make_model(raw, "gelu"))
    assert "genept_projection_activation" not in linear.student.options.payload()
    assert nonlinear.student.options.payload()["genept_projection_activation"] == "gelu"
    path = tmp_path / "linear.pt"
    rng = np.random.default_rng(4)
    save_checkpoint(
        path, linear, V2Optimizer(linear.student, 1e-3, 0), identity={}, progress={}, generator=rng
    )
    with pytest.raises(ValueError, match="architecture identity mismatch"):
        load_checkpoint(
            path, nonlinear, V2Optimizer(nonlinear.student, 1e-3, 0), identity={}, generator=rng
        )


def test_activation_validation_and_new_config_preserve_all_other_u4_settings():
    with pytest.raises(ValueError, match="unknown GenePT"):
        V2Architecture(genept_projection_activation="relu")
    with pytest.raises(ValueError, match="requires learned"):
        V2Architecture(genept_projection_activation="gelu")
    old = yaml.safe_load(OLD.read_text())
    new = yaml.safe_load(NEW.read_text())
    assert new["model"]["parameters"].pop("genept_projection_activation")["value"] == "gelu"
    assert old == new
    old_arch, _ = V2Options.parse_parameters(load_experiment_config(OLD).model.parameters)
    new_arch, _ = V2Options.parse_parameters(load_experiment_config(NEW).model.parameters)
    assert old_arch.genept_projection_activation == "none"
    assert new_arch.genept_projection_activation == "gelu"
    assert replace(new_arch, genept_projection_activation="none").payload() == old_arch.payload()
