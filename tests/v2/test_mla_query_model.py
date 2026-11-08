"""Graph-only normalization scope, real updates and resumable synthetic training."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_restored_ffn import restored
from torch import nn

from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer


def build(rank):
    parent, batch = restored("BR")
    architecture = replace(
        parent.student.options,
        ffn_variant="gelu",
        mla_query_rank=rank,
        relay_validate_once=True,
        cache_kda_constants=True,
    )
    seeds = parent.student.graph.embedding.weight.detach().clone()
    torch.manual_seed(131)
    model = GraDPertV2(seeds, architecture)
    rng = torch.get_rng_state().clone()
    return JointObjective(model, ssl1_weights=(1, 1, 0), ssl2_weights=(1, 1, 0)), batch, rng


def test_only_dense_latent_queries_change_and_common_initialization_rng_match():
    from gradpert.modeling.v2.operators import LatentAttention

    base, _, before = build(0)
    candidate, _, after = build(4)
    assert torch.equal(before, after)
    paths = {
        name for name, m in candidate.student.named_modules() if isinstance(m, LatentAttention)
    }
    assert len(paths) == 3
    old, new = base.student.state_dict(), candidate.student.state_dict()
    qkeys = {p + ".query.weight" for p in paths}
    assert set(old) - set(new) == qkeys
    assert set(new) - set(old) == {
        p + ".query." + suffix for p in paths for suffix in ["0.weight", "1.weight", "2.weight"]
    }
    for name in set(old) & set(new):
        torch.testing.assert_close(old[name], new[name], atol=0, rtol=0)
    for name, module in candidate.student.named_modules():
        if name in paths:
            assert isinstance(module.query, nn.Sequential)
            assert module.query[1].eps == 1e-6
            x = torch.randn(2, 5, module.query[0].in_features, requires_grad=True)
            module.query(x).square().sum().backward()
            assert torch.isfinite(x.grad).all()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in module.query.parameters()
            )
    assert all(isinstance(layer.norm1, nn.LayerNorm) for layer in candidate.student.graph.layers)
    assert "mla_query_rank" not in base.student.options.payload()
    assert candidate.student.options.payload()["mla_query_rank"] == 4


def test_compressed_query_optimizer_roles_preserve_headwise_expansion():
    from gradpert.training.v2.optimizer import routes

    objective, _, _ = build(4)
    qroutes = [r for r in routes(objective.student) if ".query." in r["name"]]
    for route in qroutes:
        suffix = route["name"].rsplit(".query.", 1)[1]
        if suffix == "0.weight":
            assert (route["optimizer"], route["heads"]) == ("muon", 1)
        elif suffix == "2.weight":
            assert (route["optimizer"], route["heads"]) == ("muon", objective.student.options.heads)
        elif suffix == "1.weight":
            assert (route["optimizer"], route["heads"]) == ("adamw", 1)
    assert sum(r["name"].endswith(".query.2.weight") for r in qroutes) == 3


@pytest.mark.parametrize("norm", [0, 4])
def test_full_updates_and_resume_restore_student_teacher_centers_optimizer_and_rng(norm, tmp_path):
    objective, batch, _ = build(norm)
    optimizer = V2Optimizer(objective.student, 1e-3, 0)
    generator = np.random.default_rng(1)
    initial = {
        name: parameter.detach().clone() for name, parameter in objective.student.named_parameters()
    }
    for _ in range(2):
        losses = optimizer_step(
            objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
        )
        assert all(np.isfinite(value) for value in losses.values())
    assert any(
        not torch.equal(initial[name], parameter)
        for name, parameter in objective.student.named_parameters()
        if parameter.requires_grad
    )
    assert not objective.pending
    assert all(parameter.grad is None for parameter in objective.teacher.parameters())
    checkpoint = tmp_path / "state.pt"
    identity = {"query_rank": norm, "architecture": objective.student.options.payload()}
    save_checkpoint(
        checkpoint,
        objective,
        optimizer,
        identity=identity,
        progress={"step": 2},
        generator=generator,
    )
    expected = optimizer_step(
        objective, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    state = copy.deepcopy(objective.state_dict())
    rng = torch.get_rng_state().clone()
    resumed, _, _ = build(norm)
    resumed_optimizer = V2Optimizer(resumed.student, 1e-3, 0)
    load_checkpoint(checkpoint, resumed, resumed_optimizer, identity=identity, generator=generator)
    actual = optimizer_step(
        resumed, resumed_optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False
    )
    assert actual == expected
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    assert torch.equal(rng, torch.get_rng_state())
