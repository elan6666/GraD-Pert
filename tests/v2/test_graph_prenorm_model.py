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


def build(norm):
    parent, batch = restored("BR")
    architecture = replace(
        parent.student.options,
        ffn_variant="gelu",
        graph_input_norm=norm,
        relay_validate_once=True,
        cache_kda_constants=True,
    )
    seeds = parent.student.graph.embedding.weight.detach().clone()
    torch.manual_seed(131)
    model = GraDPertV2(seeds, architecture)
    rng = torch.get_rng_state().clone()
    return JointObjective(model, ssl1_weights=(1, 1, 0), ssl2_weights=(1, 1, 0)), batch, rng


def test_only_graph_entrance_norms_and_their_bias_parameters_change():
    baseline, _, baseline_rng = build("historical")
    candidate, _, candidate_rng = build("rmsnorm")
    assert torch.equal(baseline_rng, candidate_rng)
    old, new = baseline.student.state_dict(), candidate.student.state_dict()
    prefixes = {f"graph.layers.{i}.norm{j}" for i in range(4) for j in (1, 2)}
    assert set(old) - set(new) == {f"{path}.bias" for path in prefixes}
    assert not set(new) - set(old)
    for name in new:
        torch.testing.assert_close(old[name], new[name], atol=0, rtol=0)
    old_modules, new_modules = (
        dict(baseline.student.named_modules()),
        dict(candidate.student.named_modules()),
    )
    changed = {
        name for name in old_modules if type(old_modules[name]) is not type(new_modules[name])
    }
    assert changed == prefixes
    for name in prefixes:
        a, b = old_modules[name], new_modules[name]
        assert isinstance(a, nn.LayerNorm) and isinstance(b, nn.RMSNorm)
        assert a.eps == b.eps and a.normalized_shape == b.normalized_shape
    assert baseline.student.options.payload() != candidate.student.options.payload()
    assert "graph_input_norm" not in baseline.student.options.payload()
    assert candidate.student.options.payload()["graph_input_norm"] == "rmsnorm"
    for layer in candidate.teacher.graph.layers:
        assert isinstance(layer.norm1, nn.RMSNorm) and isinstance(layer.norm2, nn.RMSNorm)


@pytest.mark.parametrize("norm", ["historical", "rmsnorm"])
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
    identity = {"graph_norm": norm, "architecture": objective.student.options.payload()}
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
