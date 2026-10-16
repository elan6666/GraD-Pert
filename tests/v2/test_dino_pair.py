"""Scheduled targets, prototype freeze and full-state resume on the new anchor."""

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_restored_ffn import restored

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective, cross_entropy_terms
from gradpert.training.v2.optimizer import V2Optimizer, routes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from generate_dino_pair import ARMS, verify_manifest  # noqa: E402


def pair(name):
    parent, batch = restored("BR")
    architecture = replace(
        parent.student.options,
        ffn_variant="gelu",
        prediction_parameterization="softplus",
        prototype_weight_norm=True,
        prototype_freeze_epochs=1,
        prototype_fixed_scale=name == "DP2",
        prototype_freeze_gradients=name == "DP2",
    )
    torch.manual_seed(131)
    model = GraDPertV2(parent.student.graph.embedding.weight.detach().clone(), architecture)
    settings = (
        dict(
            teacher_temperature_start=0.04,
            teacher_temperature_end=0.03,
            teacher_temperature_warmup_steps=3,
        )
        if name == "DP2"
        else {}
    )
    return JointObjective(model, ssl1_weights=(1, 0, 0), ssl2_weights=(1, 1, 0), **settings), batch


def test_manifest_two_exact_arms_and_parent_protocol():
    m = verify_manifest(ROOT, ROOT / "configs/v2/dino_pair_20261016/manifest.json")
    assert m["priority"] == ["B0-DP", "DP2"]
    assert (m["epochs"], m["global_batch"], m["validation"]) == (4, 320, "disabled")
    for row in m["rows"]:
        c = load_experiment_config(ROOT / row["config"])
        a, o = V2Options.parse_parameters(c.model.parameters)
        assert a.prototype_fixed_scale == (row["name"] == "DP2")
        assert a.prototype_freeze_gradients == (row["name"] == "DP2")
        assert o.ssl1_node == o.ssl1_spread == o.ssl2_koleo == 0
        assert o.teacher_temperature_end == (0.03 if row["name"] == "DP2" else 0.04)
        assert row["changes"] == ARMS[row["name"]]


def test_temperature_logits_gradient_and_sharpening():
    s = torch.tensor([[0.4, -0.2, 0.1]], requires_grad=True)
    t = torch.tensor([[0.04, 0.0, -0.02]], requires_grad=True)
    c = torch.zeros(3)
    actual = cross_entropy_terms(s, t, c, teacher_temperature=0.03)
    p = (t.detach() / 0.03).softmax(-1)
    expected = -(p * (s / 0.1).log_softmax(-1)).sum(-1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    torch.testing.assert_close(s.grad, ((s.detach() / 0.1).softmax(-1) - p) / 0.1)
    assert t.grad is None
    warmer = (t.detach() / 0.04).softmax(-1)
    assert -(p * p.log()).sum() < -(warmer * warmer.log()).sum()


@pytest.mark.parametrize("name", ["B0-DP", "DP2"])
def test_updates_freeze_ema_centers_temperature_and_exact_resume(name, tmp_path):
    obj, batch = pair(name)
    opt = V2Optimizer(obj.student, 1e-3, 0.01, prototype_freeze_steps=2)
    before = {
        n: p.detach().clone() for n, p in obj.student.named_parameters() if "prototypes." in n
    }
    assert not any("original0" in r["name"] for r in routes(obj.student)) if name == "DP2" else True
    for i in range(2):
        old_teacher = {n: p.detach().clone() for n, p in obj.teacher.named_parameters()}
        old_center = obj.ssl2_cls_center.clone()
        optimizer_step(obj, opt, batch, microbatch=4, lr=1e-3, momentum=0.996, bf16=False)
        for n, p in obj.student.named_parameters():
            if n in before:
                torch.testing.assert_close(p, before[n], atol=0, rtol=0)
                if name == "DP2" and p.requires_grad:
                    assert p not in opt.adamw.state
        for n, p in obj.teacher.named_parameters():
            torch.testing.assert_close(
                p, old_teacher[n].lerp(dict(obj.student.named_parameters())[n], 0.004)
            )
            assert p.grad is None
        assert not torch.equal(old_center, obj.ssl2_cls_center)
        if name == "DP2":
            assert obj.teacher_temperature == pytest.approx([0.035, 0.03][i])
            assert int(obj.temperature_update_index) == i + 1
    assert obj.ssl1_node_center.count_nonzero() == 0
    p = tmp_path / "last.pt"
    rng = np.random.default_rng(51)
    save_checkpoint(p, obj, opt, identity={"arm": name}, progress={"step": 2}, generator=rng)
    expected = optimizer_step(obj, opt, batch, microbatch=4, lr=1e-3, momentum=0.996, bf16=False)
    state = copy.deepcopy(obj.state_dict())
    assert any(
        not torch.equal(v, dict(obj.student.named_parameters())[n])
        for n, v in before.items()
        if "original1" in n
    )
    new, _ = pair(name)
    newopt = V2Optimizer(new.student, 1e-3, 0.01, prototype_freeze_steps=2)
    load_checkpoint(p, new, newopt, identity={"arm": name}, generator=np.random.default_rng(0))
    actual = optimizer_step(new, newopt, batch, microbatch=4, lr=1e-3, momentum=0.996, bf16=False)
    assert expected == actual
    for n, v in state.items():
        torch.testing.assert_close(v, new.state_dict()[n], atol=0, rtol=0)
    if name == "DP2":
        for n, p in new.student.named_parameters():
            if "prototypes.parametrizations.weight.original0" in n:
                assert not p.requires_grad
                torch.testing.assert_close(p, torch.ones_like(p), atol=0, rtol=0)


def test_forward_diagnostics_do_not_advance_temperature():
    from gradpert.training.v2.process_diagnostics import evaluate_process_diagnostics

    obj, batch = pair("DP2")
    obj.commit_statistics(0.996)
    before = copy.deepcopy(obj.state_dict())
    r = evaluate_process_diagnostics(obj, batch, bf16=False, seed=31)
    assert r["schema"] == "training-process-diagnostics-1"
    assert obj.teacher_temperature == pytest.approx(0.035)
    for n, v in before.items():
        torch.testing.assert_close(v, obj.state_dict()[n], atol=0, rtol=0)
