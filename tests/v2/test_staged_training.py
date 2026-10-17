"""No-label pretraining, exact stage transition and phase-aware full-state resume."""

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_fgpo_family import new_fixture

from gradpert.config.step_schedule import EndpointLRWarmupCosine
from gradpert.data._io import read_json
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.lifecycle import PretrainThenMSELR, fit
from gradpert.training.v2.loss_diagnostics import evaluate_loss_diagnostics
from gradpert.training.v2.optimizer import V2Optimizer
from gradpert.training.v2.staged import StagedObjective

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))


def fixture(pretrain_steps=2):
    parent, batch = new_fixture("B0")
    objective = StagedObjective(
        copy.deepcopy(parent.student),
        pretrain_steps=pretrain_steps,
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
        loss_reduction="row_mean",
    )
    return objective, batch, V2Optimizer(objective.student, 1e-3, 0)


def update(obj, batch, opt):
    if opt.steps < obj.pretrain_steps:
        batch = replace(batch, truth=torch.full_like(batch.truth, float("nan")))
    else:
        batch = replace(batch, graph_views=(), cell_views=())
    return optimizer_step(obj, opt, batch, microbatch=4, lr=2e-4, momentum=0.9, bf16=False)


def test_pretraining_never_decodes_or_reads_truth_and_has_identical_loss_gradients():
    obj, batch, _ = fixture()
    obj.eval()

    def forbidden(*args):
        raise AssertionError("pretraining decoder was called")

    handles = [
        model.prediction.register_forward_pre_hook(forbidden)
        for model in (obj.student, obj.teacher)
    ]
    states = []
    for truth in (torch.full_like(batch.truth, float("nan")), batch.truth * 999):
        obj.zero_grad(set_to_none=True)
        with torch.random.fork_rng():
            torch.manual_seed(73)
            loss, terms = obj(replace(batch, truth=truth))
            loss.backward()
        assert "prediction" not in terms and torch.isfinite(loss)
        states.append(
            (
                loss.detach(),
                {
                    n: p.grad.clone()
                    for n, p in obj.student.named_parameters()
                    if p.grad is not None
                },
            )
        )
        obj.pending.clear()
    torch.testing.assert_close(states[0][0], states[1][0], atol=0, rtol=0)
    assert states[0][1].keys() == states[1][1].keys()
    for name in states[0][1]:
        torch.testing.assert_close(states[0][1][name], states[1][1][name], atol=0, rtol=0)
    assert not any(name.startswith("prediction.") for name in states[0][1])
    for h in handles:
        h.remove()


def test_boundary_resets_moments_once_and_freezes_teacher_centers_and_ssl_heads():
    obj, batch, opt = fixture()
    head = copy.deepcopy(obj.student.prediction.state_dict())
    for _ in range(2):
        terms = update(obj, batch, opt)
        assert "prediction" not in terms
    for n, value in head.items():
        torch.testing.assert_close(value, obj.student.prediction.state_dict()[n], atol=0, rtol=0)
    teacher = copy.deepcopy(obj.teacher.state_dict())
    centers = {n: b.clone() for n, b in obj.named_buffers() if n.endswith("_center")}
    heads = {n: p.clone() for n, p in obj.student.named_parameters() if n.startswith("ssl")}
    assert any(opt.muon.state.values())
    obj.prepare_update(opt)
    assert opt.stage_reset_count == 1 and opt.steps == 2
    assert len(opt.muon.state) == len(opt.adamw.state) == 0
    for _ in range(3):
        terms = update(obj, batch, opt)
        assert terms["joint_loss"] == terms["prediction"]
        assert not any(n.startswith("ssl") for n in terms)
    assert opt.stage_reset_count == 1
    assert any(not torch.equal(v, obj.student.prediction.state_dict()[n]) for n, v in head.items())
    for n, value in teacher.items():
        torch.testing.assert_close(value, obj.teacher.state_dict()[n], atol=0, rtol=0)
    for n, value in centers.items():
        torch.testing.assert_close(value, dict(obj.named_buffers())[n], atol=0, rtol=0)
    for n, value in heads.items():
        torch.testing.assert_close(value, dict(obj.student.named_parameters())[n], atol=0, rtol=0)


@pytest.mark.parametrize("saved_step", [1, 2, 3])
def test_exact_full_state_resume_before_boundary_at_boundary_and_in_finetune(tmp_path, saved_step):
    obj, batch, opt = fixture()
    rng = np.random.default_rng(31)
    for _ in range(saved_step):
        update(obj, batch, opt)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, obj, opt, identity={"id": "PT1"}, progress={"step": saved_step}, generator=rng
    )
    expected = [update(obj, batch, opt) for _ in range(5 - saved_step)]
    state, optimizer = copy.deepcopy(obj.state_dict()), copy.deepcopy(opt.state_dict())
    new, _, newopt = fixture()
    load_checkpoint(path, new, newopt, identity={"id": "PT1"}, generator=np.random.default_rng(0))
    actual = [update(new, batch, newopt) for _ in range(5 - saved_step)]
    assert expected == actual
    assert optimizer["steps"] == newopt.steps == 5 and newopt.stage_reset_count == 1
    for name, value in state.items():
        torch.testing.assert_close(value, new.state_dict()[name], atol=0, rtol=0)


def test_ssl_only_diagnostics_no_prediction_and_unchanged_state():
    obj, batch, _ = fixture()
    state = copy.deepcopy(obj.state_dict())
    result = evaluate_loss_diagnostics(
        obj, replace(batch, truth=torch.full_like(batch.truth, float("nan"))), bf16=False
    )
    assert result["prediction_status"] == "not_applicable:no_labels"
    for n, value in state.items():
        torch.testing.assert_close(value, obj.state_dict()[n], atol=0, rtol=0)


def test_four_epoch_lifecycle_retains_pretrain_and_resume_boundary(tmp_path):
    obj, batch, _ = fixture(pretrain_steps=8)
    initial = copy.deepcopy(obj.student.state_dict())

    def run(root, resume=False, interrupt=False):
        new, _, newopt = fixture(pretrain_steps=8)
        new.student.load_state_dict(initial)
        new.teacher.load_state_dict(initial)

        def batches(epoch):
            if interrupt and epoch == 2:
                raise RuntimeError("boundary interruption")
            for _ in range(4):
                yield (
                    replace(batch, truth=torch.full_like(batch.truth, float("nan")))
                    if epoch < 2
                    else batch
                )

        fit(
            new,
            newopt,
            root=root,
            identity={"id": "PT1", "run_id": "synthetic-pt1", "source": {"commit": "c" * 40}},
            generator=np.random.default_rng(3),
            epochs=4,
            steps_per_epoch=4,
            batches=batches,
            validate=None,
            schedule=PretrainThenMSELR(EndpointLRWarmupCosine(1e-3, 2e-4, 0.16), 8, 2e-4),
            teacher_start=0.99,
            teacher_end=1,
            microbatch=4,
            bf16=False,
            resume=resume,
        )
        return copy.deepcopy(new.state_dict())

    expected = run(tmp_path / "full")
    with pytest.raises(RuntimeError, match="boundary"):
        run(tmp_path / "resumed", interrupt=True)
    actual = run(tmp_path / "resumed", resume=True)
    for n in expected:
        torch.testing.assert_close(expected[n], actual[n], atol=0, rtol=0)
    root = tmp_path / "resumed"
    from gradpert.training.v2.reporting import export_curves

    export_curves(root)
    assert "training_stage" in (root / "epoch_curves.csv").read_text().splitlines()[0]
    assert (root / "pretrain.pt").read_bytes() == (root / "epoch-0002.pt").read_bytes()
    assert [r["training_stage"] for r in read_json(root / "history.json")] == [
        "ssl_pretrain"
    ] * 2 + ["mse_finetune"] * 2


def test_pt1_config_changes_only_explicit_training_mode():
    from generate_fgpo_family import configs
    from generate_pt1 import configuration, verify_manifest

    value = verify_manifest(ROOT, ROOT / "configs/v2/pt1_staged_20261017/manifest.json")
    config = configuration(ROOT)
    for key in ("training_mode", "ssl_pretrain_epochs", "finetune_learning_rate"):
        config["model"]["parameters"].pop(key)
    assert config == configs(ROOT, microbatch=80)["B0"] and value["priority"] == ["PT1"]


def test_skipping_unused_ssl_decoder_preserves_complete_nonzero_lr_updates():
    fast, batch, fastopt = fixture(pretrain_steps=3)
    reference, _, refopt = fixture(pretrain_steps=3)
    reference.load_state_dict(fast.state_dict())
    reference.decode_ssl_views = True
    for _ in range(5):
        rng = torch.get_rng_state()
        expected = update(reference, batch, refopt)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        actual = update(fast, batch, fastopt)
        assert expected == actual and torch.equal(after, torch.get_rng_state())
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, fast.state_dict()[name], atol=0, rtol=0)
        for (name, a), (_, b) in zip(
            reference.student.named_parameters(), fast.student.named_parameters(), strict=True
        ):
            assert (a.grad is None) == (b.grad is None), name
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    from capacity_probe import state_fingerprint

    assert state_fingerprint(refopt.state_dict()) == state_fingerprint(fastopt.state_dict())


def test_stage_a_process_and_graph_gradient_diagnostics_never_touch_labels():
    from gradpert.training.v2.process_diagnostics import (
        evaluate_graph_gradient_diagnostics,
        evaluate_process_diagnostics,
    )

    obj, batch, _ = fixture()
    batch = replace(batch, truth=torch.full_like(batch.truth, float("nan")))
    initial = copy.deepcopy(obj.state_dict())
    result = evaluate_process_diagnostics(obj, batch, bf16=False, seed=33)
    assert result["training_stage"] == "ssl_pretrain" and "prediction" not in result["components"]
    gradients = evaluate_graph_gradient_diagnostics(obj, batch, bf16=False)
    assert gradients["performed"]
    for name, value in initial.items():
        torch.testing.assert_close(value, obj.state_dict()[name], atol=0, rtol=0)
