"""Continuous SSL prefix/joint objectives, schedules, sampling and exact resume."""

import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_fgpo_family import new_fixture

from gradpert.config.step_schedule import EndpointLRWarmupCosine
from gradpert.config.v2 import V2Options
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.lifecycle import PretrainThenJointLR, fit
from gradpert.training.v2.optimizer import V2Optimizer
from gradpert.training.v2.runtime import Runtime
from gradpert.training.v2.staged import SSLThenJointObjective


def fixture(pretrain_steps=2):
    parent, batch = new_fixture("B0")
    obj = SSLThenJointObjective(
        copy.deepcopy(parent.student),
        pretrain_steps=pretrain_steps,
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
        loss_reduction="row_mean",
        au_alignment_weight=0.05,
        cr_weight=4e-5,
        population_aux_space="ssl2_cls_bottleneck",
        au_warmup_steps=1,
        cr_warmup_steps=1,
    )
    return (
        obj,
        replace(batch, context_index=torch.zeros_like(batch.condition_index)),
        V2Optimizer(obj.student, 1e-3, 0.01),
    )


def update(obj, batch, opt):
    if opt.steps < obj.pretrain_steps:
        batch = replace(batch, truth=torch.full_like(batch.truth, float("nan")))
    return optimizer_step(obj, opt, batch, microbatch=4, lr=2e-4, momentum=0.99, bf16=False)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_prefix_never_decodes_or_uses_truth_and_prediction_parameters_do_not_update():
    obj, batch, opt = fixture()
    assert obj.decode_ssl_views is False
    prediction = copy.deepcopy(obj.student.prediction.state_dict())

    def forbidden(*args):
        raise AssertionError("prefix prediction decoder called")

    handles = [
        m.prediction.register_forward_pre_hook(forbidden) for m in (obj.student, obj.teacher)
    ]
    obj.eval()
    gradients = []
    for truth in (torch.full_like(batch.truth, float("nan")), batch.truth * 1000):
        obj.zero_grad(set_to_none=True)
        with torch.random.fork_rng():
            torch.manual_seed(37)
            loss, terms = obj(replace(batch, truth=truth), omit_au_for_diagnostic=True)
            loss.backward()
        assert "prediction" not in terms
        gradients.append(
            {n: p.grad.clone() for n, p in obj.student.named_parameters() if p.grad is not None}
        )
        obj.pending.clear()
    assert_nested_equal(*gradients)
    assert not any(n.startswith("prediction.") for n in gradients[0])
    for _ in range(2):
        terms = update(obj, batch, opt)
        assert "prediction" not in terms and "ssl1_condition" in terms and "ssl2_dino" in terms
        assert "cr_weighted" in terms and "au_weighted" in terms
    assert_nested_equal(prediction, obj.student.prediction.state_dict())
    assert_nested_equal(prediction, obj.teacher.prediction.state_dict())
    for h in handles:
        h.remove()
    # All readout routes, including Muon views, remain uninitialized in the prefix.
    named = dict(obj.student.named_parameters())
    for name, parameter in named.items():
        if name.startswith("prediction."):
            assert parameter.grad is None
            if opt.adamw is not None:
                assert parameter not in opt.adamw.state
    for parameter, parts in opt.views:
        if any(parameter is p for n, p in named.items() if n.startswith("prediction.")):
            assert opt.muon is not None and all(p not in opt.muon.state for p in parts)


def test_boundary_retains_moments_and_keeps_ssl_teacher_centers_active():
    obj, batch, opt = fixture()
    for _ in range(2):
        update(obj, batch, opt)
    moments = copy.deepcopy(opt.state_dict())
    teacher = copy.deepcopy(obj.teacher.state_dict())
    centers = {n: b.clone() for n, b in obj.named_buffers() if n.endswith("_center")}
    obj.prepare_update(opt)
    obj.prepare_update(opt)
    now = opt.state_dict()
    assert_nested_equal(moments["muon"], now["muon"])
    assert_nested_equal(moments["adamw"], now["adamw"])
    assert opt.steps == 2 and opt.stage_reset_count == 0
    assert obj.training_stage == opt.training_stage == "joint"
    assert obj.decode_ssl_views is True
    terms = update(obj, batch, opt)
    assert all(
        k in terms
        for k in ("prediction", "ssl1_condition", "ssl2_dino", "cr_weighted", "au_weighted")
    )
    assert obj.lambda1 == obj.lambda2 == 1 and obj.teacher_updates_enabled
    assert any(not torch.equal(v, obj.teacher.state_dict()[n]) for n, v in teacher.items())
    assert any(not torch.equal(v, dict(obj.named_buffers())[n]) for n, v in centers.items())


@pytest.mark.parametrize("saved_step", [1, 2, 3])
def test_exact_checkpoint_resume_including_boundary(tmp_path, saved_step):
    obj, batch, opt = fixture()
    rng = np.random.default_rng(19)
    for _ in range(saved_step):
        update(obj, batch, opt)
    path = tmp_path / "state.pt"
    save_checkpoint(
        path, obj, opt, identity={"run": "joint"}, progress={"steps": saved_step}, generator=rng
    )
    saved_decode = obj.decode_ssl_views
    assert saved_decode is (saved_step > obj.pretrain_steps)
    expected_rng = rng.random(4)
    expected = [update(obj, batch, opt) for _ in range(5 - saved_step)]
    expected_state = copy.deepcopy(obj.state_dict())
    expected_opt = copy.deepcopy(opt.state_dict())
    other, _, other_opt = fixture()
    other_rng = np.random.default_rng(0)
    load_checkpoint(path, other, other_opt, identity={"run": "joint"}, generator=other_rng)
    assert other.decode_ssl_views is saved_decode
    np.testing.assert_array_equal(expected_rng, other_rng.random(4))
    actual = [update(other, batch, other_opt) for _ in range(5 - saved_step)]
    assert expected == actual
    assert_nested_equal(expected_state, other.state_dict())
    assert_nested_equal(expected_opt, other_opt.state_dict())
    assert other_opt.steps == 5 and other_opt.stage_reset_count == 0
    assert other.decode_ssl_views is True


def test_inconsistent_optimizer_phase_fails_closed():
    obj, _, opt = fixture()
    opt.steps = 3
    with pytest.raises(ValueError, match="phase"):
        obj.prepare_update(opt)
    opt.training_stage = "ssl_pretrain"
    with pytest.raises(ValueError, match="exact prefix"):
        obj.prepare_update(opt)
    opt.steps = 0
    opt.stage_reset_count = 1
    with pytest.raises(ValueError, match="retain its moments"):
        obj.prepare_update(opt)


def test_separate_lr_clocks_match_original_joint_300_steps():
    cosine = EndpointLRWarmupCosine(1e-3, 2e-4, 0.16)
    schedule = PretrainThenJointLR(cosine, 150)
    assert [schedule.at_step(i, 450) for i in range(150)] == [
        cosine.at_step(i, 150) for i in range(150)
    ]
    assert [schedule.at_step(i + 150, 450) for i in range(300)] == [
        cosine.at_step(i, 300) for i in range(300)
    ]
    assert schedule.at_step(149, 450)["learning_rate"] == 2e-4
    assert schedule.at_step(150, 450)["learning_rate"] == 0
    assert schedule.at_step(449, 450)["learning_rate"] == 2e-4
    for prefix, total in ((0, 450), (450, 450), (150, 150), (1, 450), (150, 151)):
        with pytest.raises(ValueError):
            PretrainThenJointLR(cosine, prefix).at_step(0, total)


def test_runtime_sampler_mapping_control_only_and_ssl_views(monkeypatch):
    from gradpert.training.v2 import runtime as module

    @dataclass
    class Options:
        training_mode: str = "ssl_then_joint"
        ssl_pretrain_epochs: int = 1
        lambda1: float = 1
        lambda2: float = 1
        max_conditions: int = 0

    observed = []
    data = SimpleNamespace()
    data.iter_train_epoch = lambda **kw: iter([kw["epoch"]])

    def assemble(raw, index, options, generator, **kw):
        observed.append((raw, data.control_only_training, options.lambda1, options.lambda2))
        return "batch"

    monkeypatch.setattr(module, "assemble_batch", assemble)
    obj = SimpleNamespace(pretrain_steps=150, prediction_enabled=False)
    opt = SimpleNamespace(steps=0)
    rt = Runtime(
        data, Options(), None, obj, opt, np.random.default_rng(1), torch.device("cpu"), 2, {}
    )
    for physical, steps in ((0, 0), (1, 150), (2, 300)):
        opt.steps = steps
        list(rt.batches(physical))
    opt.steps = 150
    list(rt._batches(0, rt.device, rt.generator, diagnostic=True))
    assert observed == [(0, True, 1, 1), (0, False, 1, 1), (1, False, 1, 1), (0, True, 1, 1)]


def test_new_mode_allows_r6_auxiliaries_and_old_mode_rejects_them():
    from pathlib import Path

    from gradpert.config import load_experiment_config

    root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        root / "configs/v2/representation_round6_20261024/R6-EMA99/gradpert_v2/nadig_jurkat.yaml"
    )
    _, options = V2Options.parse_parameters(config.model.parameters)
    staged = replace(options, training_mode="ssl_then_joint", ssl_pretrain_epochs=1)
    assert staged.cr_weight == 4e-5 and staged.au_alignment_weight == 0.05
    with pytest.raises(ValueError):
        replace(staged, training_mode="ssl_then_mse")
    with pytest.raises(ValueError):
        replace(staged, ssl_pretrain_epochs=0)


def test_lifecycle_prefix_retention_counts_and_boundary_resume(tmp_path):
    obj, _, _ = fixture(pretrain_steps=4)
    initial = copy.deepcopy(obj.state_dict())

    def run(root, *, resume=False, interrupt=False):
        new, batch, opt = fixture(pretrain_steps=4)
        new.load_state_dict(initial)

        def batches(epoch):
            if interrupt and epoch == 1:
                raise RuntimeError("boundary interruption")
            yield from [
                replace(batch, truth=torch.full_like(batch.truth, float("nan")))
                if epoch == 0
                else batch
            ] * 4

        journal = fit(
            new,
            opt,
            root=root,
            identity={"run_id": "synthetic-r9"},
            generator=np.random.default_rng(5),
            epochs=3,
            steps_per_epoch=4,
            batches=batches,
            validate=None,
            schedule=PretrainThenJointLR(EndpointLRWarmupCosine(1e-3, 2e-4, 0.25), 4),
            teacher_start=0.99,
            teacher_end=0.99,
            microbatch=4,
            bf16=False,
            resume=resume,
        )
        return new.state_dict(), opt.state_dict(), journal

    expected, expected_opt, _ = run(tmp_path / "full")
    with pytest.raises(RuntimeError, match="boundary interruption"):
        run(tmp_path / "resumed", interrupt=True)
    actual, actual_opt, journal = run(tmp_path / "resumed", resume=True)
    assert_nested_equal(expected, actual)
    assert_nested_equal(expected_opt, actual_opt)
    assert journal["last"]["epoch"] == 3 and journal["best"] is None
    prefix = torch.load(tmp_path / "resumed/pretrain.pt", weights_only=False)
    final = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    assert prefix["progress"]["staged_training"]["joint_successful_updates"] == 0
    assert final["progress"]["staged_training"] == {
        "global_successful_updates": 12,
        "pretrain_successful_updates": 4,
        "joint_successful_updates": 8,
        "phase_successful_updates": 8,
    }
    assert [h["sampler_epoch"] for h in final["progress"]["history"]] == [0, 0, 1]
    assert [h["training_stage"] for h in final["progress"]["history"]] == [
        "ssl_pretrain",
        "joint",
        "joint",
    ]


def test_fixed_three_final_only_schema_is_supported_without_loosening_legacy(tmp_path):
    from pathlib import Path

    import yaml

    from gradpert.config import load_experiment_config

    root = Path(__file__).resolve().parents[2]
    source = (
        root / "configs/v2/representation_round6_20261024/R6-EMA99/gradpert_v2/nadig_jurkat.yaml"
    )
    raw = yaml.safe_load(source.read_text())
    raw["training"]["max_epochs"]["value"] = 3
    raw["training"]["formal_run_policy"] = "v2_fixed_3"
    params = raw["model"]["parameters"]
    for key, value in (("training_mode", "ssl_then_joint"), ("ssl_pretrain_epochs", 1)):
        params[key] = {"value": value, "source": "project_preregistered", "reference": "R9"}
    path = tmp_path / "gradpert_v2/nadig_jurkat.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(raw))
    config = load_experiment_config(path)
    assert config.training.max_epochs.value == 3 and config.training.monitor == "none"
    raw["training"]["formal_run_policy"] = "v2_fixed_5"
    raw["training"]["max_epochs"]["value"] = 5
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_experiment_config(path)
