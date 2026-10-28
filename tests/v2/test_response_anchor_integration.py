"""Native synthetic update integration, with no real expression or checkpoint inputs."""

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_components import fixture

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer
from gradpert.training.v2.response_anchor import attach_anchor_readout
from gradpert.training.v2.response_anchor_runtime import ResponseAnchorController


def integrated():
    model, batch = fixture()
    attach_anchor_readout(model, 8, 20261028)
    objective = JointObjective(model, 0, 0, loss_reduction="row_mean")
    batch = replace(batch, graph_views=(), cell_views=())
    genes = tuple(f"g{i}" for i in range(1322))
    conditions = ("g5+ctrl", *(f"g{i}+ctrl" for i in range(1321) if i != 5), "g1321+ctrl")
    anchors = np.array([5] + [i for i in range(1321) if i != 5] + [1321])
    controls = np.arange(16, dtype=np.float32).reshape(4, 4) / 100
    bank = SimpleNamespace(
        identity={"split_content_sha256": "split", "selection": {}, "allowed_policy": {}},
        condition_ids=conditions,
        condition_anchor_gene_indices=anchors,
        allowed_gene_indices=np.arange(4),
        condition_targets=np.ones((1322, 8), np.float32),
        control_slot_offsets=np.arange(1323) * 4,
        control_for_condition=lambda p: controls.copy(),
    )

    def view(ids, targets, **kwargs):
        # Preserve the native graph tensors, while selecting an out-of-query target.
        return replace(
            batch.graph,
            target_positions=torch.tensor([[targets[0][0]]]),
            target_valid=torch.ones(1, 1, dtype=torch.bool),
        )

    optimizer = V2Optimizer(model, 1e-3, 0.01)
    runtime = SimpleNamespace(
        data=SimpleNamespace(
            split=SimpleNamespace(
                split_content_sha256="split",
                train_conditions=conditions,
                control_condition_id="ctrl",
            ),
            graph_gene_ids=genes,
        ),
        identity={"training_row_selection": {}, "training_expression_policy": {}},
        steps_per_epoch=150,
        objective=objective,
        optimizer=optimizer,
        options=SimpleNamespace(response_anchor_seed=20261028, response_anchor_weight=0.2),
        device=torch.device("cpu"),
        index=SimpleNamespace(view=view),
    )
    controller = ResponseAnchorController(runtime, bank)
    controller.groups = lambda step: (0,)
    objective.response_anchor_backward = controller.backward
    return objective, batch, optimizer, controller


def test_group_path_excludes_decoder_truth_teacher_and_preserves_rng():
    objective, batch, _, controller = integrated()
    before = copy.deepcopy(objective.teacher.state_dict())
    rng = torch.get_rng_state().clone()
    loss = controller.group_loss(batch, 0, 0)
    changed = replace(batch, truth=torch.full_like(batch.truth, float("nan")))
    torch.testing.assert_close(loss, controller.group_loss(changed, 0, 0), rtol=0, atol=0)
    loss.backward()
    assert torch.equal(rng, torch.get_rng_state())
    assert any(
        p.grad is not None and p.grad.abs().sum() for p in objective.student.response.parameters()
    )
    assert all(p.grad is None for p in objective.student.prediction.parameters())
    assert objective.student.response_anchor_readout.weight.grad.abs().sum() > 0
    assert not objective.pending
    assert all(p.grad is None for p in objective.teacher.parameters())
    for key, value in before.items():
        torch.testing.assert_close(value, objective.teacher.state_dict()[key], rtol=0, atol=0)


def test_native_step_once_and_checkpoint_resume_keeps_anchor_clock(tmp_path):
    objective, batch, optimizer, _ = integrated()
    rng = np.random.default_rng(41)

    def step(obj, opt):
        return optimizer_step(obj, opt, batch, microbatch=1, lr=1e-3, momentum=0.99, bf16=False)

    result = step(objective, optimizer)
    assert optimizer.steps == 1
    assert result["response_anchor_rows"] == 4
    assert result["joint_loss"] == pytest.approx(
        result["prediction"] + result["response_anchor_weighted"]
    )
    assert not objective.pending
    assert not any("condition_targets" in k for k in objective.state_dict())
    path = tmp_path / "synthetic.pt"
    save_checkpoint(
        path,
        objective,
        optimizer,
        identity={"scope": "anchor"},
        progress={"step": 1},
        generator=rng,
    )
    expected = step(objective, optimizer)
    expected_state = copy.deepcopy(objective.state_dict())
    other, _, other_optimizer, _ = integrated()
    load_checkpoint(path, other, other_optimizer, identity={"scope": "anchor"}, generator=rng)
    actual = step(other, other_optimizer)
    assert actual == expected and other_optimizer.steps == 2
    for key, value in expected_state.items():
        torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)


def test_anchor_capacity_restore_fingerprint_includes_main_rng():
    from scripts.v2.capacity_probe import staged_restore_fingerprint

    objective, _, _, controller = integrated()
    objective.response_anchor_enabled = True
    runtime = controller.runtime
    runtime.generator = np.random.default_rng(44)
    saved = torch.get_rng_state()
    before = staged_restore_fingerprint(runtime, continuing_joint=False)
    torch.rand(1)
    assert staged_restore_fingerprint(runtime, continuing_joint=False) != before
    torch.set_rng_state(saved)
    assert staged_restore_fingerprint(runtime, continuing_joint=False) == before


def _distributed_anchor_gradient(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    from gradpert.training.v2.distributed import average_gradients

    objective, batch, _, controller = integrated()
    params = tuple(objective.student.parameters())
    expected = torch.autograd.grad(
        0.2 * 150 / 1322 * controller.group_loss(batch, 0, 0), params, allow_unused=True
    )
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        controller.world, controller.rank = 2, rank
        controller.groups = lambda step: (0,) if rank == 0 else ()
        for p in params:
            p.grad = torch.ones_like(p)  # Existing native main gradients must survive.
        result = controller.backward(batch, 0)
        assert result["response_anchor_groups"] == (1 if rank == 0 else 0)
        average_gradients(objective.student)
        for p, gradient in zip(params, expected, strict=True):
            torch.testing.assert_close(
                p.grad,
                torch.ones_like(p) + (gradient if gradient is not None else 0),
                rtol=2e-5,
                atol=2e-6,
            )
    finally:
        dist.destroy_process_group()


def test_native_rank_mean_with_empty_rank_and_existing_main_gradients(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_distributed_anchor_gradient, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def test_small_diagnostic_marks_anchor_missing_without_loading_bank():
    from gradpert.training.v2.diagnostic_objective import diagnostic_objective

    objective, batch, _, _ = integrated()
    objective.response_anchor_enabled = True
    _, terms, scope = diagnostic_objective(objective, batch)
    assert "response_anchor" not in terms
    assert scope["diagnostic_objective_scope"]["unavailable_terms"]["response_anchor"] == (
        "requires_complete_train_condition_bank"
    )


def test_native_bank_guard_rejects_escape_before_reading_array(tmp_path, monkeypatch):
    import hashlib
    import json

    from gradpert.training.v2 import response_anchor_runtime as module

    monkeypatch.setattr(module, "SERVER_ROOT", tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"arrays": {"path": "../outside.npz"}}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="adjacent"):
        module.load_response_bank(str(path), digest, np.arange(4))


@pytest.mark.parametrize(
    "change",
    [
        {"response_anchor_weight": 0.1},
        {"response_anchor_weight": float("nan")},
        {"response_anchor_seed": True},
        {
            "response_anchor_path": "/tmp/bank",
            "response_anchor_sha256": "0" * 64,
            "response_anchor_weight": 0.1,
        },
    ],
)
def test_config_rejects_unsealed_or_local_bank(change):
    config = load_experiment_config(
        Path(__file__).parents[2]
        / "configs/v2/representation_round3_20261023/R3-RI/gradpert_v2/nadig_jurkat.yaml"
    )
    _, options = V2Options.parse_parameters(config.model.parameters)
    assert options.response_anchor_weight == 0 and not options.response_anchor_path
    with pytest.raises(ValueError):
        replace(options, **change)


def test_enabled_config_binds_manifest_and_arrays_for_value_free_evaluation():
    config = load_experiment_config(
        Path(__file__).parents[2]
        / "configs/v2/representation_round3_20261023/R3-RI/gradpert_v2/nadig_jurkat.yaml"
    )
    _, options = V2Options.parse_parameters(config.model.parameters)
    enabled = replace(
        options,
        response_anchor_weight=0.1,
        response_anchor_path="/data/yilangliu/bank/manifest.json",
        response_anchor_sha256="a" * 64,
        response_anchor_arrays_sha256="b" * 64,
    )
    assert enabled.response_anchor_arrays_sha256 == "b" * 64
    with pytest.raises(ValueError):
        replace(enabled, response_anchor_arrays_sha256="")


def test_generated_recipe_changes_only_anchor_and_retains_original_two_epoch_ri():
    import yaml

    from scripts.v2.generate_response_anchor import PARENT, configuration

    root = Path(__file__).parents[2]
    actual = configuration(
        root,
        bank_path="/data/yilangliu/bank/manifest.json",
        manifest_sha256="a" * 64,
        arrays_sha256="b" * 64,
        weight=0.12,
    )
    parameters = actual["model"]["parameters"]
    assert parameters["response_injection_residual"]["value"] is True
    assert actual["training"]["max_epochs"]["value"] == 2
    assert actual["training"]["train_batch_size"]["value"] == 320
    for key in list(parameters):
        if key.startswith("response_anchor_"):
            del parameters[key]
    assert actual == yaml.safe_load((root / PARENT).read_text())
