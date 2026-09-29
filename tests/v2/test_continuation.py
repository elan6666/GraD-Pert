"""New-stage full-state and LoRA continuation without touching historical v1."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_components import fixture

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file, sha256_json
from gradpert.modeling.v2.lora import insert_lora, parent_to_lora_state
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.continuation import restore_parent
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.lifecycle import ConstantStageLR, fit
from gradpert.training.v2.lifecycle import test_selected as evaluate_selected
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer


def test_lora_starts_at_parent_prediction_and_updates_only_adapters(tmp_path):
    model, batch = fixture()
    parent = JointObjective(model, 0, 0).eval()
    with torch.no_grad():
        before = parent(batch)[0]
    child_model = copy.deepcopy(model)
    targets = insert_lora(child_model, rank=2, alpha=4)
    assert targets and all(
        name == "condition_fusion"
        or name.startswith(("graph.", "expression.", "cell.", "condition_fusion.", "response."))
        for name in targets
    )
    child = JointObjective(child_model, 0, 0).eval()
    child.load_state_dict(parent_to_lora_state(parent.state_dict(), child))
    with torch.no_grad():
        torch.testing.assert_close(child(batch)[0], before, rtol=1e-6, atol=1e-7)
    frozen = {
        name: value.clone()
        for name, value in child.student.state_dict().items()
        if ".base." in name
    }
    optimizer = V2Optimizer(child.student, 0.001, 0)
    assert optimizer.muon is None and optimizer.adamw is not None
    optimizer_step(child, optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False)
    assert optimizer.steps == 1
    for name, value in frozen.items():
        torch.testing.assert_close(child.student.state_dict()[name], value, rtol=0, atol=0)
    assert any(
        torch.count_nonzero(value)
        for name, value in child.student.state_dict().items()
        if name.endswith(".up")
    )
    path = tmp_path / "lora.pt"
    rng = np.random.default_rng(4)
    save_checkpoint(
        path, child, optimizer, identity={"stage": "lora"}, progress={"history": []}, generator=rng
    )
    restored = JointObjective(copy.deepcopy(child_model), 0, 0)
    restored_optimizer = V2Optimizer(restored.student, 0.001, 0)
    load_checkpoint(path, restored, restored_optimizer, identity={"stage": "lora"}, generator=rng)
    assert restored_optimizer.steps == 1
    for name, value in child.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_lora_two_nonzero_joint_updates_keep_teacher_and_centers(tmp_path):
    model, batch = fixture()
    parent = JointObjective(model, 1, 1)
    adapted = copy.deepcopy(model)
    insert_lora(adapted, rank=2, alpha=4)
    objective = JointObjective(adapted, 1, 1)
    objective.load_state_dict(parent_to_lora_state(parent.state_dict(), objective))
    optimizer = V2Optimizer(objective.student, 0.0002, 0)
    before_teacher = copy.deepcopy(objective.teacher.state_dict())
    for _ in range(2):
        terms = optimizer_step(
            objective, optimizer, batch, microbatch=2, lr=0.0002, momentum=0.99, bf16=False
        )
        assert np.isfinite(terms["joint_loss"])
    assert optimizer.steps == 2
    assert any(
        not torch.equal(value, before_teacher[name])
        for name, value in objective.teacher.state_dict().items()
        if name.endswith(".up")
    )
    assert any(
        torch.count_nonzero(getattr(objective, name))
        for name in ("ssl1_cls_center", "ssl1_node_center", "ssl2_cls_center", "ssl2_node_center")
    )
    path = tmp_path / "joint-lora.pt"
    save_checkpoint(
        path,
        objective,
        optimizer,
        identity={"stage": "joint-lora"},
        progress={"history": []},
        generator=np.random.default_rng(3),
    )
    restored = JointObjective(copy.deepcopy(adapted), 1, 1)
    restored_optimizer = V2Optimizer(restored.student, 0.0002, 0)
    load_checkpoint(
        path,
        restored,
        restored_optimizer,
        identity={"stage": "joint-lora"},
        generator=np.random.default_rng(3),
    )
    assert restored_optimizer.steps == 2
    for name, value in objective.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_full_state_bootstrap_keeps_parent_best_and_continues_global_epochs(tmp_path):
    model, batch = fixture()
    parent = JointObjective(model, 0, 0)
    optimizer = V2Optimizer(model, 0.001, 0)
    rng = np.random.default_rng(11)
    for _ in range(6):
        optimizer_step(parent, optimizer, batch, microbatch=2, lr=0.0002, momentum=0.99, bf16=False)
    history = [
        {
            "epoch": epoch,
            "optimizer_steps": epoch * 2,
            "training": {},
            "validation": {"split": "val", "joint_loss": 0.1 if epoch == 2 else 0.2},
            "seen_expression_gene_indices": [0, 1],
        }
        for epoch in (1, 2, 3)
    ]
    best_path = tmp_path / "parent-best.pt"
    save_checkpoint(
        best_path,
        parent,
        optimizer,
        identity={"run": "parent"},
        progress={"history": history[:2]},
        generator=rng,
    )
    best = {
        "file": "epoch-0002.pt",
        "sha256": sha256_file(best_path),
        "epoch": 2,
        "joint_loss": 0.1,
        "prediction_loss": 0.2,
        "training_identity": {"run": "parent"},
    }
    journal = fit(
        parent,
        optimizer,
        root=tmp_path / "child",
        identity={"run": "child"},
        generator=rng,
        epochs=5,
        steps_per_epoch=2,
        batches=lambda _: (batch for _ in range(2)),
        validate=lambda: {"split": "val", "prediction_loss": 0.4, "joint_loss": 0.3},
        schedule=ConstantStageLR(0.0002),
        teacher_start=1.0,
        teacher_end=1.0,
        microbatch=2,
        bf16=False,
        bootstrap_history=history,
        bootstrap_best=(best, best_path),
    )
    assert journal["epoch"] == 5 and optimizer.steps == 10
    assert journal["best"]["epoch"] == 2
    assert journal["last"]["epoch"] == 5
    receipts = evaluate_selected(
        parent,
        root=tmp_path / "child",
        evaluation_identity={"eval": "new"},
        test=lambda: {"split": "test"},
    )
    assert receipts["best"]["identity"]["checkpoint"]["training_identity"] == {"run": "parent"}
    assert receipts["best"]["identity"]["training"] == {"run": "parent"}
    assert receipts["last"]["identity"]["training"] == {"run": "child"}


@pytest.mark.parametrize("mode", ["full_state", "lora"])
def test_parent_restore_verifies_completed_run_and_mode(tmp_path, mode):
    model, batch = fixture()
    original = JointObjective(model, 0, 0)
    old_optimizer = V2Optimizer(model, 0.001, 0)
    for _ in range(6):
        optimizer_step(
            original, old_optimizer, batch, microbatch=2, lr=0.0002, momentum=0.99, bf16=False
        )
    history = [{"epoch": epoch, "seen_expression_gene_indices": [0, 1]} for epoch in (1, 2, 3)]
    parent = tmp_path / "parent"
    (parent / "fit").mkdir(parents=True)
    checkpoint = parent / "fit/epoch-0003.pt"
    prior_config = {
        "data": {"dataset": "synthetic"},
        "model": {"parameters": {"world_size": 1}},
        "training": {
            "formal_run_policy": "v2_fixed_3",
            "max_epochs": {"value": 3},
            "train_batch_size": 2,
        },
        "evaluation": {},
        "artifacts": {},
    }
    data_identity = {"dataset": "synthetic"}
    identity = {
        "run_id": "parent-synthetic",
        "source": {"commit": "old"},
        "config_sha256": "old-config",
        "resolved_config_sha256": sha256_json(prior_config),
        "data": data_identity,
    }
    save_checkpoint(
        checkpoint,
        original,
        old_optimizer,
        identity=identity,
        progress={"history": history},
        generator=np.random.default_rng(3),
    )
    selected = {"file": checkpoint.name, "epoch": 3, "sha256": sha256_file(checkpoint)}
    atomic_json(parent / "run_manifest.json", identity)
    atomic_json(
        parent / "COMPLETE.json",
        {"identity": identity, "epoch": 3, "test_roles": ["best", "last"]},
    )
    atomic_json(
        parent / "fit/epoch_state.json",
        {"identity": identity, "epoch": 3, "budget": [3, 2], "best": selected, "last": selected},
    )
    atomic_json(parent / "resolved_config.json", prior_config)
    current = copy.deepcopy(prior_config)
    current["training"]["formal_run_policy"] = "v2_fixed_5"
    current["training"]["max_epochs"]["value"] = 5
    stage = SimpleNamespace(
        mode=mode,
        parent_run_root=str(parent),
        parent_checkpoint=str(checkpoint),
        parent_checkpoint_sha256=sha256_file(checkpoint),
        parent_epoch=3,
        additional_epochs=2,
        learning_rate=0.0002,
    )
    config = SimpleNamespace(continuation=stage, model_dump=lambda **_: current)
    child_model = copy.deepcopy(model)
    if mode == "lora":
        insert_lora(child_model, rank=2, alpha=4)
    child = JointObjective(child_model, 0, 0)
    optimizer = V2Optimizer(child.student, 0.001, 0)
    restored_history, best, provenance = restore_parent(
        config,
        data_identity=data_identity,
        objective=child,
        optimizer=optimizer,
        generator=np.random.default_rng(4),
        steps_per_epoch=2,
        server_root=tmp_path,
    )
    assert restored_history == history and optimizer.steps == 6
    assert (best is None) == (mode == "lora")
    assert provenance["optimizer_restored"] == (mode == "full_state")
    for name, value in original.state_dict().items():
        restored = child.state_dict()
        path, _, field = name.rpartition(".")
        destination = name if name in restored else f"{path}.base.{field}"
        torch.testing.assert_close(restored[destination], value, rtol=0, atol=0)
    stage.parent_checkpoint_sha256 = "0" * 64
    with pytest.raises(ValueError, match="incomplete or last checkpoint"):
        restore_parent(
            config,
            data_identity=data_identity,
            objective=child,
            optimizer=optimizer,
            generator=np.random.default_rng(4),
            steps_per_epoch=2,
            server_root=tmp_path,
        )


def test_continuation_configs_are_self_contained_and_old_payload_is_unchanged():
    from pathlib import Path

    from gradpert.config import load_experiment_config

    base = Path("configs/v2/no_mhc_joint_eval_jurkat")
    old = load_experiment_config(base / "three_epoch_m74_a2/gradpert_v2/nadig_jurkat.yaml")
    assert "continuation" not in old.model_dump(mode="json")
    for directory, mode in (
        ("continue_full_m74_a2", "full_state"),
        ("finetune_lora_r8_m74_a2", "lora"),
    ):
        config = load_experiment_config(base / directory / "gradpert_v2/nadig_jurkat.yaml")
        assert config.continuation is not None
        assert config.continuation.mode == mode
        assert config.training.max_epochs.value == 5
        assert config.continuation.learning_rate == 0.0002
