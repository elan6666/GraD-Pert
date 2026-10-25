"""Frozen feature invariants and exact child checkpoint continuation."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_fgpo_family import new_fixture
from test_staged_joint_training import assert_nested_equal

from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.optimizer import V2Optimizer
from gradpert.training.v2.readout import (
    FrozenReadoutObjective,
    frozen_digest,
    frozen_state_proof,
    prediction_head_digest,
)


def fixture():
    parent, batch = new_fixture("B0")
    arch = replace(
        parent.student.options,
        width=256,
        heads=4,
        projector_hidden=32,
        projector_bottleneck=16,
        latent_rank=4,
        streams=1,
    )
    model = GraDPertV2(parent.student.graph.embedding.weight.detach().repeat(1, 32), arch)
    objective = FrozenReadoutObjective(model, lambda1=0, lambda2=0, loss_reduction="row_mean")
    objective.parent_frozen_sha256 = frozen_digest(objective)
    objective.parent_prediction_head_sha256 = prediction_head_digest(objective)
    return objective, replace(batch, graph_views=(), cell_views=()), V2Optimizer(model, 1e-3, 0)


def update(obj, batch, opt):
    return optimizer_step(obj, opt, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False)


def test_only_existing_mlp_changes_and_frozen_model_remains_eval():
    obj, batch, opt = fixture()
    initial = copy.deepcopy(obj.state_dict())
    obj.train()
    assert obj.training and not obj.student.training and not obj.teacher.training
    assert obj.student.prediction.training
    initial_proof = frozen_state_proof(obj, opt)
    assert initial_proof["prediction_head_changed"] is False
    before = obj.student.prediction[0].weight.detach().clone()
    terms = update(obj, batch, opt)
    assert not torch.equal(before, obj.student.prediction[0].weight)
    proof = frozen_state_proof(obj, opt)
    assert proof["prediction_head_changed"] is True
    assert proof["prediction_head_sha256"] != proof["parent_prediction_head_sha256"]
    assert proof["frozen_modules_eval"] is True
    assert proof["relay_order_randomization"] is False
    assert proof["trainable_parameters"] == 131585
    assert proof["teacher_updates"] == proof["center_updates"] == 0
    assert not any(k.startswith("ssl") for k in terms)
    for name, value in obj.state_dict().items():
        if not name.startswith("student.prediction."):
            assert torch.equal(initial[name], value), name
    assert opt.routes[0]["optimizer"] == "muon"
    assert {r["name"] for r in opt.routes if r["optimizer"] == "adamw"} == {
        "prediction.0.bias",
        "prediction.2.weight",
        "prediction.2.bias",
    }
    obj.ssl1_cls_center.add_(1)
    with pytest.raises(ValueError, match="state changed"):
        frozen_state_proof(obj, opt)


def test_child_checkpoint_resumes_head_moments_and_rng_exactly(tmp_path):
    obj, batch, opt = fixture()
    rng = np.random.default_rng(23)
    update(obj, batch, opt)
    path = tmp_path / "child.pt"
    save_checkpoint(path, obj, opt, identity={}, progress={"epoch": 1}, generator=rng)
    expected_draw = torch.rand(3)
    update(obj, batch, opt)
    expected = copy.deepcopy(obj.state_dict()), copy.deepcopy(opt.state_dict())
    resumed, _, restored_opt = fixture()
    resumed.parent_frozen_sha256 = obj.parent_frozen_sha256
    resumed.parent_prediction_head_sha256 = obj.parent_prediction_head_sha256
    progress = load_checkpoint(path, resumed, restored_opt, identity={}, generator=rng)
    assert progress == {"epoch": 1}
    assert torch.equal(torch.rand(3), expected_draw)
    update(resumed, batch, restored_opt)
    assert_nested_equal(resumed.state_dict(), expected[0])
    assert_nested_equal(restored_opt.state_dict(), expected[1])
    frozen_state_proof(resumed, restored_opt)


def test_missing_parent_or_unexpected_head_is_rejected():
    obj, batch, opt = fixture()
    del obj.parent_frozen_sha256
    with pytest.raises(ValueError, match="parent must"):
        update(obj, batch, opt)
    parent, _ = new_fixture("B0")
    with pytest.raises(ValueError, match="architecture differs"):
        FrozenReadoutObjective(parent.student, lambda1=0, lambda2=0)


def test_scientific_parent_binding_strict_load_and_tamper_rejection(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gradpert.config.v2 import V2Options
    from gradpert.data._io import atomic_json, read_json
    from gradpert.execution.v2_training_stage import training_stage_evidence
    from gradpert.hashing import sha256_file, sha256_json
    from gradpert.training.v2.readout import restore_frozen_parent

    obj, _, opt = fixture()
    with torch.no_grad():
        next(obj.teacher.parameters()).add_(0.25)
    obj.parent_frozen_sha256 = frozen_digest(obj)
    parent = tmp_path / "parent"
    (parent / "fit").mkdir(parents=True)
    prior = {
        "model": {"parameters": {"training_mode": {"value": "ssl_then_joint"}}},
        "data": {},
        "evaluation": {},
        "artifacts": {},
        "training": {
            "train_batch_size": {},
            "eval_batch_size": {},
            "weight_decay": {},
            "learning_rate": {},
            "scheduler": {},
        },
    }
    manifest = {
        "run_id": "parent-r9",
        "source": {"commit": "a" * 40, "published_commit": "a" * 40, "dirty": False},
        "config_sha256": "b" * 64,
        "resolved_config_sha256": sha256_json(prior),
        "data": {"canonical_sha256": "c" * 64, "loss_protocol": {"mode": "joint"}},
    }
    opt.steps = 450
    opt.training_stage = "joint"
    checkpoint = parent / "fit/last.pt"
    save_checkpoint(
        checkpoint,
        obj,
        opt,
        identity=manifest,
        progress={"history": [{"epoch": i} for i in (1, 2, 3)]},
        generator=np.random.default_rng(1),
    )
    sha = sha256_file(checkpoint)
    for name, payload in {
        "resolved_config.json": prior,
        "run_manifest.json": manifest,
        "fit/history.json": [{"epoch": i} for i in (1, 2, 3)],
        "fit/epoch_state.json": {
            "identity": manifest,
            "epoch": 3,
            "budget": [3, 150],
            "best": None,
            "last": {"file": "last.pt", "sha256": sha, "epoch": 3},
        },
    }.items():
        atomic_json(parent / name, payload)
    atomic_json(parent / "TRAIN_COMPLETE.json", training_stage_evidence(parent, manifest, 3))
    options = SimpleNamespace(
        training_mode="frozen_readout",
        readout_parent_run_root=str(parent),
        readout_parent_epoch=3,
        readout_parent_run_id="parent-r9",
        readout_parent_source_commit="a" * 40,
        readout_parent_checkpoint_sha256=sha,
    )
    child, _, child_opt = fixture()
    monkeypatch.setattr(V2Options, "parse_parameters", lambda _: (None, options))
    config = SimpleNamespace(
        model=SimpleNamespace(parameters={}),
        training=SimpleNamespace(max_epochs=SimpleNamespace(value=2)),
        model_dump=lambda **_: prior,
    )
    # Distinct Teacher values must be retained independently from Student.
    provenance = restore_frozen_parent(
        config,
        data_identity={**manifest["data"], "loss_protocol": {"mode": "frozen_readout"}},
        objective=child,
        optimizer=child_opt,
        server_root=tmp_path,
    )
    assert provenance["parent_full_test_complete"] is False
    assert child_opt.steps == 0 and not child_opt.adamw.state and not child_opt.muon.state
    assert_nested_equal(obj.state_dict(), child.state_dict())
    assert provenance["parent_evidence_sha256"]["TRAIN_COMPLETE.json"]
    terminal = read_json(parent / "TRAIN_COMPLETE.json")
    terminal["identity"]["source"]["commit"] = "d" * 40
    atomic_json(parent / "TRAIN_COMPLETE.json", terminal)
    with pytest.raises(ValueError, match="TRAIN_COMPLETE differs"):
        restore_frozen_parent(
            config,
            data_identity=manifest["data"],
            objective=child,
            optimizer=child_opt,
            server_root=tmp_path,
        )


def test_frozen_proof_rejects_nonpersistent_training_mode_and_relay_order_drift():
    obj, _, opt = fixture()
    before = frozen_digest(obj)
    obj.student.graph.train()
    assert frozen_digest(obj) == before
    with pytest.raises(ValueError, match="eval mode"):
        frozen_state_proof(obj, opt)
    obj.train()
    frozen_state_proof(obj, opt)
    obj.teacher.set_relay_order_randomization(True)
    assert frozen_digest(obj) == before
    with pytest.raises(ValueError, match="relay order"):
        frozen_state_proof(obj, opt)
    obj.eval()
    frozen_state_proof(obj, opt)
