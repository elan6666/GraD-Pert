"""One-factor EMA family and the unchanged native successful-update lifecycle."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.step_schedule import EndpointLRWarmupCosine
from gradpert.hashing import sha256_file
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.lifecycle import fit
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round1 as first  # noqa: E402
import generate_representation_round5 as round5  # noqa: E402
import generate_representation_round6 as round6  # noqa: E402
from test_components import fixture  # noqa: E402


@pytest.fixture
def family_root(tmp_path):
    source = tmp_path / "source"
    anchor = source / first.BASELINE
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes((ROOT / first.BASELINE).read_bytes())
    output = source / "configs/v2" / round6.FAMILY
    round6.generate(source, output)
    return source, output


def test_only_parent_teacher_end_changes_and_parent_is_not_mutated(family_root):
    source, output = family_root
    before = round5.configs(source)["R5-PROJ"]
    parent = copy.deepcopy(before)
    candidate = round6.configs(source)["R6-EMA99"]
    parameters = candidate["model"]["parameters"]
    assert parameters["teacher_start"] == parent["model"]["parameters"]["teacher_start"]
    assert parameters["teacher_start"]["value"] == parameters["teacher_end"]["value"] == 0.99
    assert parent["model"]["parameters"]["teacher_end"]["value"] == 1.0
    end_record = parameters["teacher_end"]
    assert end_record["source"] == "project_preregistered"
    assert end_record["reference"] == round6.REFERENCE
    parameters["teacher_end"] = copy.deepcopy(parent["model"]["parameters"]["teacher_end"])
    assert candidate == parent == before == round5.configs(source)["R5-PROJ"]
    manifest = round6.verify_manifest(source, output / "manifest.json")
    assert manifest["priority"] == manifest["preflight_priority"] == ["R6-EMA99"]
    assert manifest["repeat_baseline"] is False
    assert "R5-PROJ" in manifest["reused_controls"]
    assert round6.CHANGES == {**round5.CHANGES, "teacher_end": 0.99}
    assert manifest["rows"][0]["changes"] == round6.CHANGES


def test_committed_config_is_standalone_and_preserves_exact_parent_scientific_fields():
    path = ROOT / "configs/v2" / round6.FAMILY / "manifest.json"
    manifest = round6.verify_manifest(ROOT, path)
    target = ROOT / manifest["rows"][0]["config"]
    assert sha256_file(ROOT / round6.PARENT_CONFIG) == round6.PARENT_CONFIG_SHA256
    parent = yaml.safe_load((ROOT / round6.PARENT_CONFIG).read_text())
    value = yaml.safe_load(target.read_text())
    resolved = load_experiment_config(target)
    assert resolved.training.max_epochs.value == 2
    assert "defaults" not in value and "extends" not in value and "inherit" not in value
    value["model"]["parameters"]["teacher_end"] = parent["model"]["parameters"]["teacher_end"]
    assert value == parent
    assert manifest["rows"][0]["sha256"] == sha256_file(target)
    for key, expected in (
        ("epochs", 2),
        ("seed", 1),
        ("global_batch", 320),
        ("common_microbatch", 80),
        ("world_size", 2),
        ("accumulation", 2),
        ("preflight_updates", 10),
        ("eval_batch", 128),
        ("eval_capacity_reference_batch", 128),
    ):
        assert manifest[key] == expected
    assert manifest["selection"] == "final_epoch"
    assert manifest["test_roles"] == ["last"]
    assert manifest["test_timing"] == "after_research_design_lock"
    assert manifest["research_iteration"]["test_results_for_selection"] is False


def test_source_and_parent_files_unchanged_by_generation(tmp_path):
    before = {str(p.relative_to(ROOT)): sha256_file(p) for p in (ROOT / "src").rglob("*.py")}
    parent_hash = sha256_file(ROOT / round6.PARENT_CONFIG)
    manifest_hash = sha256_file(ROOT / "configs/v2/representation_round5_20261023/manifest.json")
    source = tmp_path / "source"
    target = source / first.BASELINE
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / first.BASELINE).read_bytes())
    output = source / "configs/v2" / round6.FAMILY
    round6.generate(source, output)
    round6.verify_manifest(source, output / "manifest.json")
    assert {
        str(p.relative_to(ROOT)): sha256_file(p) for p in (ROOT / "src").rglob("*.py")
    } == before
    assert sha256_file(ROOT / round6.PARENT_CONFIG) == parent_hash
    assert (
        sha256_file(ROOT / "configs/v2/representation_round5_20261023/manifest.json")
        == manifest_hash
    )
    assert not (source / "src").exists()


@pytest.mark.parametrize("mutation", ["ema", "temperature", "budget", "evidence", "repeat", "test"])
def test_rehashed_unregistered_scientific_or_evidence_drift_rejected(family_root, mutation):
    source, output = family_root
    path = output / "manifest.json"
    value = json.loads(path.read_text())
    if mutation in ("ema", "temperature", "budget"):
        target = source / value["rows"][0]["config"]
        config = yaml.safe_load(target.read_text())
        if mutation == "budget":
            config["training"]["max_epochs"]["value"] = 4
        else:
            key = "teacher_end" if mutation == "ema" else "teacher_temperature_start"
            config["model"]["parameters"][key] = {
                "value": 0.98,
                "source": "project_preregistered",
                "reference": "unregistered",
            }
        target.write_text(yaml.safe_dump(config))
        value["rows"][0]["sha256"] = sha256_file(target)
    elif mutation == "evidence":
        value["coefficient_decision"]["evidence"]["clean_global_signal"]["complete_sha256"] = (
            "a" * 64
        )
    elif mutation == "repeat":
        value["rows"].append({**value["rows"][0], "name": "R5-PROJ"})
    else:
        value["research_iteration"]["test_results_for_selection"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        round6.verify_manifest(source, path)


def test_zero_update_evidence_and_source_roles_remain_explicit(family_root):
    source, output = family_root
    value = round6.verify_manifest(source, output / "manifest.json")
    decision = value["coefficient_decision"]
    evidence = decision["evidence"]
    assert decision["updates"] == evidence["clean_global_signal"]["updates"] == 0
    assert evidence["parent_training_commit"] == round6.PARENT_TRAINING_SOURCE
    assert evidence["clean_global_signal"]["analysis_commit"] == round6.BASELINE_SOURCE
    assert evidence["native_signal"]["analysis_commit"] != round6.PARENT_TRAINING_SOURCE
    assert evidence["panel_rows"] == 128 and evidence["panel_sha256"] == round6.PANEL_SHA256
    assert evidence["parent_checkpoint_epoch"] == 2 and evidence["parent_checkpoint_role"] == "last"
    assert len(evidence["clean_global_signal"]["complete_sha256"]) == 64
    assert "successful optimizer update" in value["schedule_semantics"]["ema"]
    assert value["teacher_tracking"]["training_views"].startswith("native R5 views")
    assert "not a causal diagnosis" in decision["rationale"]
    assert len(value["teacher_tracking"]["native_source_git_tree"]) == 40


def test_native_lifecycle_equal_endpoints_commit_constant_ema_after_successful_updates(tmp_path):
    model, batch = fixture()
    objective = JointObjective(model, lambda1=0, lambda2=0)
    optimizer = V2Optimizer(model, lr=0.001, weight_decay=0)
    original_commit = objective.commit_statistics
    commits = []

    def observed_commit(momentum):
        before = objective.teacher.prediction[0].weight.detach().clone()
        student = objective.student.prediction[0].weight.detach().clone()
        original_commit(momentum)
        expected = before.lerp(student, 1 - momentum)
        torch.testing.assert_close(objective.teacher.prediction[0].weight, expected, rtol=0, atol=0)
        commits.append((optimizer.steps, momentum))

    objective.commit_statistics = observed_commit
    journal = fit(
        objective,
        optimizer,
        root=tmp_path,
        identity={"source": "synthetic-r6-test"},
        generator=np.random.default_rng(6),
        epochs=2,
        steps_per_epoch=4,
        batches=lambda _epoch: [batch] * 4,
        validate=None,
        schedule=EndpointLRWarmupCosine(0.001, 0.0002, 0.16),
        teacher_start=0.99,
        teacher_end=0.99,
        microbatch=2,
        bf16=False,
    )
    assert commits == [(i, 0.99) for i in range(1, 9)]
    assert optimizer.steps == 8
    assert journal["last"]["epoch"] == 2 and journal["selection_metric"] == "final_epoch"
    assert journal["contract"]["teacher_start"] == journal["contract"]["teacher_end"] == 0.99
    history = json.loads((tmp_path / "history.json").read_text())
    assert all(row["training"]["teacher_momentum"] == pytest.approx(0.99) for row in history)


def test_native_nonfinite_failure_does_not_commit_ema_or_advance_optimizer():
    model, batch = fixture()
    objective = JointObjective(model, lambda1=0, lambda2=0)
    optimizer = V2Optimizer(model, lr=0.001, weight_decay=0)
    commits = []
    original_commit = objective.commit_statistics

    def observed_commit(momentum):
        original_commit(momentum)
        commits.append((optimizer.steps, momentum))

    objective.commit_statistics = observed_commit
    optimizer_step(objective, optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False)
    teacher = {name: value.clone() for name, value in objective.teacher.state_dict().items()}
    bad = replace(batch, truth=torch.full_like(batch.truth, float("nan")))
    with pytest.raises(FloatingPointError):
        optimizer_step(objective, optimizer, bad, microbatch=2, lr=0.001, momentum=0.99, bf16=False)
    assert optimizer.steps == 1 and commits == [(1, 0.99)]
    assert objective.pending == {}
    for name, value in objective.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher[name], rtol=0, atol=0)
