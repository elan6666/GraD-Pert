"""One-factor SSL1 dose, sealed dispatch and native loss/update invariants."""

import copy
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round1 as first  # noqa: E402
import generate_representation_round6 as round6  # noqa: E402
import generate_representation_round7 as round7  # noqa: E402
import test_cr_medium_adapter as observer_contract  # noqa: E402
import test_representation_round4 as dispatch_contract  # noqa: E402
from test_br400_group import prepared as prepared  # noqa: E402
from test_components import fixture  # noqa: E402
from test_fgpo_group import new_queue as new_queue  # noqa: E402


@pytest.fixture
def family_root(tmp_path):
    source = tmp_path / "source"
    anchor = source / first.BASELINE
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes((ROOT / first.BASELINE).read_bytes())
    output = source / "configs/v2" / round7.FAMILY
    round7.generate(source, output)
    return source, output


def test_exact_single_parent_difference_and_standalone_committed_config(family_root):
    source, output = family_root
    parent = round6.configs(source)["R6-EMA99"]
    before = copy.deepcopy(parent)
    candidate = round7.configs(source)["R7-S1D05"]
    parameter = candidate["model"]["parameters"].pop("lambda1")
    assert parameter == dict(value=0.05, source="project_preregistered", reference=round7.REFERENCE)
    candidate["model"]["parameters"]["lambda1"] = copy.deepcopy(
        parent["model"]["parameters"]["lambda1"]
    )
    assert candidate == parent == before == round6.configs(source)["R6-EMA99"]
    assert round7.CHANGES == {**round6.CHANGES, "lambda1": 0.05}
    value = round7.verify_manifest(source, output / "manifest.json")
    assert value["priority"] == value["preflight_priority"] == ["R7-S1D05"]
    assert value["rows"][0]["changes"] == round7.CHANGES
    assert value["repeat_baseline"] is False and "R6-EMA99" in value["reused_controls"]

    committed = round7.verify_manifest(ROOT, ROOT / "configs/v2" / round7.FAMILY / "manifest.json")
    path = ROOT / committed["rows"][0]["config"]
    raw = yaml.safe_load(path.read_text())
    assert not {"defaults", "extends", "inherit"}.intersection(raw)
    assert sha256_file(ROOT / round7.PARENT_CONFIG) == round7.PARENT_CONFIG_SHA256
    original = yaml.safe_load((ROOT / round7.PARENT_CONFIG).read_text())
    raw["model"]["parameters"]["lambda1"] = original["model"]["parameters"]["lambda1"]
    assert raw == original
    resolved = load_experiment_config(path)
    assert resolved.training.max_epochs.value == 2
    assert committed["rows"][0]["sha256"] == sha256_file(path)
    locked = yaml.safe_load(path.read_text())["model"]["parameters"]["lambda2"]
    assert locked == original["model"]["parameters"]["lambda2"]
    assert locked["value"] == 1 and locked["source"] == "user_locked"


def test_generation_preserves_all_historical_configs_and_native_sources(family_root, tmp_path):
    paths = [*(ROOT / "src").rglob("*.py")]
    paths.extend(
        p
        for n in range(1, 7)
        for p in (ROOT / "configs/v2").glob(f"representation_round{n}_*/**/*")
        if p.is_file()
    )
    before = {p: sha256_file(p) for p in paths}
    source, output = family_root
    round7.generate(source, tmp_path / "source/configs/v2/second-output")
    round7.verify_manifest(source, output / "manifest.json")
    assert {p: sha256_file(p) for p in paths} == before
    assert not (source / "src").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "lambda1",
        "lambda2",
        "lambda2_source",
        "ema",
        "space",
        "budget",
        "parent",
        "gate",
        "test",
        "repeat",
    ],
)
def test_rehash_cannot_bypass_registered_scope(family_root, mutation):
    source, output = family_root
    path = output / "manifest.json"
    value = json.loads(path.read_text())
    if mutation in {"lambda1", "lambda2", "lambda2_source", "ema", "space", "budget"}:
        target = source / value["rows"][0]["config"]
        config = yaml.safe_load(target.read_text())
        parameters = config["model"]["parameters"]
        if mutation == "budget":
            config["training"]["max_epochs"]["value"] = 3
        elif mutation == "lambda2_source":
            parameters["lambda2"]["source"] = "project_preregistered"
        else:
            key, changed = {
                "lambda1": ("lambda1", 0.1),
                "lambda2": ("lambda2", 0.5),
                "ema": ("teacher_end", 1.0),
                "space": ("population_aux_space", "response_cls"),
            }[mutation]
            parameters[key]["value"] = changed
        target.write_text(yaml.safe_dump(config))
        value["rows"][0]["sha256"] = sha256_file(target)
    elif mutation == "parent":
        value["coefficient_decision"]["evidence"]["parent_training_commit"] = "b" * 40
    elif mutation == "gate":
        value["zero_update_launch_gate"]["replay_max_abs_error_lte"] = 1e-4
    elif mutation == "test":
        value["research_iteration"]["test_results_for_selection"] = True
    else:
        value["rows"].append({**value["rows"][0], "name": "R6-EMA99"})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        round7.verify_manifest(source, path)


def test_identity_budget_and_pending_gate_are_explicit(family_root):
    source, output = family_root
    value = round7.verify_manifest(source, output / "manifest.json")
    for key, expected in dict(
        epochs=2,
        seed=1,
        global_batch=320,
        common_microbatch=80,
        world_size=2,
        accumulation=2,
        preflight_updates=10,
        eval_batch=128,
        eval_capacity_reference_batch=128,
    ).items():
        assert value[key] == expected
    assert value["selection"] == "final_epoch" and value["test_roles"] == ["last"]
    assert value["test_timing"] == "after_research_design_lock"
    assert value["research_iteration"]["test_results_for_selection"] is False
    assert value["research_iteration"]["round"] == 7
    assert value["research_iteration"]["authorized_rounds"] is None
    evidence = value["coefficient_decision"]["evidence"]
    assert evidence["parent_training_commit"] == round7.PARENT_TRAINING_SOURCE
    assert evidence["parent_config_sha256"] == round7.PARENT_CONFIG_SHA256
    assert evidence["parent_checkpoint_sha256"] == round7.PARENT_CHECKPOINT_SHA256
    assert evidence["parent_run_id"] == round7.PARENT_RUN
    assert evidence["parent_checkpoint_epoch"] == 2 and evidence["parent_checkpoint_role"] == "last"
    assert evidence["panel_rows"] == 128 and evidence["panel_sha256"] == round7.PANEL_SHA256
    gate = value["zero_update_launch_gate"]
    assert gate["status"] == "required_before_launch; receipt supplied independently"
    assert "passed" not in gate and "complete_sha256" not in gate
    assert gate["replay_max_abs_error_lte"] == 2e-6
    assert gate["complete_populations_per_endpoint"] == 1 and gate["population_size"] == 320
    assert gate["updates"] == 0 and "not eight-population completion" in gate["scope"]


def test_training_and_both_eval_dispatch_keep_existing_gates(family_root, new_queue, monkeypatch):
    dispatch_contract.test_native_two_epoch_training_and_both_evaluation_bindings(
        family_root, round7, new_queue, monkeypatch
    )


@pytest.fixture
def medium_case(tmp_path):
    parent, identity, queue, args, kwargs = observer_contract.cr_case.__wrapped__(tmp_path, round6)
    args.config.write_text(yaml.safe_dump(round7.configs(ROOT)["R7-S1D05"]))
    identity["run_id"] = "R7-S1D05-run"
    queue["family"] = round7.SCHEMA
    row = queue["rows"][0]
    row["name"] = kwargs["arm"] = "R7-S1D05"
    row["plan"].update(
        run_id=identity["run_id"], config_sha256=observer_contract.medium.sha(args.config)
    )
    return parent, identity, queue, args, kwargs


def test_medium_and_panel_keep_matched_population(medium_case):
    observer_contract.test_cr_binding_preserves_original_parent_and_panel_binding(medium_case)


@pytest.mark.parametrize("population_drift", [False, True])
def test_medium_dispatch_uses_native_seal(medium_case, monkeypatch, population_drift):
    observer_contract.test_cr_medium_execute_binds_before_native_seal(
        medium_case, monkeypatch, population_drift
    )


@pytest.mark.parametrize("field,value", [("lambda1", 1.0), ("lambda2", 0.1), ("teacher_end", 1.0)])
def test_medium_rejects_wrong_dose_or_teacher(medium_case, field, value):
    parent, identity, _, args, kwargs = medium_case
    config = yaml.safe_load(args.config.read_text())
    config["model"]["parameters"][field]["value"] = value
    args.config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="parent binding requires"):
        observer_contract.medium.cr_parent_binding(parent, identity, **kwargs)


def test_unavailable_full_population_terms_stay_na(tmp_path):
    research = observer_contract.research
    root, run = tmp_path / "medium", tmp_path / "run"
    row = dict(
        name="R7-S1D05",
        plan=dict(
            run_root=str(run),
            config_sha256="c" * 64,
            config=str(
                ROOT / "configs/v2" / round7.FAMILY / "R7-S1D05/gradpert_v2/nadig_jurkat.yaml"
            ),
        ),
    )
    queue = dict(output=str(tmp_path), source_commit="a" * 40, family=round7.SCHEMA)
    atomic_json(root / "analysis/summary.json", dict(global_features={}, global_heads={}))
    unavailable = {
        key: "requires_complete_global_population" for key in ("au_alignment", "cr_negative")
    }
    atomic_json(
        root / "analysis/deep.json",
        [
            dict(
                condition_id="p",
                joint=2,
                loss_terms=dict(prediction=1, ssl2_dino=1),
                diagnostic_objective_scope=dict(
                    unavailable_terms=unavailable,
                    joint_scope="prediction_and_ssl_excluding_unavailable_representation_terms",
                ),
            )
        ],
    )
    atomic_json(root / "analysis/gradients.json", [])
    atomic_json(run / "fit/history.json", [dict(epoch=1), dict(epoch=2)])
    atomic_json(
        research.validation_path(queue, row) / "COMPLETE.json",
        dict(result=dict(metrics=[], metric_gene_groups={})),
    )
    research.render_medium_report(queue, row, root)
    loss = next(csv.DictReader((root / "report/frozen_losses.csv").open()))
    for key, reason in unavailable.items():
        assert loss[key] == "" and loss["unavailable_terms." + key] == reason
    assert loss["joint"] == "2"
    report = (root / "report/REPORT_CN.md").read_text()
    assert "lambda1" in report and "user_locked 1" in report and "A/CR仍为NA" in report


def test_report_keeps_raw_losses_and_actual_weighted_contributions_separate():
    config = ROOT / "configs/v2" / round7.FAMILY / "R7-S1D05/gradpert_v2/nadig_jurkat.yaml"
    history = [
        dict(
            epoch=1,
            training=dict(
                prediction=0.2,
                ssl1_condition=2.0,
                ssl2_dino=3.0,
                ssl2_ibot=4.0,
                au_weighted=0.003,
                cr_weighted=-0.001,
                joint_loss=7.302,
            ),
        ),
        dict(epoch=2),
    ]
    before = copy.deepcopy(history)
    first_row, missing = observer_contract.research.ssl_loss_contributions(history, config)
    assert first_row["ssl1_before_overall_weight"] == 2.0
    assert first_row["ssl1_contribution"] == pytest.approx(0.1)
    assert first_row["ssl2_before_overall_weight"] == first_row["ssl2_contribution"] == 7.0
    assert first_row["au_weighted"] == 0.003 and first_row["cr_weighted"] == -0.001
    assert first_row["joint_loss"] == 7.302
    assert missing["ssl1_contribution"] is None and missing["ssl2_contribution"] is None
    assert missing["prediction"] is None and missing["cr_weighted"] is None
    assert history == before


def test_native_ssl1_dose_scales_only_its_full_student_gradient():
    model, batch = fixture()
    objectives = [
        JointObjective(
            copy.deepcopy(model),
            lambda1=dose,
            lambda2=1,
            ssl1_weights=(0.8, 0.4, 0),
            ssl2_weights=(0.8, 0.4, 0),
        ).train()
        for dose in (0, 1, 0.05)
    ]
    losses, metrics, gradients = [], [], []
    pending = []
    for objective in objectives:
        before = copy.deepcopy(objective.state_dict())
        parameters = [p for p in objective.student.parameters() if p.requires_grad]
        with torch.random.fork_rng():
            torch.manual_seed(73)
            loss, terms = objective(batch)
        raw = torch.autograd.grad(loss, parameters, allow_unused=True)
        gradients.append(
            [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, raw, strict=True)]
        )
        losses.append(loss.detach())
        metrics.append({k: v.detach() for k, v in terms.items()})
        pending.append(copy.deepcopy(objective.pending))
        for name, value in objective.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert all(p.grad is None for p in objective.teacher.parameters())
    assert metrics[1].keys() == metrics[2].keys()
    for key in metrics[1]:
        torch.testing.assert_close(metrics[1][key], metrics[2][key], rtol=0, atol=0)
        if not key.startswith("ssl1_"):
            torch.testing.assert_close(metrics[0][key], metrics[1][key], rtol=0, atol=0)
    torch.testing.assert_close(losses[2] - losses[0], (losses[1] - losses[0]) * 0.05)
    assert pending[1].keys() == pending[2].keys()
    for name in pending[1]:
        for a, b in zip(pending[1][name], pending[2][name], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    ssl1_norm = 0.0
    for base, parent, candidate in zip(*gradients, strict=True):
        isolated = parent - base
        ssl1_norm += float(isolated.square().sum())
        torch.testing.assert_close(candidate - base, isolated * 0.05, rtol=3e-4, atol=2e-7)
    assert ssl1_norm > 1e-6


@pytest.mark.parametrize("dose", [1.0, 0.05])
def test_native_teacher_and_centers_commit_once_per_successful_update(dose):
    model, batch = fixture()
    objective = JointObjective(
        model,
        lambda1=dose,
        lambda2=1,
        ssl1_weights=(0.8, 0.4, 0),
        ssl2_weights=(0.8, 0.4, 0),
    )
    optimizer = V2Optimizer(model, lr=0.001, weight_decay=0)
    original_commit = objective.commit_statistics
    calls = []

    def observed_commit(momentum):
        before = {n: p.clone() for n, p in objective.teacher.named_parameters()}
        student = dict(objective.student.named_parameters())
        center_names = ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node")
        centers = {n: getattr(objective, n + "_center").clone() for n in center_names}
        stats = {n: torch.stack(objective.pending[n]).sum(0) for n in center_names}
        original_commit(momentum)
        for name, teacher in objective.teacher.named_parameters():
            torch.testing.assert_close(teacher, before[name].lerp(student[name], 1 - momentum))
        for name in center_names:
            expected = centers[name].lerp(stats[name][0] / stats[name][1], 0.1)
            torch.testing.assert_close(getattr(objective, name + "_center"), expected)
        calls.append((optimizer.steps, momentum))

    objective.commit_statistics = observed_commit
    for _ in range(2):
        optimizer_step(
            objective, optimizer, batch, microbatch=1, lr=0.001, momentum=0.99, bf16=False
        )
    assert optimizer.steps == 2 and calls == [(1, 0.99), (2, 0.99)]
    assert objective.teacher_temperature == 0.04 and not objective.pending
    teacher = copy.deepcopy(objective.teacher.state_dict())
    centers = {n: v.clone() for n, v in objective.named_buffers() if "center" in n}
    bad = replace(batch, truth=torch.full_like(batch.truth, float("nan")))
    with pytest.raises(FloatingPointError):
        optimizer_step(objective, optimizer, bad, microbatch=1, lr=0.001, momentum=0.99, bf16=False)
    assert optimizer.steps == 2 and len(calls) == 2 and not objective.pending
    assert objective.teacher_temperature == 0.04
    for name, value in objective.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher[name], rtol=0, atol=0)
    for name, value in objective.named_buffers():
        if name in centers:
            torch.testing.assert_close(value, centers[name], rtol=0, atol=0)
