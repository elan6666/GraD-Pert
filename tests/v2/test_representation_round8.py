"""R6-parent CR0 family, immutable history and exact native dispatch contracts."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.hashing import sha256_file
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round1 as first  # noqa: E402
import generate_representation_round6 as round6  # noqa: E402
import generate_representation_round8 as round8  # noqa: E402
import test_cr_medium_adapter as observer  # noqa: E402
import test_representation_round4 as dispatch  # noqa: E402
from test_br400_group import prepared as prepared  # noqa: E402
from test_components import fixture  # noqa: E402
from test_fgpo_group import new_queue as new_queue  # noqa: E402


@pytest.fixture
def family_root(tmp_path):
    source = tmp_path / "source"
    anchor = source / first.BASELINE
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes((ROOT / first.BASELINE).read_bytes())
    output = source / "configs/v2" / round8.FAMILY
    round8.generate(source, output)
    return source, output


def test_exact_parent_record_difference_and_self_contained_config(family_root):
    source, output = family_root
    before = round6.configs(source)["R6-EMA99"]
    candidate = round8.configs(source)["R8-CR0"]
    assert candidate["model"]["parameters"]["cr_weight"] == dict(
        value=0.0, source="project_preregistered", reference=round8.REFERENCE
    )
    candidate["model"]["parameters"]["cr_weight"] = copy.deepcopy(
        before["model"]["parameters"]["cr_weight"]
    )
    assert candidate == before == round6.configs(source)["R6-EMA99"]
    assert round8.CHANGES == {**round6.CHANGES, "cr_weight": 0.0}
    manifest = round8.verify_manifest(source, output / "manifest.json")
    assert manifest["rows"][0]["changes"] == round8.CHANGES
    committed = round8.verify_manifest(ROOT, ROOT / "configs/v2" / round8.FAMILY / "manifest.json")
    target = ROOT / committed["rows"][0]["config"]
    assert sha256_file(ROOT / round8.PARENT_CONFIG) == round8.PARENT_CONFIG_SHA256
    raw = yaml.safe_load(target.read_text())
    assert not {"defaults", "extends", "inherit"}.intersection(raw)
    parent = yaml.safe_load((ROOT / round8.PARENT_CONFIG).read_text())
    raw["model"]["parameters"]["cr_weight"] = parent["model"]["parameters"]["cr_weight"]
    assert raw == parent
    _, options = V2Options.parse_parameters(load_experiment_config(target).model.parameters)
    assert (options.lambda1, options.lambda2, options.cr_weight) == (1.0, 1.0, 0.0)
    assert (options.teacher_start, options.teacher_end) == (0.99, 0.99)
    assert options.population_aux_space == "ssl2_cls_bottleneck"
    assert parent["model"]["parameters"]["lambda2"]["source"] == "user_locked"


def test_generation_does_not_change_any_old_config_or_native_source(family_root):
    source, _ = family_root
    paths = [*(ROOT / "src").rglob("*.py")]
    paths += [
        p
        for n in range(1, 8)
        for p in (ROOT / "configs/v2").glob(f"representation_round{n}_*/**/*")
        if p.is_file()
    ]
    before = {p: sha256_file(p) for p in paths}
    round8.generate(source, source / "configs/v2/another-output")
    assert {p: sha256_file(p) for p in paths} == before
    assert not (source / "src").exists()


@pytest.mark.parametrize(
    "mutation",
    ["cr", "lambda1", "lambda2", "lambda2_source", "ema", "space", "budget", "test", "parent"],
)
def test_rehashed_drift_cannot_bypass_frozen_family(family_root, mutation):
    source, output = family_root
    path = output / "manifest.json"
    value = json.loads(path.read_text())
    if mutation in ("test", "parent"):
        if mutation == "test":
            value["research_iteration"]["test_results_for_selection"] = True
        else:
            value["coefficient_decision"]["evidence"]["parent_checkpoint_sha256"] = "b" * 64
    else:
        target = source / value["rows"][0]["config"]
        raw = yaml.safe_load(target.read_text())
        parameters = raw["model"]["parameters"]
        if mutation == "budget":
            raw["training"]["max_epochs"]["value"] = 4
        elif mutation == "lambda2_source":
            parameters["lambda2"]["source"] = "project_preregistered"
        else:
            key, changed = {
                "cr": ("cr_weight", 0.00004),
                "lambda1": ("lambda1", 0.05),
                "lambda2": ("lambda2", 0.5),
                "ema": ("teacher_end", 1.0),
                "space": ("population_aux_space", "response_cls"),
            }[mutation]
            parameters[key]["value"] = changed
        target.write_text(yaml.safe_dump(raw))
        value["rows"][0]["sha256"] = sha256_file(target)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        round8.verify_manifest(source, path)


def test_parent_evidence_budget_and_counterexamples_remain_explicit(family_root):
    source, output = family_root
    value = round8.verify_manifest(source, output / "manifest.json")
    assert value["priority"] == value["preflight_priority"] == ["R8-CR0"]
    assert value["repeat_baseline"] is False and value["research_iteration"]["round"] == 8
    assert value["research_iteration"]["authorized_rounds"] is None
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
    evidence = value["coefficient_decision"]["evidence"]
    assert evidence["parent_training_commit"] == round8.PARENT_TRAINING_SOURCE
    assert evidence["parent_config_sha256"] == round8.PARENT_CONFIG_SHA256
    assert evidence["parent_checkpoint_sha256"] == round8.PARENT_CHECKPOINT_SHA256
    assert evidence["parent_run_id"] == round8.PARENT_RUN
    assert evidence["r7_dose_counterexample"]["pearson_below_r6"] == 14
    full = evidence["full_population_observation"]
    assert full["observed_populations_per_endpoint"] == 1 and full["updates"] == 0
    assert full["original_planned_populations"] == 8 and full["replay_tolerance"] == 2e-6
    assert (
        full["r5_cr0_total_mse_cosine_counterexample"][1]
        < full["r5_cr0_total_mse_cosine_counterexample"][0]
    )
    assert value["coding_rate"]["enabled"] is False
    assert "missing measurements stay NA" in value["coding_rate"]["partial_diagnostics"]


def test_native_training_and_dual_eval_dispatch(family_root, new_queue, monkeypatch):
    dispatch.test_native_two_epoch_training_and_both_evaluation_bindings(
        family_root, round8, new_queue, monkeypatch
    )


@pytest.fixture
def medium_case(tmp_path):
    parent, identity, queue, args, kwargs = observer.cr_case.__wrapped__(tmp_path, round6)
    args.config.write_text(yaml.safe_dump(round8.configs(ROOT)["R8-CR0"]))
    identity["data"]["loss_protocol"].pop("coding_rate")
    identity["run_id"] = "R8-CR0-run"
    queue["family"] = round8.SCHEMA
    row = queue["rows"][0]
    row["name"] = kwargs["arm"] = "R8-CR0"
    row["plan"].update(run_id=identity["run_id"], config_sha256=observer.medium.sha(args.config))
    return parent, identity, queue, args, kwargs


def test_native_medium_and_panel_bind_only_actual_alignment_protocol(medium_case):
    parent, identity, queue, args, kwargs = medium_case
    before = copy.deepcopy((parent, identity, queue))
    bound = observer.medium.cr_parent_binding(parent, identity, **kwargs)
    binding = bound["training_protocol_binding"]
    assert binding["changed_fields"] == ["data.loss_protocol.alignment_uniformity"]
    assert "coding_rate" not in bound["data"]["loss_protocol"]
    assert bound["schedule"] == parent["schedule"] and binding["sampling_changed"] is False
    assert observer.panel_parent_binding(parent, identity, queue, args) == binding
    assert (parent, identity, queue) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("cr_weight", 0.00004),
        ("lambda1", 0.05),
        ("lambda2", 0.1),
        ("teacher_end", 1.0),
        ("population_aux_space", "response_cls"),
    ],
)
def test_medium_rejects_undeclared_config(medium_case, field, value):
    parent, identity, _, args, kwargs = medium_case
    raw = yaml.safe_load(args.config.read_text())
    raw["model"]["parameters"][field]["value"] = value
    args.config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        observer.medium.cr_parent_binding(parent, identity, **kwargs)


def test_native_cr_disabled_keeps_alignment_and_does_not_invent_cr_history(monkeypatch):
    from gradpert.training.v2 import alignment_uniformity as auxiliary

    model, batch = fixture()
    batch = replace(batch, context_index=torch.zeros(2, dtype=torch.long))
    objective = JointObjective(
        model,
        lambda1=1,
        lambda2=1,
        loss_reduction="row_mean",
        ssl1_weights=(1, 0, 0),
        ssl2_weights=(1, 1, 0),
        au_alignment_weight=0.05,
        cr_weight=0,
        population_aux_space="ssl2_cls_bottleneck",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled CR executed")

    monkeypatch.setattr(auxiliary, "negative_coding_rate", forbidden)
    metrics = optimizer_step(
        objective,
        V2Optimizer(model, lr=0.001, weight_decay=0),
        batch,
        microbatch=1,
        lr=0.001,
        momentum=0.99,
        bf16=False,
    )
    assert objective.au_enabled and not objective.cr_enabled
    assert "au_alignment" in metrics and "au_alignment_weighted" in metrics
    assert not any(key.startswith("cr_") for key in metrics)
