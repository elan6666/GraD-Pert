"""R9 sealing, exact registration, immutable R6 recipe and population adapters."""

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import generate_representation_round1 as first  # noqa: E402
import generate_representation_round6 as r6  # noqa: E402
import generate_representation_round9 as family  # noqa: E402
import run_br400_group as runner  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402
import run_representation_postfit as postfit  # noqa: E402
import test_cr_medium_adapter as observer  # noqa: E402


@pytest.fixture
def family_root(tmp_path):
    source = tmp_path / "source"
    anchor = source / first.BASELINE
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes((ROOT / first.BASELINE).read_bytes())
    output = source / "configs/v2" / family.FAMILY
    family.generate(source, output)
    return source, output


def test_exact_r6_recipe_and_budget(family_root):
    source, output = family_root
    value = family.verify_manifest(source, output / "manifest.json")
    candidate = family.configs(source)[family.ARMS[0]]
    parent = r6.configs(source)["R6-EMA99"]
    assert candidate["training"]["max_epochs"]["value"] == 3
    for key in ("formal_run_policy", "max_epochs"):
        candidate["training"][key] = copy.deepcopy(parent["training"][key])
    for key in ("training_mode", "ssl_pretrain_epochs"):
        candidate["model"]["parameters"].pop(key)
    assert candidate == parent
    assert value["staged_joint"]["total_steps"] == 450
    assert value["staged_joint"]["joint_steps"] == 300
    assert value["staged_joint"]["total_budget_matched"] is False
    assert value["scientific_complete"] is False and value["repeat_baseline"] is False
    assert (
        value["coefficient_decision"]["evidence"]["parent_training_commit"]
        == family.PARENT_TRAINING_SOURCE
    )
    assert value["eval_batch"] == value["eval_capacity_reference_batch"] == 128
    assert value["schedule_semantics"]["sampler_epochs"] == [0, 0, 1]
    target = source / value["rows"][0]["config"]
    plan = dict(
        repository_root=str(source),
        config=str(target),
        resolved_config=load_experiment_config(target).model_dump(mode="json"),
    )
    assert runner.expected_epochs(plan) == 3
    assert family.SCHEMA in runner.RESEARCH_FAMILIES
    assert family.SCHEMA in medium.FAMILY_ARMS and family.SCHEMA in postfit.FAMILY_ARMS
    plan["resolved_config"]["model"]["parameters"]["ssl_pretrain_epochs"]["value"] = 2
    with pytest.raises(ValueError):
        runner.expected_epochs(plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cr_weight", 0),
        ("lambda1", 0.05),
        ("lambda2", 0.5),
        ("teacher_end", 1),
        ("ssl_pretrain_epochs", 2),
        ("training_mode", "ssl_then_mse"),
        ("population_aux_space", "response_cls"),
        ("budget", 4),
        ("lambda2_source", "project_preregistered"),
        ("endpoint", 2),
        ("test", True),
    ],
)
def test_rehashed_tampering_rejected(family_root, field, value):
    source, output = family_root
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    target = source / manifest["rows"][0]["config"]
    raw = yaml.safe_load(target.read_text())
    if field == "endpoint":
        manifest["diagnostic_endpoints"]["last"]["epoch"] = value
    elif field == "test":
        manifest["research_iteration"]["test_results_for_selection"] = value
    else:
        if field == "budget":
            raw["training"]["max_epochs"]["value"] = value
        elif field == "lambda2_source":
            raw["model"]["parameters"]["lambda2"]["source"] = value
        else:
            raw["model"]["parameters"][field]["value"] = value
        target.write_text(yaml.safe_dump(raw))
        manifest["rows"][0]["sha256"] = sha256_file(target)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        family.verify_manifest(source, path)


@pytest.fixture
def medium_case(tmp_path):
    parent, identity, queue, args, kwargs = observer.cr_case.__wrapped__(tmp_path, r6)
    args.config.write_text(yaml.safe_dump(family.configs(ROOT)[family.ARMS[0]]))
    identity["data"]["loss_protocol"]["staged_training"] = dict(
        mode="ssl_then_joint",
        pretrain_epochs=1,
        pretrain_steps=150,
        joint_epochs=2,
        joint_steps=300,
        total_steps=450,
        pretrain_truth="not_loaded_nan_sentinel",
        transition="student_teacher_centers_optimizer_moments_retained",
        joint_teacher_centers="continuously_updated",
        learning_rate_clocks="separate_prefix_and_joint_endpoint_cosines",
        auxiliary_warmup_clock="global_successful_updates",
        sampler_epochs="prefix_local_then_joint_local",
        view_rng="continuous_checkpointed_not_baseline_replay",
    )
    queue["family"] = family.SCHEMA
    row = queue["rows"][0]
    row["name"] = kwargs["arm"] = family.ARMS[0]
    row["plan"]["config_sha256"] = sha256_file(args.config)
    return parent, identity, queue, args, kwargs


def test_medium_and_panel_retain_exact_parent_population(medium_case):
    parent, identity, queue, args, kwargs = medium_case
    before = copy.deepcopy((parent, identity))
    bound = medium.staged_joint_parent_binding(parent, identity, **kwargs)
    assert bound["schedule"] == parent["schedule"]
    assert bound["training_protocol_binding"]["sampling_changed"] is False
    assert (
        observer.panel_parent_binding(parent, identity, queue, args)
        == bound["training_protocol_binding"]
    )
    assert before == (parent, identity)


@pytest.mark.parametrize("change", ["budget", "transition", "canonical", "parent"])
def test_medium_rejects_phase_or_population_drift(medium_case, change):
    parent, identity, _, _, kwargs = medium_case
    if change == "canonical":
        identity["data"]["canonical_sha256"] = "f" * 64
    elif change == "parent":
        parent["data"]["loss_protocol"]["coding_rate"] = {}
    else:
        field = "total_steps" if change == "budget" else "transition"
        identity["data"]["loss_protocol"]["staged_training"][field] = 900
    with pytest.raises(ValueError):
        medium.staged_joint_parent_binding(parent, identity, **kwargs)


def test_capacity_gate_rejects_legacy_or_incomplete_phase_proof():
    checks = dict.fromkeys(
        (
            "phase_order",
            "teacher_centers_continuously_updated",
            "teacher_updated_in_both_phases",
            "centers_updated_in_both_phases",
            "prefix_keeps_ssl_and_aux",
            "prefix_head_has_no_grad_or_moments",
            "optimizer_never_reset",
            "boundary_moments_retained",
            "joint_keeps_ssl_prediction_and_aux",
            "prediction_head_learned_only_in_joint",
            "both_phase_resumes_exact",
        ),
        True,
    )
    gate = dict(
        mode="ssl_then_joint",
        passed=True,
        formal_pretrain_steps=150,
        engineering_pretrain_steps=5,
        engineering_joint_steps=5,
        checks=checks,
    )
    runner.validate_staged_joint_gate(gate)
    for key in checks:
        bad = copy.deepcopy(gate)
        bad["checks"][key] = False
        with pytest.raises(ValueError):
            runner.validate_staged_joint_gate(bad)
    for key, changed in (
        ("mode", "ssl_then_mse"),
        ("formal_pretrain_steps", 300),
        ("engineering_pretrain_steps", 1),
    ):
        bad = copy.deepcopy(gate)
        bad[key] = changed
        with pytest.raises(ValueError):
            runner.validate_staged_joint_gate(bad)
