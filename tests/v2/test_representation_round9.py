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
from test_br400_group import prepared as prepared  # noqa: E402
from test_fgpo_group import new_queue as new_queue  # noqa: E402


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


@pytest.fixture
def generated_r9_queue(family_root, new_queue, monkeypatch):
    """Use the real generator and queue builder; replace only external source resolution."""
    from gradpert.data._io import atomic_json, read_json

    source, family_output = family_root
    old, old_output = new_queue
    runtime = Path(old["runtime"])
    atomic_json(runtime, {**read_json(runtime), "repository_root": str(source)})

    def resolve(args):
        plan = copy.deepcopy(old["rows"][0]["plan"])
        plan.update(
            config=str(args.config),
            config_sha256=sha256_file(args.config),
            resolved_config=load_experiment_config(args.config).model_dump(mode="json"),
            runtime_sha256=sha256_file(runtime),
            repository_root=str(source),
            run_id="fresh-r9-eval",
            run_root=str(Path(plan["run_root"]).parent / "fresh-r9-eval"),
            test_root=str(Path(plan["test_root"]).parent / "fresh-r9-eval-test"),
        )
        return plan

    monkeypatch.setattr(runner, "resolve_plan", resolve)
    monkeypatch.setattr(runner, "__file__", str(source / "scripts/v2/run_br400_group.py"))
    return runner.prepare(
        source,
        family_output / "manifest.json",
        runtime,
        old_output.parent / "r9-eval",
        old["shared_gpu_policy"],
    )


def r9_eval_args(queue, gpu):
    from types import SimpleNamespace

    row = queue["rows"][0]
    plan = row["plan"]
    binding = next(binding for binding in queue["eval_bindings"] if binding["gpu"] == gpu)
    return SimpleNamespace(
        gpu=gpu,
        cell_batch=128,
        reference_batch=128,
        config=Path(plan["config"]),
        publication=Path(plan["publication"]),
        publication_sha256=plan["publication_sha256"],
        data_root=Path(plan["data_root"]),
        output=Path(binding["output"]),
        shared_queue=Path(queue["output"]) / "queue.json",
        row_name=row["name"],
    )


@pytest.mark.parametrize("gpu", ["0", "1"])
def test_exact_r9_eval_leaf_accepts_both_real_generated_gpu_bindings(generated_r9_queue, gpu):
    from eval_batch_capacity import PARITY_ATOL, PARITY_RTOL, validate_eval_binding

    queue = generated_r9_queue
    assert len(queue["eval_bindings"]) == 2
    assert {binding["gpu"] for binding in queue["eval_bindings"]} == {"0", "1"}
    assert (
        validate_eval_binding(queue, r9_eval_args(queue, gpu), queue["source_commit"])
        == queue["rows"][0]
    )
    assert PARITY_ATOL == PARITY_RTOL == 2e-6


@pytest.mark.parametrize(
    "change",
    [
        "mode",
        "extra_arm",
        "steps",
        "selected_subset",
        "source",
        "plan_source",
        "config_sha",
        "missing_resolved",
        "resolved",
        "cell_batch",
        "reference_batch",
        "output",
        "data_root",
        "publication",
    ],
)
def test_r9_eval_rejects_scope_or_identity_drift(generated_r9_queue, change):
    from eval_batch_capacity import validate_eval_binding

    queue = copy.deepcopy(generated_r9_queue)
    args = r9_eval_args(queue, "0")
    plan = queue["rows"][0]["plan"]
    if change == "mode":
        queue["mode"] = r6.SCHEMA
    elif change == "extra_arm":
        queue["rows"].append({**queue["rows"][0], "name": "R6-EMA99"})
    elif change == "steps":
        queue["steps"] = 9
    elif change == "selected_subset":
        queue["only_arms"] = list(family.ARMS)
    elif change == "source":
        queue["source_commit"] = "f" * 40
    elif change == "plan_source":
        plan["source_commit"] = "f" * 40
    elif change == "config_sha":
        plan["config_sha256"] = "f" * 64
    elif change == "missing_resolved":
        plan.pop("resolved_config")
    elif change == "resolved":
        plan["resolved_config"]["training"]["max_epochs"]["value"] = 2
    elif change == "cell_batch":
        args.cell_batch = 129
    elif change == "reference_batch":
        args.reference_batch = 129
    elif change == "output":
        args.output = args.output.parent / "unbound-output"
    elif change == "data_root":
        args.data_root = args.data_root / "other-data"
    elif change == "publication":
        args.publication_sha256 = "f" * 64
    with pytest.raises(ValueError):
        validate_eval_binding(queue, args, generated_r9_queue["source_commit"])
