"""Frozen readout family registration and immutable protocol regressions."""

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
import generate_representation_round9 as r9  # noqa: E402
import generate_representation_round11 as family  # noqa: E402
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


def test_exact_parent_recipe_and_all_dispatch(family_root):
    source, output = family_root
    value = family.verify_manifest(source, output / "manifest.json")
    candidate = family.configs(source)[family.ARMS[0]]
    parent = r9.configs(source)["R9-SSL1J2"]
    for key in ("formal_run_policy", "max_epochs"):
        candidate["training"][key] = copy.deepcopy(parent["training"][key])
    for key in family.CHANGES:
        if key in parent["model"]["parameters"]:
            candidate["model"]["parameters"][key] = parent["model"]["parameters"][key]
        else:
            candidate["model"]["parameters"].pop(key)
    assert candidate == parent
    target = source / value["rows"][0]["config"]
    plan = dict(
        repository_root=str(source),
        config=str(target),
        resolved_config=load_experiment_config(target).model_dump(mode="json"),
    )
    assert runner.expected_epochs(plan) == 2
    assert runner.verify_manifest(source, output / "manifest.json") == value
    assert family.SCHEMA in runner.RESEARCH_FAMILIES
    assert medium.FAMILY_ARMS[family.SCHEMA] == postfit.FAMILY_ARMS[family.SCHEMA] == family.ARMS
    assert value["frozen_readout"]["total_updates"] == 750
    assert value["frozen_readout"]["total_mse_updates"] == 600
    assert value["frozen_readout"]["total_budget_matched"] is False
    assert value["eval_batch"] == value["eval_capacity_reference_batch"] == 128


@pytest.mark.parametrize(
    "field,value",
    [
        ("lambda1", 1.0),
        ("lambda2", 1.0),
        ("training_mode", "joint"),
        ("readout_parent_epoch", 2),
        ("readout_parent_checkpoint_sha256", "a" * 64),
        ("readout_parent_source_commit", "b" * 40),
        ("readout_parent_run_id", "other"),
        ("readout_parent_run_root", "/data/yilangliu/other"),
        ("cr_weight", 0.1),
    ],
)
def test_rehashed_config_drift_rejected(family_root, field, value):
    source, output = family_root
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    target = source / manifest["rows"][0]["config"]
    raw = yaml.safe_load(target.read_text())
    raw["model"]["parameters"][field]["value"] = value
    target.write_text(yaml.safe_dump(raw))
    manifest["rows"][0]["sha256"] = sha256_file(target)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        family.verify_manifest(source, path)


def test_physical_parent_population_and_panel_binding(tmp_path):
    parent, identity, queue, args, kwargs = observer.cr_case.__wrapped__(tmp_path, r9)
    args.config.write_text(yaml.safe_dump(family.configs(ROOT)[family.ARMS[0]]))
    queue["family"] = family.SCHEMA
    queue["rows"][0]["name"] = kwargs["arm"] = family.ARMS[0]
    queue["rows"][0]["plan"]["config_sha256"] = sha256_file(args.config)
    bound = medium.frozen_readout_parent_binding(parent, identity, **kwargs)
    assert bound["schedule"] == parent["schedule"]
    assert (
        observer.panel_parent_binding(parent, identity, queue, args)
        == bound["training_protocol_binding"]
    )
    identity["data"]["canonical_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        medium.frozen_readout_parent_binding(parent, identity, **kwargs)


@pytest.fixture
def generated_r11_queue(family_root, new_queue, monkeypatch):
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
            run_id="fresh-r11-eval",
            run_root=str(Path(plan["run_root"]).parent / "fresh-r11-eval"),
            test_root=str(Path(plan["test_root"]).parent / "fresh-r11-eval-test"),
        )
        return plan

    monkeypatch.setattr(runner, "resolve_plan", resolve)
    monkeypatch.setattr(runner, "__file__", str(source / "scripts/v2/run_br400_group.py"))
    return runner.prepare(
        source,
        family_output / "manifest.json",
        runtime,
        old_output.parent / "r11-eval",
        old["shared_gpu_policy"],
    )


def r11_eval_args(queue, gpu):
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
def test_exact_r11_eval_leaf_accepts_both_real_generated_gpu_bindings(generated_r11_queue, gpu):
    from eval_batch_capacity import PARITY_ATOL, PARITY_RTOL, validate_eval_binding

    queue = generated_r11_queue
    assert len(queue["eval_bindings"]) == 2
    assert {binding["gpu"] for binding in queue["eval_bindings"]} == {"0", "1"}
    assert (
        validate_eval_binding(queue, r11_eval_args(queue, gpu), queue["source_commit"])
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
def test_r11_eval_rejects_scope_or_identity_drift(generated_r11_queue, change):
    from eval_batch_capacity import validate_eval_binding

    queue = copy.deepcopy(generated_r11_queue)
    args = r11_eval_args(queue, "0")
    plan = queue["rows"][0]["plan"]
    if change == "mode":
        queue["mode"] = r9.SCHEMA
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
        plan["resolved_config"]["training"]["max_epochs"]["value"] = 3
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
        validate_eval_binding(queue, args, generated_r11_queue["source_commit"])


def test_head_capacity_gate_requires_every_frozen_update_and_restore_proof():
    proof = dict.fromkeys(
        (
            "passed",
            "exact_restore",
            "head_changed",
            "frozen_equal",
            "frozen_gradients_none",
            "all_four_head_tensors_updated",
            "prediction_changed",
            "native_softplus_preserved",
        ),
        True,
    )
    proof.update(steps=10, teacher_updates=0, center_updates=0, trainable_parameters=131585)
    runner.validate_frozen_readout_gate(proof)
    for key in proof:
        bad = copy.deepcopy(proof)
        bad[key] = False if isinstance(bad[key], bool) else bad[key] + 1
        with pytest.raises(ValueError):
            runner.validate_frozen_readout_gate(bad)


def test_frozen_graph_diagnostics_and_report_export(tmp_path, monkeypatch):
    import csv
    from types import SimpleNamespace

    import analyze_completed_ablation as analysis
    import torch
    from cap40_posthoc_report import gradient_caption

    from gradpert.training.v2.process_diagnostics import evaluate_graph_gradient_diagnostics

    # The native function must terminate before any forward/autograd on frozen graph.
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.graph = torch.nn.Linear(2, 2)
            self.graph.requires_grad_(False)

    batch = SimpleNamespace(control=torch.zeros(2, 2))
    monkeypatch.setattr("gradpert.training.v2.process_diagnostics.slice_cells", lambda b, *args: b)
    native = evaluate_graph_gradient_diagnostics(
        SimpleNamespace(student=Student()), batch, bf16=False
    )
    assert native == dict(performed=False, reason="no_trainable_graph_parameters")
    point = dict(
        epoch=2,
        training_stage="frozen_readout",
        graph_gradients=native,
        features={},
        distributions={},
        pairs={},
        outputs={},
    )
    assert analysis.graph_gradient_groups(point) == {}
    assert (
        analysis.graph_gradient_rows([point])[0]["status"]
        == "unavailable:no_trainable_graph_parameters"
    )
    assert "主干冻结" in gradient_caption([native])
    assert "NA不替换为0" in gradient_caption([native])
    active = dict(
        performed=True,
        groups={
            "graph": dict(
                weighted_gradient_l2={"prediction": 1.0}, cosine={"prediction_ssl1": None}
            )
        },
    )
    assert "共享图参数上比较" in gradient_caption([active])
    # Exercise the real standalone aggregate exporter with native missing-groups payload.
    monkeypatch.setattr(analysis, "plots", lambda *args: None)
    monkeypatch.setattr(
        analysis, "report", lambda d: analysis.graph_gradient_status(d["snapshots"][-1])
    )
    output = tmp_path / "report"
    data = dict(
        snapshots=[point], history=[], metrics=[], run_id="child", analysis_identity={}, inputs={}
    )
    analysis.export(data, output)
    rows = list(csv.DictReader((output / "gradients.csv").open()))
    assert rows[0]["status"] == "unavailable:no_trainable_graph_parameters"
    assert (output / "report.md").read_text() == "unavailable:no_trainable_graph_parameters"


def test_exact_original_parent_receipt_hashes_required():
    parent = dict(
        parent_run_id=family.PARENT_RUN,
        parent_checkpoint_sha256=family.PARENT_CHECKPOINT_SHA256,
        parent_training_source=dict(commit=family.PARENT_TRAINING_SOURCE),
        parent_epoch=3,
        parent_scientific_training_complete=True,
        parent_full_test_complete=False,
        optimizer_restored=False,
        child_initial_steps=0,
        parent_evidence_sha256=family.metadata()["frozen_readout"]["parent_evidence_sha256"],
    )
    runner.validate_frozen_readout_parent(parent)
    for key in parent["parent_evidence_sha256"]:
        bad = copy.deepcopy(parent)
        bad["parent_evidence_sha256"][key] = "f" * 64
        with pytest.raises(ValueError):
            runner.validate_frozen_readout_parent(bad)


def test_actual_standalone_report_and_plots_accept_inactive_ssl(tmp_path):
    import analyze_completed_ablation as analysis

    point = dict(
        epoch=2,
        training_stage="frozen_readout",
        graph_gradients=dict(performed=False, reason="no_trainable_graph_parameters"),
        features={
            name: dict(samples=4, covariance_effective_rank=2.0, relative_centered_variance=0.1)
            for name in ("student.graph.output", "student.response.output.genes")
        },
        distributions={},
        pairs={},
        outputs=dict.fromkeys(
            (
                "prediction_mse",
                "control_copy_mse",
                "mse_gain_over_copy",
                "delta_rms",
                "paired_effect_rms_ratio",
                "negative_prediction_fraction",
                "top_one_percent_loss_fraction",
            ),
            0.1,
        ),
    )
    data = dict(
        name="R11 fixture",
        run_id="child",
        epochs=2,
        training_source=dict(commit="a" * 40),
        evaluation_source=dict(commit="a" * 40),
        analysis_identity=dict(commit="a" * 40),
        checkpoint_sha256="b" * 64,
        snapshots=[point],
        history=[],
        metrics=[],
        inputs={},
        test_prediction_mse=0.1,
        limitations=[],
    )
    output = tmp_path / "full-report"
    analysis.export(data, output)
    report = (output / "report.md").read_text()
    assert "unavailable:no_trainable_graph_parameters" in report
    assert "SSL/A/U/CR/EMA停用" in report
    assert (output / "04_outputs_gradients.png").is_file()


def test_original_parent_file_bytes_are_not_canonical_resolved_json():
    # Independent literals were verified from the original server files and
    # parent manifest; do not derive both sides from the generator constants.
    metadata = family.metadata()["frozen_readout"]
    assert metadata["parent_evidence_sha256"]["resolved_config.json"] == (
        "0ff2c230123622e1ea404fd5388f91af2360f01207fe96c5018a064c53635f85"
    )
    assert metadata["parent_resolved_config_canonical_sha256"] == (
        "5d7534762fcb0c8682e0a9e15d432dc1c101cfffb5b04b618634f83762ad7cf2"
    )
    assert (
        metadata["parent_evidence_sha256"]["resolved_config.json"]
        != metadata["parent_resolved_config_canonical_sha256"]
    )


def test_pretty_file_checksum_and_canonical_json_identity_are_distinct(tmp_path):
    from gradpert.hashing import sha256_json

    value = {"b": 2, "a": [1, 3]}
    compact = tmp_path / "compact.json"
    pretty = tmp_path / "pretty.json"
    compact.write_text(json.dumps(value, separators=(",", ":")))
    pretty.write_text(json.dumps(value, indent=2) + "\n")
    assert sha256_file(compact) != sha256_file(pretty)
    assert sha256_json(json.loads(compact.read_text())) == sha256_json(
        json.loads(pretty.read_text())
    )
