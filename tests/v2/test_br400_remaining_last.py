"""Strict remaining-arm scope, actual native gates and last-only completion."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_br400_group import (
    CHECKPOINT_MANIFEST,
    ROOT,
    execution_mocks,
    generator,
    runner,
    training_receipt,
)
from test_br400_group import prepared as prepared

from gradpert.data._io import atomic_json, read_json
from gradpert.execution.v2_training_stage import training_stage_evidence
from gradpert.hashing import sha256_file, sha256_json

MANIFEST = ROOT / "configs/v2/br400_remaining_last_20261012/manifest.json"


@pytest.fixture
def remaining(prepared):
    old, oldout = prepared
    policy = copy.deepcopy(old["shared_gpu_policy"])
    policy.update(
        additional_command_basenames=["run_totalvi_source_validation_development.py"],
        authorization_reference="2026-10-12-user-shared-memory90-coexistence",
        max_own_memory_fraction=0.90,
    )
    output = oldout.parent / "remaining"
    return runner.prepare(ROOT, MANIFEST, Path(old["runtime"]), output, policy), output


def test_config_only_disabled_validation_monitor_changes_and_no_br():
    old = generator.verify_manifest(ROOT, CHECKPOINT_MANIFEST)
    matrix = generator.verify_manifest(ROOT, MANIFEST)
    assert matrix["schema"] == generator.REMAINING_SCHEMA
    assert matrix["priority"] == matrix["preflight_priority"] == ["F1", "N1", "GM1"]
    assert matrix["test_roles"] == ["last"] and matrix["checkpoint_genept"] is True
    original = {
        row["name"]: yaml.safe_load((ROOT / row["config"]).read_text()) for row in old["rows"]
    }
    for row in matrix["rows"]:
        config = yaml.safe_load((ROOT / row["config"]).read_text())
        assert config["model"]["parameters"]["validation_mode"] == {
            "value": "disabled",
            "source": "user_locked",
            "reference": generator.LAST_REFERENCE,
        }
        assert config["training"]["monitor"] == config["training"]["monitor_mode"] == "none"
        config["model"]["parameters"]["validation_mode"] = original[row["name"]]["model"][
            "parameters"
        ]["validation_mode"]
        for field in ("monitor", "monitor_mode"):
            config["training"][field] = original[row["name"]]["training"][field]
        assert config == original[row["name"]]


@pytest.mark.parametrize("change", ["br", "mode", "best", "no_checkpoint", "memory85"])
def test_malformed_family_scope_rejected(remaining, change):
    queue, _ = remaining
    queue = copy.deepcopy(queue)
    if change == "br":
        queue["rows"].append({**queue["rows"][0], "name": "BR"})
    elif change == "mode":
        queue["mode"] = generator.SCHEMA
    elif change == "memory85":
        queue["shared_gpu_policy"].update(
            max_own_memory_fraction=0.85,
            authorization_reference="2026-10-12-user-totalvi-coexistence",
        )
        queue["policy_sha256"] = sha256_json(queue["shared_gpu_policy"])
    else:
        plan = queue["rows"][0]["plan"]
        if change == "best":
            plan["resolved_config"]["training"]["monitor"] = "val/joint_loss"
        else:
            plan["resolved_config"]["model"]["parameters"]["checkpoint_genept_mapping"]["value"] = (
                False
            )
    with pytest.raises(ValueError):
        runner.verify(queue)


def test_all_three_gates_then_sequential_fit_last_and_no_joint_or_br(remaining, monkeypatch):
    queue, output, calls = execution_mocks(remaining, monkeypatch)
    runner.execute(queue, output)
    assert [(name, stage) for name, stage, _, _ in calls[:9]] == [
        (name, stage)
        for name in generator.REMAINING_PRIORITY
        for stage in ("preflight", "eval-capacity-g0", "eval-capacity-g1")
    ]
    assert [(name, stage) for name, stage, _, _ in calls[9:]] == [
        (name, stage) for name in generator.REMAINING_PRIORITY for stage in ("fit", "last")
    ]
    assert all(
        "--joint-validation-check" not in cmd for _, stage, cmd, _ in calls if stage == "preflight"
    )
    complete = read_json(output / "COMPLETE.json")
    assert set(complete["completed"]) == {"F1", "N1", "GM1"}
    assert all(entry["roles"] == ["last"] for entry in complete["completed"].values())
    ready = read_json(output / "PREFLIGHT_COMPLETE.json")
    assert ready["formal_order"] == ["F1", "N1", "GM1"] and len(ready["preflights"]) == 3


def test_last_only_joint_gate_rejected(remaining):
    queue, output = remaining
    row = queue["rows"][0]
    receipt = training_receipt(queue, row)
    receipt["joint_validation_check"] = {"status": "passed"}
    path = output / "F1-preflight/receipt.json"
    path.parent.mkdir()
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="must not perform joint"):
        runner.validate_training_gate(queue, row, runner.gate_entry(path))


def test_last_eval_failure_stops_before_next_fit(remaining, monkeypatch):
    queue, output, calls = execution_mocks(remaining, monkeypatch, fail=("F1", "last"))
    with pytest.raises(RuntimeError):
        runner.execute(queue, output)
    assert calls[-1][:2] == ("F1", "last")
    assert ("N1", "fit") not in [item[:2] for item in calls]
    assert not (output / "COMPLETE.json").exists()


def native_complete(plan):
    root = Path(plan["run_root"])
    fit = root / "fit"
    fit.mkdir(parents=True)
    atomic_json(root / "launch.json", plan)
    atomic_json(root / "resolved_config.json", plan["resolved_config"])
    source = {
        "commit": plan["source_commit"],
        "published_commit": plan["source_commit"],
        "dirty": False,
    }
    identity = {
        "source": source,
        "run_id": plan["run_id"],
        "config_sha256": plan["config_sha256"],
        "resolved_config_sha256": sha256_json(plan["resolved_config"]),
    }
    atomic_json(root / "run_manifest.json", identity)
    checkpoint = fit / "epoch-0004.pt"
    checkpoint.write_bytes(b"last native fixture")
    selected = {"file": checkpoint.name, "sha256": sha256_file(checkpoint), "epoch": 4}
    atomic_json(
        fit / "epoch_state.json",
        {
            "identity": identity,
            "epoch": 4,
            "budget": [4, 125],
            "selection_metric": "final_epoch",
            "best": None,
            "last": selected,
        },
    )
    atomic_json(
        fit / "history.json",
        [
            {"epoch": epoch, "validation": {"validation_mode": "disabled", "performed": False}}
            for epoch in range(1, 5)
        ],
    )
    training = training_stage_evidence(root, identity, 4)
    atomic_json(root / "TRAIN_COMPLETE.json", training)
    metrics = [
        {
            "metric_id": metric,
            "macro_mean": 0.5,
            "finite_condition_count": 1,
            "total_condition_count": 1,
        }
        for metric in runner.METRICS
    ]
    result = {
        "split": "test",
        "prediction_loss": 0.5,
        "metrics": metrics,
        "metric_gene_groups": {
            group: {"metrics": copy.deepcopy(metrics)}
            for group in ("seen_expression", "unseen_expression")
        },
    }
    independent = root / "postfit-last/COMPLETE.json"
    independent.parent.mkdir()
    atomic_json(
        independent,
        {
            "plan": {
                "training_identity": identity,
                "checkpoint_sha256": selected["sha256"],
                "checkpoint_role": "last",
                "checkpoint_epoch": 4,
                "training_run_root": str(root),
                "split": "test",
                "output_root": str(independent.parent),
                "evaluation_source": source,
            },
            "result": result,
            "evaluation_environment": {},
            "zero_pkl": True,
        },
    )
    atomic_json(
        fit / "last-test.json",
        {
            "identity": {
                "training": identity,
                "checkpoint": selected,
                "role": "last",
                "evaluation": {"source": source, "environment": {}},
            },
            "result": result,
            "independent_evaluation_receipt": str(independent),
            "independent_evaluation_sha256": sha256_file(independent),
        },
    )
    runner.finalize_deferred_run(plan, training)


@pytest.mark.parametrize("bad", [None, "validation", "best", "metric", "pkl"])
def test_actual_native_last_only_completion_and_failure_boundaries(remaining, bad):
    queue, _ = remaining
    plan = queue["rows"][0]["plan"]
    native_complete(plan)
    root = Path(plan["run_root"])
    if bad == "validation":
        history = read_json(root / "fit/history.json")
        history[0]["validation"] = {"split": "val", "joint_loss": 1}
        atomic_json(root / "fit/history.json", history)
        training = training_stage_evidence(root, read_json(root / "run_manifest.json"), 4)
        atomic_json(root / "TRAIN_COMPLETE.json", training)
    elif bad == "best":
        atomic_json(root / "fit/best-test.json", {})
    elif bad == "metric":
        test = read_json(root / "fit/last-test.json")
        test["result"]["metrics"].pop()
        atomic_json(root / "fit/last-test.json", test)
    elif bad == "pkl":
        (root / "unexpected.pkl").write_text("x")
    if bad:
        with pytest.raises(ValueError):
            runner.validate_complete(plan)
    else:
        complete = runner.validate_complete(plan)
        assert [row["role"] for row in complete["roles"]] == ["last"]
        assert set(complete["test_sha256"]) == {"last"}


def test_native_inference_audit_without_joint_flag():
    import capacity_probe

    draw = SimpleNamespace(ordered_row_ids=tuple(range(300)))
    evaluation = SimpleNamespace(manifest=SimpleNamespace(expression_gene_order_sha256="a" * 64))
    identity = capacity_probe.inference_identity(evaluation, draw)
    assert identity == {
        "ordered_control_row_ids_sha256": sha256_json(list(range(300))),
        "inference_expression_gene_order_sha256": "a" * 64,
    }


def test_remaining_rejects_lower_evaluation_batch():
    with pytest.raises(ValueError, match="batch300"):
        generator.configurations(
            ROOT, eval_batch=128, checkpoint_genept=True, remaining_last_only=True
        )


@pytest.mark.parametrize("change", [None, "wrong_gpu", "br_scope", "wrong_mode"])
def test_actual_remaining_eval_leaf_binding_is_strict(remaining, change):
    import eval_batch_capacity

    queue, output = remaining
    row, binding = queue["rows"][0], queue["eval_bindings"][0]
    args = SimpleNamespace(
        config=Path(row["plan"]["config"]),
        data_root=Path(row["plan"]["data_root"]),
        publication=Path(row["plan"]["publication"]),
        publication_sha256=row["plan"]["publication_sha256"],
        shared_queue=output / "queue.json",
        output=Path(binding["output"]),
        row_name=row["name"],
        gpu=binding["gpu"],
        cell_batch=300,
        reference_batch=128,
    )
    if change == "wrong_gpu":
        args.gpu = "1"
    elif change == "br_scope":
        queue["rows"].append({**row, "name": "BR"})
    elif change == "wrong_mode":
        queue["mode"] = generator.SCHEMA
    if change:
        with pytest.raises(ValueError):
            eval_batch_capacity.validate_eval_binding(queue, args, queue["source_commit"])
    else:
        assert eval_batch_capacity.validate_eval_binding(queue, args, queue["source_commit"]) == row
