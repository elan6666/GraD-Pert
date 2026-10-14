"""Bounded read-only validation and sealed fresh evaluation-capacity contracts."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from test_components import fixture

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import capacity_probe as probe  # noqa: E402
import eval_batch_capacity as evaluation  # noqa: E402
import run_br400_group as controller  # noqa: E402


def test_one_real_joint_validation_batch_preserves_nonzero_lr_updated_state(monkeypatch):
    model, batch = fixture()
    objective = JointObjective(model, lambda1=1, lambda2=1, loss_reduction="row_mean").train()
    optimizer = V2Optimizer(model, lr=1e-4, weight_decay=0)
    optimizer_step(objective, optimizer, batch, microbatch=2, lr=1e-4, momentum=0.9, bf16=False)
    consumed = []

    def batches(*, batch_size):
        assert batch_size == 2
        for i in range(3):
            consumed.append(i)
            yield batch, f"fixed-batch-{i}"

    runtime = SimpleNamespace(
        objective=objective,
        optimizer=optimizer,
        generator=np.random.default_rng(7),
        device=torch.device("cpu"),
        validation_batches=batches,
    )

    def forbidden_collective(*_args, **_kwargs):
        raise AssertionError("primary-only validation must not enter a rank collective")

    monkeypatch.setattr(torch.distributed, "all_reduce", forbidden_collective)
    monkeypatch.setattr(torch.distributed, "all_gather", forbidden_collective)
    before = probe.validation_state(runtime)
    result = probe.joint_validation_check(runtime, batch_size=2, bf16=False)
    assert consumed == [0]
    assert result["batch_count"] == 1 and result["cell_count"] == 2
    assert result["scope"] == "one_complete_native_validation_batch_engineering_only"
    assert result["full_validation"] is False
    assert result["state_unchanged"] == dict.fromkeys(before, True)
    assert result["before_sha256"] == result["after_sha256"] == before
    assert optimizer.steps == 1 and result["components"]["ssl2_dino"] > 0


def test_joint_check_rejects_a_validation_side_effect(monkeypatch):
    from gradpert.training.v2 import joint_validation

    model, batch = fixture()
    objective = JointObjective(model)
    runtime = SimpleNamespace(
        objective=objective,
        optimizer=V2Optimizer(model, lr=1e-4, weight_decay=0),
        generator=np.random.default_rng(4),
        device=torch.device("cpu"),
        validation_batches=lambda **_: (row for row in [(batch, "fixed")]),
    )
    original = joint_validation.evaluate_joint_loss

    def mutating(*args, **kwargs):
        result = original(*args, **kwargs)
        objective.ssl1_cls_center.add_(1)
        return result

    monkeypatch.setattr(joint_validation, "evaluate_joint_loss", mutating)
    with pytest.raises(ValueError, match=r"mutated protected state.*centers"):
        probe.joint_validation_check(runtime, batch_size=2, bf16=False)


@pytest.mark.parametrize("change", [None, "scope", "mode", "steps", "gpu", "config"])
def test_ffn_five_eval_leaf_preserves_exact_scope_and_binding(tmp_path, change):
    from generate_ffn_five import PRIORITY, SCHEMA

    queue, row, binding, _, _ = _queue(tmp_path)
    queue.update(family=SCHEMA, mode=SCHEMA, steps=10)
    queue["rows"] = [{**row, "name": name} for name in PRIORITY]
    row = queue["rows"][0]
    binding.update(row_name=row["name"], output=str(tmp_path / "queue/DP-eval-b300-g0"))
    args = SimpleNamespace(
        config=Path(row["plan"]["config"]),
        data_root=Path(row["plan"]["data_root"]),
        publication=Path(row["plan"]["publication"]),
        publication_sha256=row["plan"]["publication_sha256"],
        shared_queue=tmp_path / "queue/queue.json",
        output=Path(binding["output"]),
        row_name=row["name"],
        gpu="0",
        cell_batch=300,
        reference_batch=128,
    )
    if change == "scope":
        queue["rows"].append({**row, "name": "BR"})
    elif change == "mode":
        queue["mode"] = "br400-best-last-1"
    elif change == "steps":
        queue["steps"] = 128
    elif change == "gpu":
        args.gpu = "1"
    elif change == "config":
        row["plan"]["config_sha256"] = "0" * 64
    if change:
        with pytest.raises(ValueError):
            evaluation.validate_eval_binding(queue, args, queue["source_commit"])
    else:
        assert evaluation.validate_eval_binding(queue, args, queue["source_commit"]) == row


def _queue(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config_path = source / "gradpert_v2/nadig_jurkat.yaml"
    config_path.parent.mkdir()
    value = yaml.safe_load(
        (ROOT / "configs/v2/ffn_four_20261011/BR/gradpert_v2/nadig_jurkat.yaml").read_text()
    )
    value["training"]["eval_batch_size"]["value"] = 300
    config_path.write_text(yaml.safe_dump(value))
    output = tmp_path / "queue"
    output.mkdir()
    binding = {
        "row_name": "BR",
        "gpu": "0",
        "cell_batch": 300,
        "reference_batch": 128,
        "output": str(output / "BR-eval-b300-g0"),
    }
    row = {
        "name": "BR",
        "plan": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "data_root": "/data/yilangliu/canonical",
            "publication": str(source / "publication.json"),
            "publication_sha256": "b" * 64,
            "source_commit": "a" * 40,
            "seed": 1,
            "gpu": "0,1",
        },
    }
    policy = {
        "schema": "shared-gpu-admission-3",
        "devices": [{"gpu_index": i, "gpu_uuid": f"GPU-{i}"} for i in range(2)],
        "foreign_uid": 1003,
        "command_basename": "run_scbutterfly_archived_predictions.py",
        "max_foreign_processes_per_gpu": 1,
        "max_foreign_memory_mib": 1024,
        "min_free_memory_mib": 2048,
        "max_own_memory_fraction": 0.85,
        "authorization_reference": "2026-10-12-user-scbutterfly-coexistence",
    }
    queue = {
        "schema": "unified-ddp-queue-1",
        "family": "br400-best-last-1",
        "mode": "br400-best-last-1",
        "source": str(source),
        "source_commit": "a" * 40,
        "rows": [row],
        "eval_bindings": [binding],
        "shared_gpu_policy": policy,
        "output": str(output),
    }
    (output / "queue.json").write_text(json.dumps(queue))
    config = load_experiment_config(config_path)
    architecture, options = V2Options.parse_parameters(config.model.parameters)
    controls = [f"control-{i}" for i in range(300)]
    receipt = {
        "kind": evaluation.KIND,
        "status": "passed",
        "config_sha256": row["plan"]["config_sha256"],
        "data_root": row["plan"]["data_root"],
        "gpu": "0",
        "cell_batch": 300,
        "reference_batch": 128,
        "shared_queue_sha256": sha256_file(output / "queue.json"),
        "shared_gpu_policy": policy,
        "inference_shape": [300, 5000],
        "split": "val",
        "no_truth_access": True,
        "no_scientific_scores": True,
        "model_state_unchanged": True,
        "rng_unchanged": True,
        "model_state_before_sha256": "c" * 64,
        "model_state_after_sha256": "c" * 64,
        "rng_before_sha256": "d" * 64,
        "rng_after_sha256": "d" * 64,
        "ordered_control_row_ids": controls,
        "ordered_control_row_ids_sha256": sha256_json(controls),
        "inference_control_manifest_sha256": "e" * 64,
        "prediction_sha256": "f" * 64,
        "reference_prediction_sha256": "f" * 64,
        "inference_seconds": 1,
        "reference_seconds": 2,
        "peak_allocated_bytes": 3000,
        "peak_reserved_bytes": 4000,
        "architecture": architecture.payload(),
        "evaluation_query_count": options.eval_query_count,
        "only_student_on_cuda": True,
        "runtime_purpose": "evaluation",
        "source": {
            "commit": "a" * 40,
            "published_commit": "a" * 40,
            "dirty": False,
            "formal_eligible": True,
            "publication_receipt_sha256": "b" * 64,
        },
        "comparison": {
            "allclose": True,
            "atol": 2e-6,
            "rtol": 2e-6,
            "max_abs_error": 0,
            "mean_abs_error": 0,
            "max_scaled_error": 0,
        },
        "data": {
            **{
                name: "e" * 64
                for name in (
                    "canonical_sha256",
                    "split_sha256",
                    "expression_gene_order_sha256",
                    "graph_gene_order_sha256",
                )
            },
            "graph_manifest_sha256": options.graph_manifest_sha256,
            "genept_sha256": options.genept_sha256,
        },
    }
    leaf = Path(binding["output"])
    leaf.mkdir()
    path = leaf / "receipt.json"
    path.write_text(json.dumps(receipt))
    native_path = output / "BR-preflight/receipt.json"
    native_path.parent.mkdir()
    native = {
        "kind": "preflight_only",
        "status": "passed",
        "steps_requested": 10,
        "steps_completed": 10,
        "resume_checkpoint_sha256": "c" * 64,
        "inference_exercised": True,
        "inference_shape": [300, 5000],
        "source": receipt["source"],
        "config_sha256": receipt["config_sha256"],
        "data_root": receipt["data_root"],
        "data": {**receipt["data"], "run_seed": 1},
        "world_size": 2,
        "peak_allocated_bytes": 1000,
        "last_terms": {"gradient_norm": 1},
        "inference_control_manifest_sha256": receipt["inference_control_manifest_sha256"],
        "ordered_control_row_ids_sha256": receipt["ordered_control_row_ids_sha256"],
    }
    native_path.write_text(json.dumps(native))
    (output / "state.json").write_text(
        json.dumps(
            {
                "queue_sha256": sha256_file(output / "queue.json"),
                "source_commit": queue["source_commit"],
                "preflights": {
                    "BR": {"receipt": str(native_path), "sha256": sha256_file(native_path)}
                },
            }
        )
    )
    return queue, row, binding, receipt, path


def test_real_controller_calls_leaf_receipt_validator(tmp_path):
    queue, row, binding, receipt, path = _queue(tmp_path)
    result = controller.validate_eval_gate(
        queue, row, binding, {"receipt": str(path), "sha256": sha256_file(path)}
    )
    assert result == receipt
    command = controller.eval_capacity_command(queue, row, binding)
    assert command[command.index("--cell-batch") + 1] == "300"
    assert command[command.index("--row-name") + 1] == "BR"


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("gpu", "1"),
        ("cell_batch", 192),
        ("reference_batch", 64),
        ("no_truth_access", False),
        ("no_scientific_scores", False),
        ("inference_shape", [300, 1000]),
        ("model_state_unchanged", "true"),
        ("rng_after_sha256", "x" * 64),
        ("only_student_on_cuda", False),
        ("evaluation_query_count", 999),
    ],
)
def test_incomplete_or_rebound_eval_receipts_are_rejected(tmp_path, field, value):
    queue, row, binding, receipt, path = _queue(tmp_path)
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        controller.validate_eval_gate(
            queue, row, binding, {"receipt": str(path), "sha256": sha256_file(path)}
        )


def test_receipt_rejects_relaxed_parity_and_changed_graph_semantics(tmp_path):
    queue, row, binding, receipt, path = _queue(tmp_path)
    for altered in (
        {**receipt, "comparison": {**receipt["comparison"], "atol": 1e-3}},
        {**receipt, "architecture": {**receipt["architecture"], "graph_source_key_gate": False}},
    ):
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            evaluation.validate_receipt(
                queue, row, binding, {"receipt": str(path), "sha256": sha256_file(path)}
            )


@pytest.mark.parametrize("name", evaluation.DATA_HASH_FIELDS)
def test_fresh_eval_rejects_even_resealed_native_data_drift(tmp_path, name):
    queue, row, binding, _receipt, path = _queue(tmp_path)
    native_path = Path(queue["output"]) / "BR-preflight/receipt.json"
    native = json.loads(native_path.read_text())
    native["data"][name] = "9" * 64
    native_path.write_text(json.dumps(native))
    state_path = Path(queue["output"]) / "state.json"
    state = json.loads(state_path.read_text())
    state["preflights"]["BR"]["sha256"] = sha256_file(native_path)
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="data/controls differ"):
        controller.validate_eval_gate(
            queue, row, binding, {"receipt": str(path), "sha256": sha256_file(path)}
        )


def test_fixed_parity_checks_cover_finite_precision_and_all_genes():
    reference = np.ones((300, 5000), dtype=np.float32)
    prediction = reference.copy()
    prediction[299, 4999] += 1e-6
    result = evaluation.compare_predictions(reference, prediction)
    assert result["allclose"] and result["max_abs_error"] > 0
    prediction[299, 4999] += 1e-3
    with pytest.raises(ValueError, match="fixed inference parity"):
        evaluation.compare_predictions(reference, prediction)
    prediction[299, 4999] = np.nan
    with pytest.raises(FloatingPointError, match="finite"):
        evaluation.compare_predictions(reference, prediction)
    with pytest.raises(ValueError, match="complete frozen"):
        evaluation.compare_predictions(reference[:, :1000], reference[:, :1000])


@pytest.mark.parametrize(
    "script,flags,message",
    [
        ("capacity_probe.py", ["--integration-only", "--joint-validation-check"], "ten-update"),
        ("capacity_probe.py", ["--benchmark-only", "--joint-validation-check"], "ten-update"),
        ("eval_batch_capacity.py", ["--cell-batch", "301"], "cell batch must be"),
    ],
)
def test_invalid_probe_flags_fail_before_cuda_import(script, flags, message):
    arguments = [
        sys.executable,
        str(ROOT / "scripts/v2" / script),
        "--config",
        "/not-read.yaml",
        "--data-root",
        "/data/yilangliu/not-read",
        "--output",
        "/data/yilangliu/not-created",
        "--gpu",
        "0",
        "--publication",
        "/data/yilangliu/not-read.json",
        "--publication-sha256",
        "unused",
    ]
    if script == "eval_batch_capacity.py":
        arguments.extend(
            [
                "--shared-queue",
                "/data/yilangliu/not-read-queue.json",
                "--resource-owner-pid",
                "1",
                "--row-name",
                "BR",
            ]
        )
    result = subprocess.run(
        arguments + flags,
        text=True,
        capture_output=True,
        timeout=10,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "WORLD_SIZE": "1"},
    )
    assert result.returncode == 2 and message in result.stderr


def test_state_fingerprint_handles_optimizer_scalar_bfloat_and_numpy_rng():
    state = {"tensor": torch.tensor(3, dtype=torch.bfloat16), "rng": np.random.get_state()}
    assert probe.state_fingerprint(state) == probe.state_fingerprint(copy.deepcopy(state))
    state["tensor"] += 1
    assert probe.state_fingerprint(state) != probe.state_fingerprint(
        {"tensor": torch.tensor(3, dtype=torch.bfloat16), "rng": np.random.get_state()}
    )
