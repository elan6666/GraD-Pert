"""Distinct training/validation-only gate; historical external100 is unchanged."""

import json
import math
from pathlib import Path

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file


def _identity(config, config_sha256, training_data, source_commit):
    return {
        "schema": "external-r50-smoke-v1",
        "status": "trained_validation_only",
        "scientific_completion": False,
        "source_dirty": False,
        "source_commit": source_commit,
        "upstream_commit": config.source_code.commit,
        "model_id": config.model_id,
        "dataset_id": config.dataset_id,
        "config_sha256": config_sha256,
        "canonical_data_sha256": training_data.manifest.canonical_adata_sha256,
        "split_content_sha256": training_data.split.split_content_sha256,
        "test_evaluations": 0,
        "canonical_test_truth_opened": False,
    }


def seal_r50_smoke(
    root,
    *,
    config,
    config_sha256,
    training_data,
    source,
    environment_sha256,
    best_checkpoint,
    last_checkpoint,
    validation_value,
):
    """Call only after exactly one continuous training/validation epoch succeeds."""
    if source.dirty or not source.formal_eligible:
        raise ValueError("R50 smoke requires clean published source")
    root = Path(root).resolve(strict=True)
    records = {}
    for role, checkpoint in (("best", best_checkpoint), ("last", last_checkpoint)):
        path = Path(checkpoint).resolve(strict=True)
        records[role] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "epoch": 1,
        }
    receipt = {
        **_identity(config, config_sha256, training_data, source.commit),
        "environment_sha256": environment_sha256,
        "epochs_completed": 1,
        "validation": [{"epoch": 1, "value": float(validation_value)}],
        "checkpoints": records,
    }
    # Validate before publishing success, including both files and zero PKL.
    _validate(
        root, receipt, config, config_sha256, training_data, source.commit, environment_sha256
    )
    atomic_json(root / "small_results/r50_smoke.json", receipt)
    return receipt


def _validate(
    root, receipt, config, config_sha256, training_data, source_commit, environment_sha256
):
    if config.training.formal_run_policy != "external_fixed_50":
        raise ValueError("R50 smoke requires the fixed50 policy")
    expected = _identity(config, config_sha256, training_data, source_commit)
    expected["environment_sha256"] = environment_sha256
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("R50 smoke identity or truth-scope mismatch")
    validation = receipt.get("validation", [])
    if (
        receipt.get("epochs_completed") != 1
        or len(validation) != 1
        or validation[0].get("epoch") != 1
        or not math.isfinite(float(validation[0].get("value", float("nan"))))
    ):
        raise ValueError("R50 smoke requires one finite validation after one epoch")
    records = receipt.get("checkpoints", {})
    if set(records) != {"best", "last"}:
        raise ValueError("R50 smoke requires best and true-last checkpoints")
    paths = set()
    for record in records.values():
        path = (root / record["path"]).resolve(strict=True)
        if (
            not path.is_relative_to(root)
            or path.suffix not in {".pt", ".ckpt"}
            or record.get("epoch") != 1
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError("R50 checkpoint identity mismatch")
        paths.add(path)
    observed = {p.resolve() for p in root.rglob("*") if p.suffix in {".pt", ".ckpt"}}
    if len(paths) != 2 or paths != observed:
        raise ValueError("R50 smoke requires exactly two distinct checkpoint paths")
    if any(
        p.suffix.lower() in {".pkl", ".pickle"} or p.name.startswith(".result-work-")
        for p in root.rglob("*")
    ):
        raise ValueError("R50 smoke zero-PKL/work postcondition failed")


def require_r50_smoke(
    root, *, config, config_sha256, training_data, source_commit, environment_sha256
):
    if root is None:
        raise ValueError("external R50 requires --smoke-run-root")
    root = Path(root).resolve(strict=True)
    if (root / "small_results/one_step_smoke.json").exists():
        path = root / "small_results/one_step_smoke.json"
        receipt = json.loads(path.read_text())
        validate_external_step(
            root,
            receipt,
            config=config,
            config_sha256=config_sha256,
            training_data=training_data,
            source_commit=source_commit,
            environment_sha256=environment_sha256,
        )
        return {"smoke_root": str(root), "receipt_sha256": sha256_file(path)}
    path = root / "small_results/r50_smoke.json"
    receipt = json.loads(path.read_text())
    _validate(
        root, receipt, config, config_sha256, training_data, source_commit, environment_sha256
    )
    return {"smoke_root": str(root), "receipt_sha256": sha256_file(path)}


def validate_external_step(
    root, receipt, *, config, config_sha256, training_data, source_commit, environment_sha256
):
    expected = _identity(config, config_sha256, training_data, source_commit)
    expected.update(
        schema="external-r50-one-step-v1",
        status="single_update_complete",
        environment_sha256=environment_sha256,
    )
    if config.training.formal_run_policy != "external_fixed_50" or any(
        receipt.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("single-step identity or scope mismatch")
    if (
        receipt.get("completed_steps") != 1
        or receipt.get("completed_epochs") != 0
        or receipt.get("checkpoint_serialization_exact") is not True
    ):
        raise ValueError("single-step update/checkpoint gate failed")
    path = (root / receipt["checkpoint_path"]).resolve(strict=True)
    checkpoints = {p.resolve() for p in root.rglob("*") if p.suffix in {".pt", ".ckpt"}}
    if (
        not path.is_relative_to(root)
        or checkpoints != {path}
        or sha256_file(path) != receipt["checkpoint_sha256"]
    ):
        raise ValueError("single-step diagnostic checkpoint changed")
    if any(
        p.suffix.lower() in {".pkl", ".pickle"} or p.name.startswith(".result-work-")
        for p in root.rglob("*")
    ):
        raise ValueError("single-step zero-PKL/work gate failed")


def seal_external_step(
    root, *, config, config_sha256, training_data, source, environment_sha256, checkpoint, update
):
    root = Path(root).resolve(strict=True)
    if source.dirty or not source.formal_eligible:
        raise ValueError("single-step needs clean published source")
    receipt = {
        **_identity(config, config_sha256, training_data, source.commit),
        **update,
        "schema": "external-r50-one-step-v1",
        "status": "single_update_complete",
        "environment_sha256": environment_sha256,
        "checkpoint_path": str(Path(checkpoint).resolve(strict=True).relative_to(root)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_role": "diagnostic_after_step_not_best",
    }
    validate_external_step(
        root,
        receipt,
        config=config,
        config_sha256=config_sha256,
        training_data=training_data,
        source_commit=source.commit,
        environment_sha256=environment_sha256,
    )
    atomic_json(root / "small_results/one_step_smoke.json", receipt)
    return receipt
