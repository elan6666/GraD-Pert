"""Deterministic server experiment matrix and learned-smoke dependency gates."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gradpert.config import load_experiment_config, verify_config_matrix
from gradpert.config.matrix import DATASET_IDS
from gradpert.contracts import RunManifest
from gradpert.hashing import sha256_file

ExperimentPhase = Literal["smoke", "nonlearned", "full"]

LEARNED_MODEL_IDS = ("gradpert_b2", "gears", "txpert_public")
NONLEARNED_MODEL_IDS = (
    "matched_control_mean",
    "global_train_delta",
    "general_train_delta",
)


@dataclass(frozen=True)
class MatrixRuntime:
    project_root: Path
    config_root: Path
    data_root: Path
    runs_root: Path
    native_python: Path
    gears_python: Path
    gears_checkout: Path
    gears_data_root: Path
    txpert_python: Path
    txpert_checkout: Path
    devices: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.devices or any(not item for item in self.devices):
            raise ValueError("at least one non-empty device is required")


@dataclass(frozen=True)
class ExperimentTask:
    task_id: str
    phase: ExperimentPhase
    model_id: str
    dataset_id: str
    run_seed: int
    run_id: str
    config_path: Path
    config_sha256: str
    run_root: Path
    device: str
    expected_commit: str
    formal: bool
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    expected_epochs: int | None

    def payload(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "phase": self.phase,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "run_seed": self.run_seed,
            "run_id": self.run_id,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "run_root": str(self.run_root),
            "device": self.device,
            "expected_commit": self.expected_commit,
            "formal": self.formal,
            "command": list(self.command),
            "environment": dict(self.environment),
            "expected_epochs": self.expected_epochs,
        }


def _identity_arguments(*, formal: bool, expected_commit: str) -> tuple[str, ...]:
    if not expected_commit:
        raise ValueError("expected_commit must be non-empty")
    return ("--formal",) if formal else ("--development-commit", expected_commit)


def _run_identity(
    namespace: str, phase: ExperimentPhase, model: str, dataset: str, seed: int
) -> str:
    if not namespace or any(part in namespace for part in ("/", "\\", "..")):
        raise ValueError("run namespace must be a safe non-empty path component")
    return f"{namespace}__{phase}__{model}__{dataset}__seed{seed}"


def _native_command(
    *,
    runtime: MatrixRuntime,
    phase: ExperimentPhase,
    config_path: Path,
    run_root: Path,
    run_id: str,
    seed: int,
    device: str,
    identity_arguments: tuple[str, ...],
    resume: bool,
) -> tuple[str, ...]:
    mode = "smoke" if phase == "smoke" else "full"
    command = (
        str(runtime.native_python),
        "-m",
        "gradpert",
        "model",
        mode,
        "--config",
        str(config_path),
        "--data-root",
        str(runtime.data_root),
        "--run-root",
        str(run_root),
        "--run-id",
        run_id,
        "--run-seed",
        str(seed),
        "--device",
        device,
        "--repository-root",
        str(runtime.project_root),
        *identity_arguments,
    )
    return (*command, "--resume") if resume else command


def _baseline_command(
    *,
    runtime: MatrixRuntime,
    config_path: Path,
    run_root: Path,
    run_id: str,
    identity_arguments: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        str(runtime.native_python),
        "-m",
        "gradpert",
        "baseline",
        "--config",
        str(config_path),
        "--data-root",
        str(runtime.data_root),
        "--run-root",
        str(run_root),
        "--run-id",
        run_id,
        "--repository-root",
        str(runtime.project_root),
        *identity_arguments,
    )


def _official_command(
    *,
    runtime: MatrixRuntime,
    model_id: str,
    config_path: Path,
    run_root: Path,
    run_id: str,
    device: str,
    identity_arguments: tuple[str, ...],
) -> tuple[str, ...]:
    if model_id == "gears":
        python = runtime.gears_python
        module = "benchmarks.gears.runner"
        checkout = runtime.gears_checkout
    elif model_id == "txpert_public":
        python = runtime.txpert_python
        module = "benchmarks.txpert.runner"
        checkout = runtime.txpert_checkout
    else:  # pragma: no cover - private caller fixes the model set
        raise AssertionError(f"unsupported official model: {model_id}")
    command = (
        str(python),
        "-m",
        module,
        "--config",
        str(config_path),
        "--official-checkout",
        str(checkout),
        "--data-root",
        str(runtime.data_root),
        "--run-root",
        str(run_root),
        "--run-id",
        run_id,
        "--device",
        device,
        "--repository-root",
        str(runtime.project_root),
        *identity_arguments,
    )
    if model_id == "gears":
        return (*command, "--official-data-root", str(runtime.gears_data_root))
    return command


def _txpert_process_device(device: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Isolate one physical GPU while preserving TxPert's local CUDA convention."""

    prefix = "cuda:"
    if not device.startswith(prefix):
        raise ValueError("TxPert matrix devices must use an explicit cuda:<index> value")
    physical_index = device.removeprefix(prefix)
    if not physical_index.isdigit():
        raise ValueError("TxPert matrix devices must use a non-negative CUDA index")
    return "cuda:0", (("CUDA_VISIBLE_DEVICES", physical_index),)


def build_experiment_tasks(
    *,
    phase: ExperimentPhase,
    runtime: MatrixRuntime,
    namespace: str,
    expected_commit: str,
    formal: bool,
    resume_native_full: bool = False,
) -> tuple[ExperimentTask, ...]:
    """Return the exact preregistered phase with no filesystem mutation."""

    report = verify_config_matrix(runtime.config_root)
    if report["count"] != 30:
        raise ValueError("experiment matrix must contain exactly 30 configs")
    identity_arguments = _identity_arguments(formal=formal, expected_commit=expected_commit)
    if phase == "smoke":
        coordinates = tuple(
            (model_id, dataset_id, 1)
            for model_id in LEARNED_MODEL_IDS
            for dataset_id in DATASET_IDS
        )
    elif phase == "nonlearned":
        coordinates = tuple(
            (model_id, dataset_id, 1)
            for model_id in NONLEARNED_MODEL_IDS
            for dataset_id in DATASET_IDS
        )
    elif phase == "full":
        if not formal:
            raise ValueError("full native matrix is formal-only")
        coordinates = tuple(
            ("gradpert_b2", dataset_id, seed) for dataset_id in DATASET_IDS for seed in (1, 2, 3, 4)
        )
    else:  # pragma: no cover - Literal plus argparse closes this
        raise ValueError(f"unsupported phase: {phase}")

    pythonpath = os.pathsep.join((str(runtime.project_root / "src"), str(runtime.project_root)))
    tasks: list[ExperimentTask] = []
    for index, (model_id, dataset_id, seed) in enumerate(coordinates):
        config_path = runtime.config_root / model_id / f"{dataset_id}.yaml"
        config = load_experiment_config(config_path)
        if seed not in config.training.run_seeds:
            raise ValueError(f"task seed is absent from config: {model_id}/{dataset_id}/{seed}")
        run_id = _run_identity(namespace, phase, model_id, dataset_id, seed)
        task_root = runtime.runs_root / phase / model_id / dataset_id / f"seed-{seed}"
        device = runtime.devices[index % len(runtime.devices)]
        environment: tuple[tuple[str, str], ...] = ()
        if model_id == "gradpert_b2":
            command = _native_command(
                runtime=runtime,
                phase=phase,
                config_path=config_path,
                run_root=task_root,
                run_id=run_id,
                seed=seed,
                device=device,
                identity_arguments=identity_arguments,
                resume=resume_native_full and phase == "full",
            )
            expected_epochs = 1 if phase == "smoke" else int(config.training.max_epochs.value)
        elif model_id in LEARNED_MODEL_IDS:
            command_device = device
            environment = (("PYTHONPATH", pythonpath),)
            if model_id == "txpert_public":
                command_device, device_environment = _txpert_process_device(device)
                environment = (*environment, *device_environment)
            command = _official_command(
                runtime=runtime,
                model_id=model_id,
                config_path=config_path,
                run_root=task_root,
                run_id=run_id,
                device=command_device,
                identity_arguments=identity_arguments,
            )
            expected_epochs = 1
        else:
            command = _baseline_command(
                runtime=runtime,
                config_path=config_path,
                run_root=task_root,
                run_id=run_id,
                identity_arguments=identity_arguments,
            )
            expected_epochs = None
        tasks.append(
            ExperimentTask(
                task_id=f"{phase}/{model_id}/{dataset_id}/seed-{seed}",
                phase=phase,
                model_id=model_id,
                dataset_id=dataset_id,
                run_seed=seed,
                run_id=run_id,
                config_path=config_path,
                config_sha256=sha256_file(config_path),
                run_root=task_root,
                device=device,
                expected_commit=expected_commit,
                formal=formal,
                command=command,
                environment=environment,
                expected_epochs=expected_epochs,
            )
        )
    return tuple(tasks)


def require_completed_task(task: ExperimentTask) -> Path:
    """Validate an existing evaluated run before an orchestrator skips it."""

    manifest_path = task.run_root / "small_results" / "run_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"completed task manifest is missing or unsafe: {manifest_path}")
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    expected = (
        task.run_id,
        task.model_id,
        task.dataset_id,
        task.run_seed,
        task.config_sha256,
        task.expected_commit,
    )
    observed = (
        manifest.run_id,
        manifest.model_id,
        manifest.dataset_id,
        manifest.run_seed,
        manifest.config_sha256,
        manifest.source_commit,
    )
    if observed != expected:
        raise ValueError(f"completed task identity mismatch: {manifest_path}")
    if manifest.status != "evaluated" or manifest.test_evaluations != 1:
        raise ValueError(f"completed task lifecycle is incomplete: {manifest_path}")
    if task.formal and not manifest.formal_eligible:
        raise ValueError(f"completed task is not formal-eligible: {manifest_path}")
    return manifest_path


def require_learned_smoke_gate(
    *,
    runtime: MatrixRuntime,
    expected_commit: str,
    require_formal: bool,
) -> tuple[Path, ...]:
    """Require all 15 evaluated seed-1 receipts before native full training."""

    verified: list[Path] = []
    fairness_by_dataset: dict[str, tuple[str, str, str, str]] = {}
    for model_id in LEARNED_MODEL_IDS:
        for dataset_id in DATASET_IDS:
            config_path = runtime.config_root / model_id / f"{dataset_id}.yaml"
            manifest_path = (
                runtime.runs_root
                / "smoke"
                / model_id
                / dataset_id
                / "seed-1"
                / "small_results"
                / "run_manifest.json"
            )
            if not manifest_path.is_file() or manifest_path.is_symlink():
                raise ValueError(f"learned smoke receipt is missing or unsafe: {manifest_path}")
            manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            expected = (model_id, dataset_id, 1, sha256_file(config_path))
            observed = (
                manifest.model_id,
                manifest.dataset_id,
                manifest.run_seed,
                manifest.config_sha256,
            )
            if observed != expected:
                raise ValueError(f"learned smoke identity mismatch: {manifest_path}")
            if (
                manifest.status != "evaluated"
                or manifest.test_evaluations != 1
                or manifest.source_commit != expected_commit
            ):
                raise ValueError(f"learned smoke lifecycle/commit gate failed: {manifest_path}")
            if require_formal and not manifest.formal_eligible:
                raise ValueError(f"learned smoke is not formal-eligible: {manifest_path}")
            training_receipt_path = manifest_path.parent / "training_receipt.json"
            if not training_receipt_path.is_file() or training_receipt_path.is_symlink():
                raise ValueError(
                    f"learned smoke training receipt is missing or unsafe: {training_receipt_path}"
                )
            training_receipt = json.loads(training_receipt_path.read_text(encoding="utf-8"))
            if not isinstance(training_receipt, dict):
                raise ValueError(
                    f"learned smoke training receipt is invalid: {training_receipt_path}"
                )
            no_test_truth = (
                training_receipt.get("canonical_test_truth_present_during_fit") is False
                or training_receipt.get("canonical_test_loader_present_during_fit") is False
            )
            if (
                training_receipt.get("model_id") != model_id
                or training_receipt.get("dataset_id") != dataset_id
                or training_receipt.get("epochs_requested") != 1
                or training_receipt.get("epochs_completed") != 1
                or training_receipt.get("checkpoint_sha256") != manifest.best_checkpoint_sha256
                or not no_test_truth
            ):
                raise ValueError(f"learned smoke training gate failed: {training_receipt_path}")
            fairness = (
                manifest.protocol_id,
                manifest.canonical_data_sha256,
                manifest.split_content_sha256,
                manifest.control_manifest_sha256,
            )
            previous = fairness_by_dataset.setdefault(dataset_id, fairness)
            if previous != fairness:
                raise ValueError(f"learned smoke fairness hashes differ for dataset: {dataset_id}")
            verified.append(manifest_path)
    return tuple(verified)
