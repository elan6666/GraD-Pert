"""End-to-end native B2 training, prediction, evaluation, and receipts."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gradpert.config import ExperimentConfig, load_experiment_config
from gradpert.contracts import RunManifest, ServerArtifactPointer
from gradpert.data._io import atomic_json, atomic_text
from gradpert.evaluation import CanonicalEvaluationData
from gradpert.execution.artifact_run import seal_evaluation_outputs
from gradpert.execution.identity import (
    EnvironmentIdentity,
    SourceIdentity,
    inspect_environment,
    inspect_source_identity,
)
from gradpert.graphs import build_prediction_graph_view, load_dataset_graph_topology
from gradpert.hashing import sha256_file, sha256_json
from gradpert.modeling import CenterState, GraDPertJointModel
from gradpert.training.checkpoint import CheckpointIdentity
from gradpert.training.data import CanonicalTrainingData, write_training_data_receipt
from gradpert.training.inference import predict_frozen_controls
from gradpert.training.step import GraDPertStepEngine, build_native_optimizer
from gradpert.training.trainer import GraDPertTrainer
from gradpert.training.validation import evaluate_validation_macro_delta

CUDA_ALLOCATOR_CONFIG = "expandable_segments:True"


@dataclass(frozen=True)
class NativeRunResult:
    run_id: str
    run_root: Path
    run_manifest: RunManifest
    source: SourceIdentity
    environment: EnvironmentIdentity


def _integer_parameter(config: ExperimentConfig, name: str) -> int:
    parameters = config.model.parameters
    value = parameters[name].value
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"model parameter {name} must be an integer")
    return value


def _training_integer(config: ExperimentConfig, name: str) -> int:
    value = getattr(config.training, name).value
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"training parameter {name} must be an integer")
    return value


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _write_contract(path: Path, contract: Any) -> None:
    atomic_json(path, contract.model_dump(mode="json"))


def _write_or_require_text(path: Path, value: str, *, resume: bool) -> None:
    if resume:
        if not path.is_file() or path.read_text(encoding="utf-8") != value:
            raise ValueError(f"resumed run receipt differs: {path.name}")
        return
    atomic_text(path, value)


def _write_or_require_json(path: Path, value: dict[str, object], *, resume: bool) -> None:
    if resume:
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"resumed run receipt differs: {path.name}")
        return
    atomic_json(path, value)


def run_native_experiment(
    *,
    config_path: str | Path,
    data_root: str | Path,
    run_root: str | Path,
    run_id: str,
    run_seed: int,
    mode: Literal["smoke", "full"],
    device_name: str,
    repository_root: str | Path,
    formal: bool,
    development_commit: str | None = None,
    resume: bool = False,
) -> NativeRunResult:
    """Run one isolated smoke/full lifecycle with exactly one final test access."""

    if (
        device_name.startswith("cuda:")
        and os.environ.get("PYTORCH_ALLOC_CONF") != CUDA_ALLOCATOR_CONFIG
    ):
        raise RuntimeError("native CUDA runs require PYTORCH_ALLOC_CONF=" + CUDA_ALLOCATOR_CONFIG)
    import torch

    config_file = Path(config_path).resolve(strict=True)
    config = load_experiment_config(config_file)
    if config.model_id != "gradpert_b2" or config.model.family != "native_learned":
        raise ValueError("native runner requires one gradpert_b2 experiment config")
    if run_seed not in config.training.run_seeds:
        raise ValueError("run seed is outside the experiment config")
    if mode == "smoke" and run_seed != config.training.run_seeds[0]:
        raise ValueError("the one-epoch integration gate uses the first configured seed")
    destination = Path(run_root).resolve()
    if resume:
        if not destination.is_dir():
            raise FileNotFoundError("resumed run root must already exist")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("run root must be new and empty")
        destination.mkdir(parents=True, exist_ok=True)

    source = inspect_source_identity(
        repository_root,
        formal=formal,
        expected_repository=config.source_code.repository,
        development_commit=development_commit,
    )
    environment = inspect_environment(repository_root, device_name=device_name)
    config_sha256 = sha256_file(config_file)
    small_root = destination / "small_results"
    _write_or_require_text(
        small_root / "config.resolved.yaml",
        config_file.read_text(encoding="utf-8"),
        resume=resume,
    )
    _write_or_require_json(
        small_root / "source_identity.json",
        source.payload(),
        resume=resume,
    )
    _write_or_require_json(
        small_root / "environment.json",
        environment.payload(),
        resume=resume,
    )

    device = torch.device(device_name)
    _seed_everything(torch, run_seed)
    train_batch_size = _training_integer(config, "train_batch_size")
    eval_batch_size = _training_integer(config, "eval_batch_size")
    max_unique_conditions = _integer_parameter(config, "max_unique_conditions_per_batch")
    prototype_count = _integer_parameter(config, "prototype_count")
    max_epochs = _training_integer(config, "max_epochs")

    with (
        CanonicalTrainingData(
            dataset_id=config.dataset_id,
            protocol_id=config.data.protocol_id,
            data_root=data_root,
            run_seed=run_seed,
        ) as training_data,
        CanonicalEvaluationData(
            dataset_id=config.dataset_id,
            protocol_id=config.data.protocol_id,
            split_name="val",
            data_root=data_root,
        ) as validation_data,
    ):
        training_data.require_experiment_data_contract(
            registry_version=config.data.registry_version,
            split_policy=config.data.split_policy,
        )
        topology = load_dataset_graph_topology(
            dataset_id=config.dataset_id,
            protocol_id=config.data.protocol_id,
            data_root=data_root,
        )
        if topology.gene_ids != training_data.graph_gene_ids:
            raise ValueError("native topology and canonical graph axes differ")
        steps_per_epoch = training_data.steps_per_epoch(
            batch_size=train_batch_size,
            max_unique_conditions=max_unique_conditions,
        )
        model = GraDPertJointModel(
            graph_gene_count=training_data.manifest.n_graph_genes,
            expression_gene_count=training_data.manifest.n_expression_genes,
            prototype_count=prototype_count,
        ).to(device)
        optimizer = build_native_optimizer(
            model,
            learning_rate=float(config.training.learning_rate.value),
            weight_decay=float(config.training.weight_decay.value),
        )
        centers = CenterState.zeros(prototype_count=prototype_count, device=device)
        heldout_ids = tuple(
            sorted(
                {
                    anchor
                    for condition in (
                        *training_data.split.val_conditions,
                        *training_data.split.test_conditions,
                    )
                    for anchor in training_data.anchors_by_condition[condition]
                }
            )
        )
        engine = GraDPertStepEngine(
            model=model,
            topology=topology,
            optimizer=optimizer,
            centers=centers,
            run_seed=run_seed,
            total_schedule_steps=max_epochs * steps_per_epoch,
            heldout_target_ids=heldout_ids,
        )
        checkpoint_identity = CheckpointIdentity(
            source_commit=source.commit,
            source_tree_sha256=source.tree_sha256,
            config_sha256=config_sha256,
            environment_sha256=environment.payload_sha256,
            canonical_data_sha256=training_data.manifest.canonical_adata_sha256,
            split_content_sha256=training_data.split.split_content_sha256,
        )
        run_meta = {
            "schema_version": "native-run-meta-v1",
            "run_id": run_id,
            "mode": mode,
            "model_id": config.model_id,
            "dataset_id": config.dataset_id,
            "protocol_id": config.data.protocol_id,
            "run_seed": run_seed,
            "source": source.payload(),
            "environment": environment.payload(),
            "config_path": str(config_file),
            "config_sha256": config_sha256,
            "steps_per_epoch": steps_per_epoch,
            "max_epochs": 1 if mode == "smoke" else max_epochs,
            "early_stopping_patience": int(config.training.early_stopping_patience.value),
            "validation_monitor": config.training.monitor,
        }
        write_training_data_receipt(training_data, small_root / "training_data.json")
        trainer = GraDPertTrainer(
            engine=engine,
            checkpoint_identity=checkpoint_identity,
            run_root=destination,
            steps_per_epoch=steps_per_epoch,
            max_epochs=max_epochs,
            run_meta=run_meta,
        )
        if resume:
            trainer.resume()
        validation_anchors = {
            condition: training_data.anchors_by_condition[condition]
            for condition in training_data.split.val_conditions
        }

        def validate(current_model: GraDPertJointModel, epoch: int) -> float:
            result = evaluate_validation_macro_delta(
                model=current_model,
                topology=topology,
                data=validation_data,
                anchors_by_condition=validation_anchors,
                device=device,
                decode_batch_size=eval_batch_size,
            )
            atomic_json(
                small_root / f"validation.epoch-{epoch:03d}.json",
                {
                    "schema_version": "native-validation-v1",
                    "epoch": epoch,
                    **result.__dict__,
                },
            )
            return result.txpert_macro_pearson_delta

        progress = trainer.fit(
            mode=mode,
            train_epoch_factory=lambda epoch: training_data.iter_train_epoch(
                epoch=epoch,
                device=device,
                batch_size=train_batch_size,
                max_unique_conditions=max_unique_conditions,
            ),
            validate=validate,
        )
        best_checkpoint_sha256 = sha256_file(trainer.best_checkpoint)
        atomic_json(
            small_root / "training_receipt.json",
            {
                "schema_version": "native-training-receipt-v1",
                "model_id": config.model_id,
                "dataset_id": config.dataset_id,
                "mode": mode,
                "epochs_requested": 1 if mode == "smoke" else max_epochs,
                "epochs_completed": progress.completed_epochs,
                "optimizer_steps": progress.global_step,
                "early_stopping_patience": int(config.training.early_stopping_patience.value),
                "validation_monitor": config.training.monitor,
                "canonical_test_truth_present_during_fit": False,
                "checkpoint_sha256": best_checkpoint_sha256,
            },
        )

        test_control_manifest_sha256: str | None = None

        def evaluate_test_once(current_model: GraDPertJointModel) -> None:
            nonlocal test_control_manifest_sha256
            with CanonicalEvaluationData(
                dataset_id=config.dataset_id,
                protocol_id=config.data.protocol_id,
                split_name="test",
                data_root=data_root,
            ) as test_data:
                test_control_manifest_sha256 = test_data.control_manifest_file_sha256
                test_anchors = {
                    condition: training_data.anchors_by_condition[condition]
                    for condition in training_data.split.test_conditions
                }
                condition_predictions = predict_frozen_controls(
                    model=current_model,
                    prediction_view=build_prediction_graph_view(topology),
                    control_manifest=test_data.control_manifest,
                    anchors_by_condition=test_anchors,
                    load_control_rows=test_data.load_control_rows,
                    device=device,
                    decode_batch_size=eval_batch_size,
                )
                seal_evaluation_outputs(
                    destination=destination,
                    config=config,
                    config_sha256=config_sha256,
                    run_id=run_id,
                    run_seed=run_seed,
                    source=source,
                    environment=environment,
                    training_data=training_data,
                    test_data=test_data,
                    predictions=condition_predictions,
                    checkpoint_sha256=best_checkpoint_sha256,
                )

        trainer.test_best_once(evaluate_test_once)

    if test_control_manifest_sha256 is None:
        raise RuntimeError("test callback did not record its control manifest hash")

    run_manifest = RunManifest(
        schema_version="run-manifest-v1",
        run_id=run_id,
        model_id=config.model_id,
        dataset_id=config.dataset_id,
        protocol_id=config.data.protocol_id,
        run_seed=run_seed,
        source_commit=source.commit,
        source_dirty=source.dirty,
        formal_eligible=source.formal_eligible,
        config_sha256=config_sha256,
        environment_sha256=environment.payload_sha256,
        canonical_data_sha256=training_data.manifest.canonical_adata_sha256,
        split_content_sha256=training_data.split.split_content_sha256,
        control_manifest_sha256=test_control_manifest_sha256,
        status="evaluated",
        best_checkpoint_sha256=best_checkpoint_sha256,
        test_evaluations=trainer.progress.test_evaluations,
    )
    _write_contract(small_root / "run_manifest.json", run_manifest)
    prediction_manifest_path = destination / "small_results" / "prediction_manifest.json"
    evaluation_manifest_path = destination / "small_results" / "evaluation_manifest.json"
    pointer = ServerArtifactPointer(
        schema_version="server-artifact-pointer-v1",
        run_id=run_id,
        source_commit=source.commit,
        server_root=str(destination),
        prediction_manifest_path=str(prediction_manifest_path),
        prediction_manifest_sha256=sha256_file(prediction_manifest_path),
        evaluation_manifest_path=str(evaluation_manifest_path),
        evaluation_manifest_sha256=sha256_file(evaluation_manifest_path),
        synchronized_large_artifacts=False,
    )
    _write_contract(small_root / "server_pointer.json", pointer)
    atomic_json(
        small_root / "run_identity.json",
        {
            "schema_version": "run-identity-v1",
            "run_manifest_sha256": sha256_json(run_manifest.model_dump(mode="json")),
            "source_tree_sha256": source.tree_sha256,
            "formal_eligible": source.formal_eligible,
        },
    )
    # ``last.pt`` is a crash/resume checkpoint only. A completed evaluated run
    # retains the hash-pinned best checkpoint used for inference and removes the
    # redundant lifecycle copy after every durable receipt has been written.
    trainer.last_checkpoint.unlink(missing_ok=True)
    atomic_json(
        small_root / "checkpoint_retention.json",
        {
            "schema_version": "checkpoint-retention-v1",
            "policy": "best_only_after_successful_evaluation",
            "best_checkpoint_path": str(trainer.best_checkpoint),
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "last_checkpoint_removed": True,
        },
    )
    return NativeRunResult(
        run_id=run_id,
        run_root=destination,
        run_manifest=run_manifest,
        source=source,
        environment=environment,
    )
