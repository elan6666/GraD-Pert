"""End-to-end native B2 training, prediction, evaluation, and receipts."""

from __future__ import annotations

import csv
import json
import os
import random
import resource
import sys
import time
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
from gradpert.graphs import load_dataset_graph_topology
from gradpert.hashing import sha256_file, sha256_json
from gradpert.modeling import CenterState, GraDPertJointModel
from gradpert.pilots import load_reduced_graph_topology
from gradpert.training.checkpoint import CheckpointIdentity
from gradpert.training.data import CanonicalTrainingData, write_training_data_receipt
from gradpert.training.inference import predict_frozen_controls
from gradpert.training.step import GraDPertStepEngine, build_native_optimizer
from gradpert.training.systems import NativeSystemOptions
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


def _optional_string_parameter(config: ExperimentConfig, name: str) -> str | None:
    parameter = config.model.parameters.get(name)
    if parameter is None:
        return None
    if not isinstance(parameter.value, str) or not parameter.value:
        raise ValueError(f"model parameter {name} must be a non-empty string")
    return parameter.value


def _optional_boolean_parameter(
    config: ExperimentConfig, name: str, *, default: bool = False
) -> bool:
    parameter = config.model.parameters.get(name)
    if parameter is None:
        return default
    if not isinstance(parameter.value, bool):
        raise ValueError(f"model parameter {name} must be a boolean")
    return parameter.value


def _optional_integer_parameter(config: ExperimentConfig, name: str, *, default: int) -> int:
    parameter = config.model.parameters.get(name)
    if parameter is None:
        return default
    if not isinstance(parameter.value, int) or isinstance(parameter.value, bool):
        raise ValueError(f"model parameter {name} must be an integer")
    return parameter.value


def _native_system_options(config: ExperimentConfig) -> NativeSystemOptions:
    label = _optional_string_parameter(config, "systems_optimizations") or "disabled"
    options = NativeSystemOptions(
        merged_hdf5_reads=_optional_boolean_parameter(config, "systems_merged_hdf5_reads"),
        control_expression_cache=_optional_boolean_parameter(
            config, "systems_control_expression_cache"
        ),
        background_prefetch=_optional_boolean_parameter(config, "systems_background_prefetch"),
        resident_graph_tensors=_optional_boolean_parameter(
            config, "systems_resident_graph_tensors"
        ),
        validation_expression_cache=_optional_boolean_parameter(
            config, "systems_validation_expression_cache"
        ),
        buffered_training_logs=_optional_boolean_parameter(
            config, "systems_buffered_training_logs"
        ),
        single_checkpoint_serialization=_optional_boolean_parameter(
            config, "systems_single_checkpoint_serialization"
        ),
        pin_memory=_optional_boolean_parameter(config, "systems_pin_memory"),
        nonblocking_transfer=_optional_boolean_parameter(config, "systems_nonblocking_transfer"),
        prefetch_depth=_optional_integer_parameter(config, "systems_prefetch_depth", default=1),
        log_buffer_steps=_optional_integer_parameter(
            config, "systems_log_buffer_steps", default=64
        ),
    )
    expected_label = "all_seven_semantics_preserving_v1" if options.enabled else "disabled"
    if label != expected_label:
        raise ValueError(
            f"systems_optimizations={label!r} differs from explicit flags ({expected_label})"
        )
    return options


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


def _read_step_timings(path: Path) -> list[dict[str, float]]:
    fields = (
        "batch_cell_count",
        "data_read_ms",
        "host_to_device_ms",
        "view_build_ms",
        "teacher_forward_ms",
        "student_global_ms",
        "student_local_ms",
        "prediction_ms",
        "backward_update_ms",
        "step_wall_ms",
    )
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("performance receipt requires at least one training step")
    return [{field: float(row[field]) for field in fields} for row in rows]


def _peak_cpu_ram_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def run_native_experiment(
    *,
    config_path: str | Path,
    data_root: str | Path,
    run_root: str | Path,
    run_id: str,
    run_seed: int,
    mode: Literal["smoke", "pilot", "full"],
    device_name: str,
    repository_root: str | Path,
    formal: bool,
    development_commit: str | None = None,
    resume: bool = False,
) -> NativeRunResult:
    """Run one isolated smoke/pilot/full lifecycle with exactly one final test access."""

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
    if mode == "pilot" and config.training.formal_run_policy != "fixed_epoch_pilot":
        raise ValueError("native pilot mode requires formal_run_policy=fixed_epoch_pilot")
    if mode == "full" and config.training.formal_run_policy != "smoke_then_full":
        raise ValueError("native full mode requires formal_run_policy=smoke_then_full")
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
    system_options = _native_system_options(config)
    cold_start_started = time.perf_counter()
    graph_axis_policy = _optional_string_parameter(config, "graph_axis_policy")
    reduced_manifest = None
    if graph_axis_policy is None or graph_axis_policy == "canonical_full":
        topology = load_dataset_graph_topology(
            dataset_id=config.dataset_id,
            protocol_id=config.data.protocol_id,
            data_root=data_root,
        )
        graph_manifest_path = (
            Path(data_root) / config.dataset_id / config.data.protocol_id / ("graphs/manifest.json")
        )
    elif graph_axis_policy == "recomputed_top500_union_candidate_targets":
        relative_root = _optional_string_parameter(config, "runtime_graph_root")
        if relative_root is None:
            raise ValueError("reduced graph policy requires runtime_graph_root")
        relative_path = Path(relative_root)
        if relative_path.is_absolute() or any(
            part in {"", ".", ".."} for part in relative_path.parts
        ):
            raise ValueError("runtime_graph_root must be a safe relative path")
        graph_root = Path(data_root).joinpath(*relative_path.parts)
        topology, reduced_manifest = load_reduced_graph_topology(graph_root)
        if (
            reduced_manifest.dataset_id != config.dataset_id
            or reduced_manifest.protocol_id != config.data.protocol_id
        ):
            raise ValueError("reduced graph identity differs from the experiment config")
        graph_manifest_path = graph_root / "manifest.json"
    else:
        raise ValueError(f"unsupported graph_axis_policy: {graph_axis_policy}")

    with (
        CanonicalTrainingData(
            dataset_id=config.dataset_id,
            protocol_id=config.data.protocol_id,
            data_root=data_root,
            run_seed=run_seed,
            graph_gene_ids_override=topology.gene_ids,
            graph_manifest_path_override=graph_manifest_path,
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
        if topology.gene_ids != training_data.graph_gene_ids:
            raise ValueError("native topology and canonical graph axes differ")
        if reduced_manifest is not None and (
            reduced_manifest.canonical_data_sha256 != training_data.manifest.canonical_adata_sha256
            or reduced_manifest.split_content_sha256 != training_data.split.split_content_sha256
        ):
            raise ValueError("reduced graph canonical data or split identity differs")
        training_cache_ms = training_data.configure_system_optimizations(system_options)
        validation_cache_ms = validation_data.configure_expression_cache(
            enabled=system_options.validation_expression_cache
        )
        cache_build_ms = training_cache_ms + validation_cache_ms
        steps_per_epoch = training_data.steps_per_epoch(
            batch_size=train_batch_size,
            max_unique_conditions=max_unique_conditions,
        )
        model = GraDPertJointModel(
            graph_gene_count=len(training_data.graph_gene_ids),
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
            resident_graph_tensors=system_options.resident_graph_tensors,
            capture_equivalence_health=system_options.enabled,
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
            "graph_axis_policy": graph_axis_policy or "canonical_full",
            "runtime_graph_gene_count": len(training_data.graph_gene_ids),
            "runtime_graph_gene_order_sha256": sha256_json(list(training_data.graph_gene_ids)),
            "systems_optimizations": system_options.payload(),
        }
        write_training_data_receipt(training_data, small_root / "training_data.json")
        trainer = GraDPertTrainer(
            engine=engine,
            checkpoint_identity=checkpoint_identity,
            run_root=destination,
            steps_per_epoch=steps_per_epoch,
            max_epochs=max_epochs,
            run_meta=run_meta,
            log_buffer_steps=(
                system_options.log_buffer_steps if system_options.buffered_training_logs else 1
            ),
            single_checkpoint_serialization=system_options.single_checkpoint_serialization,
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
                prediction_view=engine.prediction_view,
            )
            atomic_json(
                small_root / f"validation.epoch-{epoch:03d}.json",
                {
                    "schema_version": "native-validation-v1",
                    "epoch": epoch,
                    **result.__dict__,
                },
            )
            return float(result.txpert_macro_pearson_delta)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        cold_start_ms = (time.perf_counter() - cold_start_started) * 1000.0
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
        if system_options.enabled:
            if engine.first_step_health is None:
                raise RuntimeError("enabled systems did not capture first-step equivalence health")
            atomic_json(
                small_root / "first_step_equivalence.json",
                engine.first_step_health,
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
                    prediction_view=engine.prediction_view,
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
        timing_rows = _read_step_timings(small_root / "train_steps.csv")
        warmup_steps = min(10, max(0, len(timing_rows) - 1))
        measured = timing_rows[warmup_steps:]
        measured_wall_ms = sum(
            row["data_read_ms"] + row["host_to_device_ms"] + row["step_wall_ms"] for row in measured
        )
        measured_cells = int(sum(row["batch_cell_count"] for row in measured))
        stage_fields = (
            "data_read_ms",
            "host_to_device_ms",
            "view_build_ms",
            "teacher_forward_ms",
            "student_global_ms",
            "student_local_ms",
            "prediction_ms",
            "backward_update_ms",
            "step_wall_ms",
        )
        metrics_summary = json.loads(
            (small_root / "metrics_summary.json").read_text(encoding="utf-8")
        )
        atomic_json(
            small_root / "performance_receipt.json",
            {
                "schema_version": "native-performance-v2",
                "run_mode": mode,
                "epochs_completed": progress.completed_epochs,
                "selection_policy": (
                    "validation_selected_best_checkpoint_test_once"
                    if mode == "full"
                    else "speed_only_one_epoch_metrics_non_decisional"
                ),
                "graph_axis_policy": graph_axis_policy or "canonical_full",
                "expression_gene_count": len(training_data.expression_gene_ids),
                "output_gene_count": len(training_data.expression_gene_ids),
                "evaluation_gene_count": len(training_data.expression_gene_ids),
                "graph_node_count": topology.n_nodes,
                "graph_nonself_edge_count": sum(
                    len(graph.edges) for graph in topology.sources.values()
                ),
                "cold_start_ms": cold_start_ms,
                "cache_build_ms": cache_build_ms,
                "one_epoch_fit_wall_ms": trainer.fit_wall_ms,
                "one_epoch_training_wall_ms": trainer.training_wall_ms,
                "fit_wall_ms": trainer.fit_wall_ms,
                "training_wall_ms": trainer.training_wall_ms,
                "validation_ms": trainer.validation_wall_ms,
                "checkpoint_ms": trainer.checkpoint_wall_ms,
                "logging_ms": trainer.logging_wall_ms,
                "warmup_steps": warmup_steps,
                "measured_steps": len(measured),
                "measured_cells": measured_cells,
                "measured_end_to_end_wall_ms": measured_wall_ms,
                "steps_per_second": len(measured) / (measured_wall_ms / 1000.0),
                "cells_per_second": measured_cells / (measured_wall_ms / 1000.0),
                "mean_stage_ms": {
                    field: sum(row[field] for row in measured) / len(measured)
                    for field in stage_fields
                },
                "peak_allocated_gpu_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                ),
                "peak_reserved_gpu_bytes": (
                    int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
                ),
                "peak_cpu_ram_bytes": _peak_cpu_ram_bytes(),
                "headline_metrics": metrics_summary["metrics"],
                "headline_metrics_non_decisional": (
                    None if mode == "full" else metrics_summary["metrics"]
                ),
                "systems_optimizations": system_options.payload(),
                "checkpoint_peer_method": trainer.checkpoint_peer_method,
            },
        )
        atomic_json(
            small_root / "systems_runtime.json",
            {
                "schema_version": "native-systems-runtime-v1",
                "requested": system_options.payload(),
                "training_pipeline": training_data.pipeline_stats.payload(),
                "validation_cache": validation_data.cache_stats.payload(),
                "resident_graph_tensors": {
                    "student": model.student_encoder.resident_graph_tensor_payload(),
                    "teacher": model.teacher_encoder.resident_graph_tensor_payload(),
                },
                "checkpoint": {
                    "single_serialization_per_epoch": (
                        system_options.single_checkpoint_serialization
                    ),
                    "peer_method": trainer.checkpoint_peer_method,
                },
                "log_buffer_steps": (
                    system_options.log_buffer_steps if system_options.buffered_training_logs else 1
                ),
            },
        )

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
