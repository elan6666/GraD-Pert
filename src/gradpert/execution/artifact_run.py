"""Shared condition-prediction to evaluated-run artifact lifecycle."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from gradpert.artifacts import (
    PredictionConditionArrays,
    PredictionProvenance,
    seal_frozen_evaluation_bundle,
    seal_prediction_artifact,
)
from gradpert.config import ExperimentConfig
from gradpert.contracts import RunManifest, ServerArtifactPointer
from gradpert.data._io import atomic_json
from gradpert.evaluation import (
    CanonicalEvaluationData,
    load_evaluation_state,
    write_small_metric_exports,
)
from gradpert.execution.identity import EnvironmentIdentity, SourceIdentity
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.data import CanonicalTrainingData


@dataclass(frozen=True)
class SealedEvaluatedRun:
    run_manifest: RunManifest
    prediction_manifest_path: Path
    evaluation_manifest_path: Path


@dataclass(frozen=True)
class SealedEvaluationOutputs:
    prediction_manifest_path: Path
    evaluation_manifest_path: Path
    result_pkl_path: Path | None


def _write_contract(path: Path, contract: Any) -> None:
    atomic_json(path, contract.model_dump(mode="json"))


def seal_evaluation_outputs(
    *,
    destination: Path,
    config: ExperimentConfig,
    config_sha256: str,
    run_id: str,
    run_seed: int,
    source: SourceIdentity,
    environment: EnvironmentIdentity,
    training_data: CanonicalTrainingData,
    test_data: CanonicalEvaluationData,
    predictions: Sequence[PredictionConditionArrays],
    checkpoint_sha256: str | None,
) -> SealedEvaluationOutputs:
    """Seal receipts always and materialize one deduplicated result PKL only on request."""

    small_root = destination / "small_results"
    artifact_root = destination / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    graph_manifest_path = training_data.layout.root / "graphs" / "manifest.json"
    graph_manifest = (
        json.loads(graph_manifest_path.read_text(encoding="utf-8"))
        if graph_manifest_path.is_file()
        else None
    )
    forbidden_existing = [
        path
        for path in (
            artifact_root / "prediction.pkl",
            artifact_root / "evaluation.pkl",
            artifact_root / config.artifacts.result_pkl_name,
        )
        if path.exists()
    ]
    if forbidden_existing:
        raise RuntimeError(
            "artifact destination is not fresh: "
            + ", ".join(str(path) for path in forbidden_existing)
        )
    with TemporaryDirectory(prefix=".result-work-", dir=destination) as temporary:
        work_root = Path(temporary)
        prediction = seal_prediction_artifact(
            work_root / "prediction-stage.pkl",
            provenance=PredictionProvenance(
                model_id=config.model_id,
                dataset_id=config.dataset_id,
                protocol_id=config.data.protocol_id,
                run_id=run_id,
                run_seed=run_seed,
                source_commit=source.commit,
                source_dirty=source.dirty,
                formal_eligible=source.formal_eligible,
                config_sha256=config_sha256,
                environment_sha256=environment.payload_sha256,
                canonical_data_sha256=training_data.manifest.canonical_adata_sha256,
                gene_order_sha256=training_data.manifest.expression_gene_order_sha256,
                split_content_sha256=training_data.split.split_content_sha256,
                control_manifest_sha256=test_data.control_manifest_file_sha256,
                checkpoint_sha256=checkpoint_sha256,
            ),
            gene_ids=training_data.expression_gene_ids,
            conditions=predictions,
        )
        prediction_manifest_path = small_root / "prediction_manifest.json"
        _write_contract(prediction_manifest_path, prediction.manifest)
        state = load_evaluation_state(
            dataset_id=config.dataset_id,
            protocol_id=config.data.protocol_id,
            data_root=training_data.layout.data_root,
        )
        keep_result = config.artifacts.result_mode == "single_pkl"
        result_path = (
            artifact_root / config.artifacts.result_pkl_name
            if keep_result
            else work_root / config.artifacts.result_pkl_name
        )
        evaluation = seal_frozen_evaluation_bundle(
            result_path,
            prediction=prediction,
            data=test_data,
            state=state,
        )
        evaluation_manifest_path = small_root / "evaluation_manifest.json"
        _write_contract(evaluation_manifest_path, evaluation.manifest)
        write_small_metric_exports(evaluation, small_root)
        atomic_json(
            small_root / "inference_recipe.json",
            {
                "schema_version": config.artifacts.inference_recipe_schema_version,
                "model_id": config.model_id,
                "dataset_id": config.dataset_id,
                "protocol_id": config.data.protocol_id,
                "run_id": run_id,
                "run_seed": run_seed,
                "source_commit": source.commit,
                "source_tree_sha256": source.tree_sha256,
                "environment_sha256": environment.payload_sha256,
                "config_sha256": config_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "canonical_data_sha256": training_data.manifest.canonical_adata_sha256,
                "gene_order_sha256": training_data.manifest.expression_gene_order_sha256,
                "graph_gene_order_sha256": training_data.manifest.graph_gene_order_sha256,
                "graph_manifest_path": str(graph_manifest_path) if graph_manifest else None,
                "graph_manifest_sha256": (
                    sha256_file(graph_manifest_path) if graph_manifest else None
                ),
                "graph_topology_content_sha256": (
                    graph_manifest.get("topology_content_sha256") if graph_manifest else None
                ),
                "split_content_sha256": training_data.split.split_content_sha256,
                "control_manifest_sha256": test_data.control_manifest_file_sha256,
                "result_mode": config.artifacts.result_mode,
                "result_pkl_path": str(result_path) if keep_result else None,
                "result_pkl_sha256": evaluation.file_sha256 if keep_result else None,
                "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
                "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
                "condition_input_control_row_ids": {
                    condition_id: list(item.input_control_row_ids)
                    for condition_id, item in sorted(prediction.conditions.items())
                },
                "condition_truth_row_ids": {
                    condition_id: list(item.truth_row_ids)
                    for condition_id, item in sorted(evaluation.conditions.items())
                },
                "reconstruction": {
                    "controls": "canonical_h5ad[ordered_input_control_row_ids, frozen_genes]",
                    "truth": "canonical_h5ad[ordered_truth_row_ids, frozen_genes]",
                    "prediction": "load checkpoint and rerun frozen inference recipe",
                },
            },
        )
    return SealedEvaluationOutputs(
        prediction_manifest_path=prediction_manifest_path,
        evaluation_manifest_path=evaluation_manifest_path,
        result_pkl_path=(artifact_root / config.artifacts.result_pkl_name if keep_result else None),
    )


def seal_evaluated_run(
    *,
    destination: Path,
    config: ExperimentConfig,
    config_sha256: str,
    run_id: str,
    run_seed: int,
    source: SourceIdentity,
    environment: EnvironmentIdentity,
    training_data: CanonicalTrainingData,
    test_data: CanonicalEvaluationData,
    predictions: Sequence[PredictionConditionArrays],
    checkpoint_sha256: str | None,
) -> SealedEvaluatedRun:
    """Seal truth-free predictions, join frozen truth once, and emit small receipts."""

    small_root = destination / "small_results"
    outputs = seal_evaluation_outputs(
        destination=destination,
        config=config,
        config_sha256=config_sha256,
        run_id=run_id,
        run_seed=run_seed,
        source=source,
        environment=environment,
        training_data=training_data,
        test_data=test_data,
        predictions=predictions,
        checkpoint_sha256=checkpoint_sha256,
    )
    prediction_manifest_path = outputs.prediction_manifest_path
    evaluation_manifest_path = outputs.evaluation_manifest_path

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
        control_manifest_sha256=test_data.control_manifest_file_sha256,
        status="evaluated",
        best_checkpoint_sha256=checkpoint_sha256,
        test_evaluations=1,
    )
    _write_contract(small_root / "run_manifest.json", run_manifest)
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
    return SealedEvaluatedRun(
        run_manifest=run_manifest,
        prediction_manifest_path=prediction_manifest_path,
        evaluation_manifest_path=evaluation_manifest_path,
    )
