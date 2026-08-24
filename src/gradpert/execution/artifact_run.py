"""Shared condition-prediction to evaluated-run artifact lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def _write_contract(path: Path, contract: Any) -> None:
    atomic_json(path, contract.model_dump(mode="json"))


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
    predictions: list[PredictionConditionArrays],
    checkpoint_sha256: str | None,
) -> SealedEvaluatedRun:
    """Seal truth-free predictions, join frozen truth once, and emit small receipts."""

    small_root = destination / "small_results"
    prediction = seal_prediction_artifact(
        destination / "artifacts" / "prediction.pkl",
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
    evaluation = seal_frozen_evaluation_bundle(
        destination / "artifacts" / "evaluation.pkl",
        prediction=prediction,
        data=test_data,
        state=state,
    )
    evaluation_manifest_path = small_root / "evaluation_manifest.json"
    _write_contract(evaluation_manifest_path, evaluation.manifest)
    write_small_metric_exports(evaluation, small_root)

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
