"""Truth-free prediction PKL construction, sealing, and verified loading."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from gradpert.artifacts._io import (
    atomic_pickle,
    load_hash_pinned_pickle,
    sha256_file,
)
from gradpert.config.schema import DatasetId, ModelId
from gradpert.contracts import (
    ArrayReference,
    PredictionArtifactManifest,
    PredictionCondition,
)
from gradpert.hashing import sha256_json

PREDICTION_PKL_SCHEMA = "prediction-pkl-v1"
_CONDITION_KEYS = {
    "condition_id",
    "Pred",
    "InputCtrl",
    "InputCtrlRowIds",
    "gene_ids",
}


def sha256_array(value: np.ndarray[Any, Any]) -> str:
    """Hash dtype, shape, and C-order bytes rather than Python serialization."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(sha256_json(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PredictionProvenance:
    model_id: ModelId
    dataset_id: DatasetId
    protocol_id: str
    run_id: str
    run_seed: int
    source_commit: str
    source_dirty: bool
    formal_eligible: bool
    config_sha256: str
    environment_sha256: str
    canonical_data_sha256: str
    gene_order_sha256: str
    split_content_sha256: str
    control_manifest_sha256: str
    checkpoint_sha256: str | None


@dataclass(frozen=True)
class PredictionConditionArrays:
    condition_id: str
    prediction: np.ndarray[Any, Any]
    input_control: np.ndarray[Any, Any]
    input_control_row_ids: tuple[str, ...]


@dataclass(frozen=True)
class PredictionArtifact:
    manifest: PredictionArtifactManifest
    gene_ids: tuple[str, ...]
    conditions: Mapping[str, PredictionConditionArrays]
    file_sha256: str


def _validated_array(
    value: np.ndarray[Any, Any],
    *,
    gene_count: int,
    name: str,
) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if array.shape != (300, gene_count):
        raise ValueError(f"{name} must have shape (300, {gene_count})")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _validate_gene_ids(gene_ids: Sequence[str], expected_sha256: str) -> tuple[str, ...]:
    normalized = tuple(str(gene_id) for gene_id in gene_ids)
    if not normalized or any(not gene_id for gene_id in normalized):
        raise ValueError("gene_ids must be non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("gene_ids contain duplicates")
    if sha256_json(list(normalized)) != expected_sha256:
        raise ValueError("gene_ids do not match gene_order_sha256")
    return normalized


def _condition_payload(
    condition: PredictionConditionArrays,
    *,
    gene_ids: tuple[str, ...],
) -> dict[str, object]:
    if not condition.condition_id:
        raise ValueError("prediction condition ID must be non-empty")
    row_ids = tuple(str(row_id) for row_id in condition.input_control_row_ids)
    if len(row_ids) != 300 or any(not row_id for row_id in row_ids):
        raise ValueError("prediction condition requires exactly 300 control row IDs")
    prediction = _validated_array(
        condition.prediction,
        gene_count=len(gene_ids),
        name=f"Pred[{condition.condition_id}]",
    )
    input_control = _validated_array(
        condition.input_control,
        gene_count=len(gene_ids),
        name=f"InputCtrl[{condition.condition_id}]",
    )
    return {
        "condition_id": condition.condition_id,
        "Pred": prediction,
        "InputCtrl": input_control,
        "InputCtrlRowIds": row_ids,
        "gene_ids": gene_ids,
    }


def _manifest_condition(
    *,
    condition_id: str,
    payload: Mapping[str, object],
    artifact_path: Path,
) -> PredictionCondition:
    prediction = cast(np.ndarray[Any, Any], payload["Pred"])
    input_control = cast(np.ndarray[Any, Any], payload["InputCtrl"])
    row_ids = list(cast(tuple[str, ...], payload["InputCtrlRowIds"]))
    key_root = f"conditions/{condition_id}"
    return PredictionCondition(
        condition_id=condition_id,
        input_control_row_ids=row_ids,
        input_control_row_ids_sha256=sha256_json(row_ids),
        prediction=ArrayReference(
            artifact_path=str(artifact_path),
            key=f"{key_root}/Pred",
            dtype=str(prediction.dtype),
            shape=list(prediction.shape),
            content_sha256=sha256_array(prediction),
        ),
        input_control=ArrayReference(
            artifact_path=str(artifact_path),
            key=f"{key_root}/InputCtrl",
            dtype=str(input_control.dtype),
            shape=list(input_control.shape),
            content_sha256=sha256_array(input_control),
        ),
    )


def _assert_truth_absent(value: object, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized == "truth_included" and item is False:
                continue
            if "truth" in normalized or "ground_truth" in normalized:
                raise ValueError(
                    f"truth-like field is forbidden in PredictionArtifact: {path}.{key}"
                )
            _assert_truth_absent(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_truth_absent(item, f"{path}[{index}]")


def seal_prediction_artifact(
    path: str | Path,
    *,
    provenance: PredictionProvenance,
    gene_ids: Sequence[str],
    conditions: Sequence[PredictionConditionArrays],
) -> PredictionArtifact:
    """Atomically write and revalidate one truth-free condition-keyed PKL."""

    artifact_path = Path(path).resolve()
    normalized_genes = _validate_gene_ids(gene_ids, provenance.gene_order_sha256)
    if not conditions:
        raise ValueError("PredictionArtifact requires at least one condition")
    payload_by_condition: dict[str, dict[str, object]] = {}
    for condition in conditions:
        if condition.condition_id in payload_by_condition:
            raise ValueError(f"duplicate prediction condition: {condition.condition_id}")
        payload_by_condition[condition.condition_id] = _condition_payload(
            condition,
            gene_ids=normalized_genes,
        )
    ordered_ids = tuple(sorted(payload_by_condition))
    payload_by_condition = {
        condition_id: payload_by_condition[condition_id] for condition_id in ordered_ids
    }
    manifest = PredictionArtifactManifest(
        schema_version="prediction-artifact-v1",
        model_id=provenance.model_id,
        dataset_id=provenance.dataset_id,
        protocol_id=provenance.protocol_id,
        run_id=provenance.run_id,
        run_seed=provenance.run_seed,
        source_commit=provenance.source_commit,
        source_dirty=provenance.source_dirty,
        formal_eligible=provenance.formal_eligible,
        config_sha256=provenance.config_sha256,
        environment_sha256=provenance.environment_sha256,
        canonical_data_sha256=provenance.canonical_data_sha256,
        gene_order_sha256=provenance.gene_order_sha256,
        split_content_sha256=provenance.split_content_sha256,
        control_manifest_sha256=provenance.control_manifest_sha256,
        checkpoint_sha256=provenance.checkpoint_sha256,
        truth_included=False,
        prediction_kind="per_control_population",
        conditions=[
            _manifest_condition(
                condition_id=condition_id,
                payload=payload_by_condition[condition_id],
                artifact_path=artifact_path,
            )
            for condition_id in ordered_ids
        ],
    )
    package = {
        "schema_version": PREDICTION_PKL_SCHEMA,
        "manifest": manifest.model_dump(mode="json"),
        "gene_ids": normalized_genes,
        "conditions": payload_by_condition,
    }
    _assert_truth_absent(package)
    atomic_pickle(artifact_path, package)
    return load_prediction_artifact(
        artifact_path,
        expected_file_sha256=sha256_file(artifact_path),
        trusted_root=artifact_path.parent,
    )


def load_prediction_artifact(
    path: str | Path,
    *,
    expected_file_sha256: str,
    trusted_root: str | Path,
) -> PredictionArtifact:
    """Load only a hash-pinned PKL inside an explicit trusted artifact root."""

    artifact_path, package = load_hash_pinned_pickle(
        path,
        expected_file_sha256=expected_file_sha256,
        trusted_root=trusted_root,
    )
    observed_file_sha256 = sha256_file(artifact_path)
    if not isinstance(package, dict) or set(package) != {
        "schema_version",
        "manifest",
        "gene_ids",
        "conditions",
    }:
        raise ValueError("prediction PKL package shape is invalid")
    if package["schema_version"] != PREDICTION_PKL_SCHEMA:
        raise ValueError("prediction PKL schema version is unsupported")
    _assert_truth_absent(package)
    manifest = PredictionArtifactManifest.model_validate(package["manifest"])
    gene_ids = _validate_gene_ids(package["gene_ids"], manifest.gene_order_sha256)
    raw_conditions = package["conditions"]
    if not isinstance(raw_conditions, dict):
        raise ValueError("prediction PKL conditions must be a mapping")
    manifest_by_id = {item.condition_id: item for item in manifest.conditions}
    if set(raw_conditions) != set(manifest_by_id):
        raise ValueError("prediction PKL and manifest condition IDs differ")
    parsed: dict[str, PredictionConditionArrays] = {}
    for condition_id in sorted(raw_conditions):
        payload = raw_conditions[condition_id]
        if not isinstance(payload, dict) or set(payload) != _CONDITION_KEYS:
            raise ValueError(f"prediction condition payload is invalid: {condition_id}")
        if payload["condition_id"] != condition_id or tuple(payload["gene_ids"]) != gene_ids:
            raise ValueError(f"prediction condition identity/gene order mismatch: {condition_id}")
        prediction = _validated_array(
            payload["Pred"], gene_count=len(gene_ids), name=f"Pred[{condition_id}]"
        )
        input_control = _validated_array(
            payload["InputCtrl"],
            gene_count=len(gene_ids),
            name=f"InputCtrl[{condition_id}]",
        )
        row_ids = tuple(str(value) for value in payload["InputCtrlRowIds"])
        item = manifest_by_id[condition_id]
        if (
            len(row_ids) != 300
            or sha256_json(list(row_ids)) != item.input_control_row_ids_sha256
            or list(row_ids) != item.input_control_row_ids
        ):
            raise ValueError(f"prediction control row IDs/hash mismatch: {condition_id}")
        if sha256_array(prediction) != item.prediction.content_sha256:
            raise ValueError(f"prediction array hash mismatch: {condition_id}")
        if sha256_array(input_control) != item.input_control.content_sha256:
            raise ValueError(f"input-control array hash mismatch: {condition_id}")
        parsed[condition_id] = PredictionConditionArrays(
            condition_id=condition_id,
            prediction=prediction,
            input_control=input_control,
            input_control_row_ids=row_ids,
        )
    return PredictionArtifact(
        manifest=manifest,
        gene_ids=gene_ids,
        conditions=parsed,
        file_sha256=observed_file_sha256,
    )
