"""Receipt-backed canonical preparation for the five frozen datasets."""

from __future__ import annotations

import importlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gradpert.contracts import (
    CanonicalDataManifest,
    EvaluationControlManifest,
    SourceManifest,
    SplitManifest,
)
from gradpert.data._io import atomic_json, atomic_text, read_json
from gradpert.data.acquisition import (
    download_source,
    inspect_source_file,
    safe_extract_zip,
)
from gradpert.data.controls import build_evaluation_control_manifest
from gradpert.data.preprocessing import (
    preprocess_norman,
    preprocess_raw_within_cell,
    preprocess_upstream_within_cell,
)
from gradpert.data.schema import DatasetRegistryEntry
from gradpert.data.split import (
    apply_benchmark_condition_policy,
    build_grouped_split_manifest,
    build_norman_combo_seen2_split_manifest,
)
from gradpert.hashing import sha256_file, sha256_json


@dataclass(frozen=True)
class DatasetLayout:
    data_root: Path
    dataset_id: str
    protocol_id: str

    @property
    def root(self) -> Path:
        return self.data_root / self.dataset_id / self.protocol_id

    @property
    def source(self) -> Path:
        return self.root / "source"

    @property
    def extracted(self) -> Path:
        return self.root / "extracted"

    @property
    def canonical(self) -> Path:
        return self.root / "canonical"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def canonical_adata(self) -> Path:
        return self.canonical / "adata.h5ad"

    @property
    def canonical_manifest(self) -> Path:
        return self.manifests / "canonical.json"


@dataclass(frozen=True)
class DatasetPreparationResult:
    dataset_id: str
    protocol_id: str
    state: Literal["canonical_ready"]
    canonical_manifest_path: str
    canonical_manifest_sha256: str
    canonical_adata_sha256: str
    split_content_sha256: str
    evaluation_controls_sha256: str
    n_cells: int
    n_expression_genes: int
    n_graph_genes: int
    n_conditions: int
    n_controls: int


def _require_anndata() -> Any:
    try:
        ad = importlib.import_module("anndata")
    except ImportError as error:  # pragma: no cover - server data environment
        raise RuntimeError("anndata is required for canonical dataset preparation") from error
    return ad


def _matrix_values(matrix: Any) -> np.ndarray[Any, Any]:
    values = matrix.data if hasattr(matrix, "data") else np.asarray(matrix).reshape(-1)
    return np.asarray(values)


def _source_h5ad(entry: DatasetRegistryEntry, layout: DatasetLayout) -> Path:
    source_path = layout.source / entry.source.filename
    if entry.source.semantics == "raw_single_cell":
        return source_path
    member = entry.source.archive_h5ad_member
    if member is None:  # pragma: no cover - closed by registry validation
        raise AssertionError("processed archive member is missing")
    target = layout.extracted.joinpath(*member.split("/"))
    if target.is_file():
        return target
    if layout.extracted.exists():
        raise RuntimeError(
            "incomplete extraction exists without frozen member; inspect manually: "
            f"{layout.extracted}"
        )
    temporary = layout.root / "extracted.preparing"
    if temporary.exists():
        raise RuntimeError(f"stale extraction staging directory requires audit: {temporary}")
    safe_extract_zip(source_path, temporary)
    staged_target = temporary.joinpath(*member.split("/"))
    if not staged_target.is_file():
        raise ValueError(f"frozen archive member was not extracted: {member}")
    os.replace(temporary, layout.extracted)
    return target


def _source_manifest(entry: DatasetRegistryEntry, layout: DatasetLayout) -> SourceManifest:
    source_path = layout.source / entry.source.filename
    return SourceManifest(
        schema_version="dataset-source-v1",
        dataset_id=entry.dataset_id,
        protocol_id=entry.protocol_id,
        registry_version="datasets-v2",
        source_url=entry.source.url,
        filename=entry.source.filename,
        source_sha256=sha256_file(source_path, chunk_size=8 * 1024 * 1024),
        size_bytes=source_path.stat().st_size,
        license_id=entry.source.license_id,
        source_semantics=(
            "raw" if entry.source.semantics == "raw_single_cell" else "upstream_processed"
        ),
    )


def _known_raw_targets(adata: Any, entry: DatasetRegistryEntry) -> set[str]:
    column = entry.source_metadata.condition_column
    control = entry.source_metadata.control_identifier
    if column is None or control is None:  # pragma: no cover - registry validation
        raise AssertionError("verified raw target mapping is missing")
    values = adata.obs[column]
    if bool(values.isna().any()):
        raise ValueError("raw perturbation target column contains null values")
    targets = {str(value) for value in values.unique() if str(value) != control}
    if any("+" in target for target in targets):
        raise ValueError("within-cell raw target IDs must be single perturbations")
    return targets


def _preprocess(adata: Any, entry: DatasetRegistryEntry) -> tuple[Any, dict[str, Any]]:
    if entry.dataset_id == "replogle_k562_essential":
        prepared, upstream_report = preprocess_upstream_within_cell(adata, entry)
        return prepared, asdict(upstream_report)
    elif entry.dataset_id == "norman":
        prepared, norman_report = preprocess_norman(adata, entry)
        return prepared, asdict(norman_report)
    prepared, raw_report = preprocess_raw_within_cell(
        adata,
        entry,
        known_candidate_targets=_known_raw_targets(adata, entry),
    )
    return prepared, asdict(raw_report)


def _gene_axes(adata: Any, entry: DatasetRegistryEntry) -> tuple[list[str], list[str]]:
    column = entry.canonical_metadata.gene_symbol_column
    if column not in adata.var:
        raise ValueError("canonical data lacks gene_name")
    genes = [str(value) for value in adata.var[column].tolist()]
    if len(genes) != len(set(genes)) or any(not gene for gene in genes):
        raise ValueError("canonical gene names must be unique and non-empty")
    expression_mask = np.asarray(adata.var["expression_output_gene"], dtype=bool)
    expression_positions = np.flatnonzero(expression_mask)
    if not np.array_equal(expression_positions, np.arange(len(expression_positions))):
        raise ValueError("expression genes must be the leading canonical graph-axis prefix")
    expression = [genes[int(index)] for index in expression_positions]
    return expression, genes


def _canonical_qc(
    adata: Any,
    entry: DatasetRegistryEntry,
    *,
    expression_genes: list[str],
    graph_genes: list[str],
) -> dict[str, Any]:
    canonical = entry.canonical_metadata
    required_obs = (
        canonical.condition_column,
        canonical.batch_column,
        canonical.cell_type_column,
        canonical.control_column,
        canonical.condition_name_column,
    )
    missing = [column for column in required_obs if column not in adata.obs]
    if missing:
        raise ValueError(f"canonical obs columns are missing: {missing}")
    if not bool(adata.obs_names.is_unique) or not bool(adata.var_names.is_unique):
        raise ValueError("canonical observation and var indexes must be unique")
    row_ids = [str(value) for value in adata.obs_names]
    if any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
        raise ValueError("canonical observation IDs must be unique non-empty strings")
    values = _matrix_values(adata.X)
    if values.size and (not bool(np.isfinite(values).all()) or float(values.min()) < 0):
        raise ValueError("canonical expression contains negative or non-finite values")
    conditions = [str(value) for value in adata.obs[canonical.condition_column]]
    batches = [str(value) for value in adata.obs[canonical.batch_column]]
    cell_types = [str(value) for value in adata.obs[canonical.cell_type_column]]
    if {value.casefold() for value in cell_types} != {entry.cell_context.casefold()}:
        raise ValueError("canonical cell context differs from registry identity")
    if any("::" in value for value in [*batches, *cell_types]):
        raise ValueError("canonical cell type/batch values cannot contain the context delimiter")
    controls = np.asarray(adata.obs[canonical.control_column], dtype=bool)
    if not bool(controls.any()) or not bool((np.asarray(conditions) == "ctrl").any()):
        raise ValueError("canonical controls are empty")
    if not np.array_equal(controls, np.asarray(conditions) == "ctrl"):
        raise ValueError("canonical control flag and condition disagree")
    condition_counts = {
        condition: int(count)
        for condition, count in zip(*np.unique(conditions, return_counts=True), strict=True)
    }
    if any(count <= 0 for count in condition_counts.values()):  # pragma: no cover
        raise AssertionError("condition count cannot be non-positive")
    return {
        "schema_version": "canonical-qc-v1",
        "dataset_id": entry.dataset_id,
        "protocol_id": entry.protocol_id,
        "state": "qc_passed",
        "n_cells": int(adata.n_obs),
        "n_expression_genes": len(expression_genes),
        "n_graph_genes": len(graph_genes),
        "n_conditions": len(condition_counts),
        "n_controls": int(controls.sum()),
        "condition_counts": condition_counts,
        "cell_contexts": sorted(set(cell_types)),
        "batch_count": len(set(batches)),
        "matrix_nonzero_value_min": float(values.min()) if values.size else 0.0,
        "matrix_nonzero_value_max": float(values.max()) if values.size else 0.0,
        "observation_order_sha256": sha256_json(row_ids),
        "expression_gene_order_sha256": sha256_json(expression_genes),
        "graph_gene_order_sha256": sha256_json(graph_genes),
    }


def _split(adata: Any, entry: DatasetRegistryEntry) -> SplitManifest:
    values = [str(value) for value in adata.obs[entry.canonical_metadata.condition_column]]
    conditions = list(dict.fromkeys(values))
    if entry.dataset_id == "norman":
        source_split = build_norman_combo_seen2_split_manifest(
            conditions=conditions,
            split_seed=entry.split_seed,
        )
    else:
        source_split = build_grouped_split_manifest(
            dataset_id=entry.dataset_id,
            protocol_id=entry.protocol_id,
            conditions=conditions,
            control_condition_id=entry.control_condition_id,
            split_seed=entry.split_seed,
        )
    policy = entry.benchmark_condition_policy
    return apply_benchmark_condition_policy(
        source_split,
        policy_id=policy.policy_id,
        excluded_conditions=policy.excluded_conditions,
    )


def _evaluation_controls(
    adata: Any,
    entry: DatasetRegistryEntry,
    split: SplitManifest,
    split_name: Literal["val", "test"],
) -> EvaluationControlManifest:
    metadata = entry.canonical_metadata
    row_ids = np.asarray([str(value) for value in adata.obs_names], dtype=str)
    conditions = np.asarray(adata.obs[metadata.condition_column].astype(str), dtype=str)
    cell_types = np.asarray(adata.obs[metadata.cell_type_column].astype(str), dtype=str)
    batches = np.asarray(adata.obs[metadata.batch_column].astype(str), dtype=str)
    contexts = np.char.add(np.char.add(cell_types, "::"), batches)
    control_mask = conditions == entry.control_condition_id
    control_by_context: dict[str, list[str]] = {}
    for context in sorted(set(contexts[control_mask].tolist())):
        control_by_context[context] = row_ids[control_mask & (contexts == context)].tolist()
    condition_ids = split.val_conditions if split_name == "val" else split.test_conditions
    pools: dict[str, dict[str, list[str]]] = {}
    truth_contexts: dict[str, list[str]] = {}
    for condition in condition_ids:
        condition_contexts = contexts[conditions == condition].tolist()
        if not condition_contexts:
            raise ValueError(f"split condition has no canonical truth cells: {condition}")
        missing = sorted(set(condition_contexts) - set(control_by_context))
        if missing:
            raise ValueError(f"condition has contexts without controls: {condition}: {missing}")
        truth_contexts[condition] = condition_contexts
        pools[condition] = {
            context: control_by_context[context] for context in sorted(set(condition_contexts))
        }
    return build_evaluation_control_manifest(
        dataset_id=entry.dataset_id,
        protocol_id=entry.protocol_id,
        split_name=split_name,
        split_manifest=split,
        control_pools=pools,
        truth_context_ids=truth_contexts,
    )


def _write_model(path: Path, model: Any) -> None:
    atomic_json(path, model.model_dump(mode="json"))


def _write_h5ad_atomic(adata: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.preparing")
    if temporary.exists():
        raise RuntimeError(f"stale canonical H5AD staging file requires audit: {temporary}")
    try:
        adata.write_h5ad(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_result(layout: DatasetLayout) -> DatasetPreparationResult | None:
    if not layout.canonical_manifest.is_file():
        return None
    manifest = CanonicalDataManifest.model_validate(read_json(layout.canonical_manifest))
    if manifest.dataset_id != layout.dataset_id or manifest.protocol_id != layout.protocol_id:
        raise ValueError("canonical manifest identity differs from its directory")
    if sha256_file(layout.canonical_adata) != manifest.canonical_adata_sha256:
        raise ValueError("canonical H5AD hash no longer matches its readiness manifest")
    return DatasetPreparationResult(
        dataset_id=manifest.dataset_id,
        protocol_id=manifest.protocol_id,
        state=manifest.state,
        canonical_manifest_path=str(layout.canonical_manifest),
        canonical_manifest_sha256=sha256_file(layout.canonical_manifest),
        canonical_adata_sha256=manifest.canonical_adata_sha256,
        split_content_sha256=manifest.split_content_sha256,
        evaluation_controls_sha256=manifest.evaluation_controls_sha256,
        n_cells=manifest.n_cells,
        n_expression_genes=manifest.n_expression_genes,
        n_graph_genes=manifest.n_graph_genes,
        n_conditions=manifest.n_conditions,
        n_controls=manifest.n_controls,
    )


def prepare_dataset(
    entry: DatasetRegistryEntry,
    data_root: str | Path,
    *,
    allow_download: bool = False,
) -> DatasetPreparationResult:
    """Materialize one canonical-ready dataset or verify its existing receipt."""

    layout = DatasetLayout(Path(data_root), entry.dataset_id, entry.protocol_id)
    existing = _existing_result(layout)
    if existing is not None:
        return existing
    layout.source.mkdir(parents=True, exist_ok=True)
    status = inspect_source_file(entry, layout.source)
    if status.state != "ready" and allow_download:
        status = download_source(entry, layout.source)
    if status.state != "ready":
        raise RuntimeError(
            f"source is not ready for {entry.dataset_id}: {status.state} "
            f"({status.observed_size_bytes}/{status.expected_size_bytes})"
        )

    source_manifest = _source_manifest(entry, layout)
    source_manifest_path = layout.manifests / "source.json"
    _write_model(source_manifest_path, source_manifest)
    source_h5ad = _source_h5ad(entry, layout)
    ad = _require_anndata()
    upstream = ad.read_h5ad(source_h5ad)
    prepared, preprocessing_report = _preprocess(upstream, entry)
    expression_genes, graph_genes = _gene_axes(prepared, entry)
    qc = _canonical_qc(
        prepared,
        entry,
        expression_genes=expression_genes,
        graph_genes=graph_genes,
    )
    split = _split(prepared, entry)
    val_controls = _evaluation_controls(prepared, entry, split, "val")
    test_controls = _evaluation_controls(prepared, entry, split, "test")

    preprocessing_payload = {
        "schema_version": "preprocessing-manifest-v1",
        "dataset_id": entry.dataset_id,
        "protocol_id": entry.protocol_id,
        "profile_id": entry.preprocessing.profile_id,
        "report": preprocessing_report,
    }
    preprocessing_path = layout.manifests / "preprocessing.json"
    qc_path = layout.manifests / "qc.json"
    split_path = layout.manifests / "split.json"
    val_controls_path = layout.manifests / "evaluation_controls.val.json"
    test_controls_path = layout.manifests / "evaluation_controls.test.json"
    atomic_json(preprocessing_path, preprocessing_payload)
    atomic_json(qc_path, qc)
    _write_model(split_path, split)
    _write_model(val_controls_path, val_controls)
    _write_model(test_controls_path, test_controls)
    atomic_text(layout.canonical / "expression_gene_ids.txt", "\n".join(expression_genes) + "\n")
    atomic_text(layout.canonical / "graph_gene_ids.txt", "\n".join(graph_genes) + "\n")
    _write_h5ad_atomic(prepared, layout.canonical_adata)

    control_hashes = {
        "val": sha256_file(val_controls_path),
        "test": sha256_file(test_controls_path),
    }
    canonical_manifest = CanonicalDataManifest(
        schema_version="canonical-data-v1",
        dataset_id=entry.dataset_id,
        protocol_id=entry.protocol_id,
        state="canonical_ready",
        canonical_adata_path=str(
            Path("data") / entry.dataset_id / entry.protocol_id / "canonical" / "adata.h5ad"
        ),
        canonical_adata_sha256=sha256_file(layout.canonical_adata, chunk_size=8 * 1024 * 1024),
        source_manifest_sha256=sha256_file(source_manifest_path),
        preprocessing_manifest_sha256=sha256_file(preprocessing_path),
        qc_manifest_sha256=sha256_file(qc_path),
        split_manifest_sha256=sha256_file(split_path),
        split_content_sha256=split.split_content_sha256,
        evaluation_controls_sha256=sha256_json(control_hashes),
        expression_gene_order_sha256=qc["expression_gene_order_sha256"],
        graph_gene_order_sha256=qc["graph_gene_order_sha256"],
        observation_order_sha256=qc["observation_order_sha256"],
        n_cells=qc["n_cells"],
        n_expression_genes=qc["n_expression_genes"],
        n_graph_genes=qc["n_graph_genes"],
        n_conditions=qc["n_conditions"],
        n_controls=qc["n_controls"],
    )
    _write_model(layout.canonical_manifest, canonical_manifest)
    receipt_paths = [
        source_manifest_path,
        preprocessing_path,
        qc_path,
        split_path,
        val_controls_path,
        test_controls_path,
        layout.canonical / "expression_gene_ids.txt",
        layout.canonical / "graph_gene_ids.txt",
        layout.canonical_adata,
        layout.canonical_manifest,
    ]
    checksum_lines = [
        f"{sha256_file(path, chunk_size=8 * 1024 * 1024)}  {path.relative_to(layout.root)}"
        for path in sorted(receipt_paths)
    ]
    atomic_text(layout.manifests / "checksums.sha256", "\n".join(checksum_lines) + "\n")
    result = _existing_result(layout)
    if result is None:  # pragma: no cover - just wrote the manifest
        raise AssertionError("canonical readiness manifest was not created")
    return result


def refresh_dataset_protocol(
    entry: DatasetRegistryEntry,
    data_root: str | Path,
) -> DatasetPreparationResult:
    """Refresh only split/control receipts after a preregistered policy change."""

    layout = DatasetLayout(Path(data_root), entry.dataset_id, entry.protocol_id)
    existing = _existing_result(layout)
    if existing is None:
        raise RuntimeError(f"canonical readiness manifest is missing for {entry.dataset_id}")
    manifest = CanonicalDataManifest.model_validate(read_json(layout.canonical_manifest))
    current_split_path = layout.manifests / "split.json"
    current_split = SplitManifest.model_validate(read_json(current_split_path))
    if (
        sha256_file(current_split_path) != manifest.split_manifest_sha256
        or current_split.split_content_sha256 != manifest.split_content_sha256
    ):
        raise ValueError("existing split receipt is not linked to the canonical manifest")

    ad = _require_anndata()
    backed = ad.read_h5ad(layout.canonical_adata, backed="r")
    try:
        refreshed_split = _split(backed, entry)
        val_controls = _evaluation_controls(backed, entry, refreshed_split, "val")
        test_controls = _evaluation_controls(backed, entry, refreshed_split, "test")
    finally:
        backed.file.close()

    val_controls_path = layout.manifests / "evaluation_controls.val.json"
    test_controls_path = layout.manifests / "evaluation_controls.test.json"
    source_manifest_path = layout.manifests / "source.json"
    _write_model(source_manifest_path, _source_manifest(entry, layout))
    _write_model(current_split_path, refreshed_split)
    _write_model(val_controls_path, val_controls)
    _write_model(test_controls_path, test_controls)
    control_hashes = {
        "val": sha256_file(val_controls_path),
        "test": sha256_file(test_controls_path),
    }
    refreshed_manifest = manifest.model_copy(
        update={
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "split_manifest_sha256": sha256_file(current_split_path),
            "split_content_sha256": refreshed_split.split_content_sha256,
            "evaluation_controls_sha256": sha256_json(control_hashes),
        }
    )
    _write_model(layout.canonical_manifest, refreshed_manifest)
    receipt_paths = [
        layout.manifests / "source.json",
        layout.manifests / "preprocessing.json",
        layout.manifests / "qc.json",
        current_split_path,
        val_controls_path,
        test_controls_path,
        layout.canonical / "expression_gene_ids.txt",
        layout.canonical / "graph_gene_ids.txt",
        layout.canonical_adata,
        layout.canonical_manifest,
    ]
    checksum_lines = [
        f"{sha256_file(path, chunk_size=8 * 1024 * 1024)}  {path.relative_to(layout.root)}"
        for path in sorted(receipt_paths)
    ]
    atomic_text(layout.manifests / "checksums.sha256", "\n".join(checksum_lines) + "\n")
    result = _existing_result(layout)
    if result is None:  # pragma: no cover - canonical manifest remains present
        raise AssertionError("refreshed canonical readiness manifest is unavailable")
    return result


def dataset_status(entry: DatasetRegistryEntry, data_root: str | Path) -> dict[str, Any]:
    layout = DatasetLayout(Path(data_root), entry.dataset_id, entry.protocol_id)
    source = inspect_source_file(entry, layout.source)
    result: dict[str, Any] = {
        "dataset_id": entry.dataset_id,
        "protocol_id": entry.protocol_id,
        "source_state": source.state,
        "source_path": str(source.path),
        "source_observed_size_bytes": source.observed_size_bytes,
        "source_expected_size_bytes": source.expected_size_bytes,
        "canonical_state": "missing",
    }
    if layout.canonical_manifest.is_file():
        ready = _existing_result(layout)
        if ready is None:  # pragma: no cover
            raise AssertionError("manifest disappeared while checking status")
        result.update(asdict(ready))
        result["canonical_state"] = ready.state
    return result


def verify_prepared_dataset(
    entry: DatasetRegistryEntry,
    data_root: str | Path,
) -> DatasetPreparationResult:
    """Recompute every small receipt link plus source and canonical H5AD hashes."""

    layout = DatasetLayout(Path(data_root), entry.dataset_id, entry.protocol_id)
    result = _existing_result(layout)
    if result is None:
        raise RuntimeError(f"canonical readiness manifest is missing for {entry.dataset_id}")
    manifest = CanonicalDataManifest.model_validate(read_json(layout.canonical_manifest))
    source_status = inspect_source_file(entry, layout.source)
    if source_status.state != "ready":
        raise ValueError(f"frozen source no longer verifies: {source_status.state}")

    paths = {
        "source": layout.manifests / "source.json",
        "preprocessing": layout.manifests / "preprocessing.json",
        "qc": layout.manifests / "qc.json",
        "split": layout.manifests / "split.json",
        "val_controls": layout.manifests / "evaluation_controls.val.json",
        "test_controls": layout.manifests / "evaluation_controls.test.json",
    }
    expected_hashes = {
        "source": manifest.source_manifest_sha256,
        "preprocessing": manifest.preprocessing_manifest_sha256,
        "qc": manifest.qc_manifest_sha256,
        "split": manifest.split_manifest_sha256,
    }
    for name, expected in expected_hashes.items():
        observed = sha256_file(paths[name])
        if observed != expected:
            raise ValueError(f"{name} manifest hash mismatch")
    source_manifest = SourceManifest.model_validate(read_json(paths["source"]))
    source_path = layout.source / entry.source.filename
    if source_manifest.source_sha256 != sha256_file(source_path, chunk_size=8 * 1024 * 1024):
        raise ValueError("source SHA-256 differs from source receipt")

    split = SplitManifest.model_validate(read_json(paths["split"]))
    if split.split_content_sha256 != manifest.split_content_sha256:
        raise ValueError("canonical and split manifests disagree on split content")
    val_controls = EvaluationControlManifest.model_validate(read_json(paths["val_controls"]))
    test_controls = EvaluationControlManifest.model_validate(read_json(paths["test_controls"]))
    if [draw.condition_id for draw in val_controls.draws] != split.val_conditions:
        raise ValueError("validation control draws differ from split condition order")
    if [draw.condition_id for draw in test_controls.draws] != split.test_conditions:
        raise ValueError("test control draws differ from split condition order")
    observed_control_hash = sha256_json(
        {
            "val": sha256_file(paths["val_controls"]),
            "test": sha256_file(paths["test_controls"]),
        }
    )
    if observed_control_hash != manifest.evaluation_controls_sha256:
        raise ValueError("evaluation control receipt hash mismatch")

    expression_path = layout.canonical / "expression_gene_ids.txt"
    graph_path = layout.canonical / "graph_gene_ids.txt"
    expression_genes = expression_path.read_text(encoding="utf-8").splitlines()
    graph_genes = graph_path.read_text(encoding="utf-8").splitlines()
    if sha256_json(expression_genes) != manifest.expression_gene_order_sha256:
        raise ValueError("expression gene order hash mismatch")
    if sha256_json(graph_genes) != manifest.graph_gene_order_sha256:
        raise ValueError("graph gene order hash mismatch")
    if graph_genes[: len(expression_genes)] != expression_genes:
        raise ValueError("expression genes are no longer the graph-axis prefix")

    ad = _require_anndata()
    backed = ad.read_h5ad(layout.canonical_adata, backed="r")
    try:
        expected_split = _split(backed, entry)
        if split != expected_split:
            raise ValueError("prepared split differs from benchmark condition policy")
        observed_rows = [str(value) for value in backed.obs_names]
        observed_genes = [
            str(value) for value in backed.var[entry.canonical_metadata.gene_symbol_column]
        ]
        if sha256_json(observed_rows) != manifest.observation_order_sha256:
            raise ValueError("canonical observation order hash mismatch")
        if observed_genes != graph_genes:
            raise ValueError("canonical H5AD gene order differs from graph gene receipt")
        if tuple(backed.shape) != (manifest.n_cells, manifest.n_graph_genes):
            raise ValueError("canonical H5AD shape differs from canonical manifest")
    finally:
        backed.file.close()
    return result
