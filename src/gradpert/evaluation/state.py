"""Frozen dataset-level DE masks and reference arrays for common evaluation."""

from __future__ import annotations

import hashlib
import importlib
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gradpert.contracts import (
    CanonicalDataManifest,
    EvaluationStateManifest,
    SplitManifest,
)
from gradpert.data import DatasetLayout
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file, sha256_json

DE_SOURCE_COMMIT = "87ac2c51c3c266391093f71a8bce2e6beaa81518"


@dataclass(frozen=True)
class EvaluationStateLayout:
    data_root: Path
    dataset_id: str
    protocol_id: str

    @property
    def dataset(self) -> DatasetLayout:
        return DatasetLayout(self.data_root, self.dataset_id, self.protocol_id)

    @property
    def root(self) -> Path:
        return self.dataset.root / "evaluation"

    @property
    def manifest(self) -> Path:
        return self.root / "state_manifest.json"

    @property
    def arrays(self) -> Path:
        return self.root / "state_arrays.npz"


@dataclass(frozen=True)
class LoadedEvaluationState:
    manifest: EvaluationStateManifest
    systema_reference: np.ndarray[Any, Any]
    metric_control_means: np.ndarray[Any, Any]
    manifest_file_sha256: str


def _sha256_array(value: np.ndarray[Any, Any]) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(sha256_json(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_arrays(
    path: Path,
    *,
    systema_reference: np.ndarray[Any, Any],
    metric_control_means: np.ndarray[Any, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                systema_reference=systema_reference,
                metric_control_means=metric_control_means,
            )
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _mean_rows(adata: Any, indices: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if indices.size == 0:
        raise ValueError("evaluation-state population is empty")
    matrix = adata.X[indices]
    return np.asarray(matrix.mean(axis=0), dtype=np.float32).reshape(-1)


def _condition_targets(condition_id: str, control_id: str) -> tuple[str, ...]:
    targets = tuple(part for part in condition_id.split("+") if part != control_id)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError(f"invalid evaluation condition: {condition_id}")
    return targets


def prepare_evaluation_state(
    *,
    dataset_id: str,
    protocol_id: str,
    data_root: str | Path,
) -> EvaluationStateManifest:
    """Materialize one model-independent evaluation state from canonical data."""

    layout = EvaluationStateLayout(Path(data_root), dataset_id, protocol_id)
    if layout.manifest.is_file():
        return load_evaluation_state(
            dataset_id=dataset_id,
            protocol_id=protocol_id,
            data_root=data_root,
        ).manifest
    canonical = CanonicalDataManifest.model_validate(
        read_json(layout.dataset.manifests / "canonical.json")
    )
    split = SplitManifest.model_validate(read_json(layout.dataset.manifests / "split.json"))
    if (canonical.dataset_id, canonical.protocol_id) != (dataset_id, protocol_id):
        raise ValueError("canonical identity differs from evaluation-state request")
    if (split.dataset_id, split.protocol_id) != (dataset_id, protocol_id):
        raise ValueError("split identity differs from evaluation-state request")
    if canonical.split_content_sha256 != split.split_content_sha256:
        raise ValueError("canonical and split hashes differ before evaluation-state build")
    expression_genes = tuple(
        (layout.dataset.canonical / "expression_gene_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if (
        len(expression_genes) != canonical.n_expression_genes
        or sha256_json(list(expression_genes)) != canonical.expression_gene_order_sha256
    ):
        raise ValueError("expression genes differ before evaluation-state build")

    try:
        ad = importlib.import_module("anndata")
        pd = importlib.import_module("pandas")
        sc = importlib.import_module("scanpy")
    except ImportError as error:  # pragma: no cover - server data environment
        raise RuntimeError("anndata and scanpy are required for evaluation-state build") from error
    backed = ad.read_h5ad(layout.dataset.canonical_adata, backed="r")
    try:
        adata = backed[:, : canonical.n_expression_genes].to_memory()
    finally:
        backed.file.close()
    adata.var_names = list(expression_genes)
    conditions = np.asarray(adata.obs["condition"], dtype=str)
    contexts = np.asarray(
        [
            f"{cell_type}::{batch}"
            for cell_type, batch in zip(
                adata.obs["cell_type"],
                adata.obs["batch"],
                strict=True,
            )
        ],
        dtype=str,
    )
    control_mask = conditions == split.control_condition_id
    if not bool(control_mask.any()):
        raise ValueError("evaluation-state build has no controls")
    evaluation_conditions = [*split.val_conditions, *split.test_conditions]
    evaluation_counts = {
        condition: int(np.count_nonzero(conditions == condition))
        for condition in evaluation_conditions
    }
    rankable_conditions = {
        condition for condition, count in evaluation_counts.items() if count >= 2
    }
    ranked_by_condition: dict[str, list[str]] = {}
    if rankable_conditions:
        rank_mask = control_mask | np.isin(conditions, list(rankable_conditions))
        rank_adata = adata[rank_mask].copy()
        rank_adata.obs["condition"] = rank_adata.obs["condition"].astype(str).astype("category")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            sc.tl.rank_genes_groups(
                rank_adata,
                groupby="condition",
                reference=split.control_condition_id,
                rankby_abs=True,
                n_genes=canonical.n_expression_genes,
                method="t-test",
            )
        names = rank_adata.uns["rank_genes_groups"]["names"]
        ranked_by_condition = {
            str(condition): [str(value) for value in names[condition].tolist()]
            for condition in names.dtype.names or ()
        }
    gene_index = {gene_id: index for index, gene_id in enumerate(expression_genes)}
    control_mean = _mean_rows(adata, np.flatnonzero(control_mask))
    de_by_condition: dict[str, list[int]] = {}
    de_unavailable_reasons: dict[str, str] = {}
    for condition in evaluation_conditions:
        condition_indices = np.flatnonzero(conditions == condition)
        if evaluation_counts[condition] < 2:
            de_by_condition[condition] = []
            de_unavailable_reasons[condition] = (
                f"insufficient_truth_cells_for_t_test:n={evaluation_counts[condition]}"
            )
            continue
        perturbed_mean = _mean_rows(adata, condition_indices)
        non_dropout = (perturbed_mean != 0) | ((perturbed_mean == 0) & (control_mean == 0))
        ranked = ranked_by_condition.get(condition)
        if ranked is None:
            raise ValueError(f"Scanpy DE output lacks condition: {condition}")
        eligible = [
            gene_index[gene_id]
            for gene_id in ranked
            if gene_id in gene_index and non_dropout[gene_index[gene_id]]
        ][:20]
        target_indices = {
            gene_index[target]
            for target in _condition_targets(condition, split.control_condition_id)
            if target in gene_index
        }
        final = sorted(set(eligible) - target_indices)
        if not final:
            raise ValueError(f"condition has no DE genes after target exclusion: {condition}")
        de_by_condition[condition] = final

    reference_conditions = [*split.train_conditions, *split.val_conditions]
    reference_sum = np.zeros(canonical.n_expression_genes, dtype=np.float64)
    for condition in reference_conditions:
        reference_sum += _mean_rows(adata, np.flatnonzero(conditions == condition))
    systema_reference = np.asarray(
        reference_sum / len(reference_conditions),
        dtype=np.float32,
    )

    control_indices_by_context = {
        context: np.flatnonzero(control_mask & (contexts == context))
        for context in sorted(set(contexts[control_mask].tolist()))
    }
    control_mean_cache: dict[tuple[str, ...], np.ndarray[Any, Any]] = {}
    metric_control_rows: list[np.ndarray[Any, Any]] = []
    for condition in evaluation_conditions:
        truth_contexts = tuple(sorted(set(contexts[conditions == condition].tolist())))
        if truth_contexts not in control_mean_cache:
            missing = [
                context for context in truth_contexts if context not in control_indices_by_context
            ]
            if missing:
                raise ValueError(f"evaluation contexts lack controls: {condition}: {missing}")
            pool = np.concatenate(
                [control_indices_by_context[context] for context in truth_contexts]
            )
            control_mean_cache[truth_contexts] = _mean_rows(adata, pool)
        metric_control_rows.append(control_mean_cache[truth_contexts])
    metric_control_means = np.asarray(metric_control_rows, dtype=np.float32)
    _atomic_arrays(
        layout.arrays,
        systema_reference=systema_reference,
        metric_control_means=metric_control_means,
    )
    conventional_arrays_path = str(
        Path("data") / dataset_id / protocol_id / "evaluation" / "state_arrays.npz"
    )
    manifest = EvaluationStateManifest(
        schema_version="evaluation-state-v1",
        dataset_id=canonical.dataset_id,
        protocol_id=canonical.protocol_id,
        canonical_data_sha256=canonical.canonical_adata_sha256,
        split_content_sha256=split.split_content_sha256,
        expression_gene_order_sha256=canonical.expression_gene_order_sha256,
        condition_ids=evaluation_conditions,
        condition_ids_sha256=sha256_json(evaluation_conditions),
        de_gene_indices=de_by_condition,
        de_gene_indices_sha256=sha256_json(de_by_condition),
        top_de_gene_indices=de_by_condition,
        top_de_gene_indices_sha256=sha256_json(de_by_condition),
        de_unavailable_reasons=de_unavailable_reasons,
        de_unavailable_reasons_sha256=sha256_json(de_unavailable_reasons),
        de_method="scanpy_t_test_rankby_abs_non_dropout_top20_exclude_targets",
        de_reference="ctrl",
        de_source_commit=DE_SOURCE_COMMIT,
        systema_reference_condition_ids=reference_conditions,
        systema_reference_condition_ids_sha256=sha256_json(reference_conditions),
        arrays_path=conventional_arrays_path,
        arrays_sha256=sha256_file(layout.arrays),
        systema_reference_content_sha256=_sha256_array(systema_reference),
        metric_control_means_content_sha256=_sha256_array(metric_control_means),
    )
    atomic_json(layout.manifest, manifest.model_dump(mode="json"))
    return load_evaluation_state(
        dataset_id=dataset_id,
        protocol_id=protocol_id,
        data_root=data_root,
    ).manifest


def load_evaluation_state(
    *,
    dataset_id: str,
    protocol_id: str,
    data_root: str | Path,
) -> LoadedEvaluationState:
    """Verify and load an existing frozen evaluation state."""

    layout = EvaluationStateLayout(Path(data_root), dataset_id, protocol_id)
    manifest = EvaluationStateManifest.model_validate(read_json(layout.manifest))
    canonical = CanonicalDataManifest.model_validate(
        read_json(layout.dataset.manifests / "canonical.json")
    )
    split = SplitManifest.model_validate(read_json(layout.dataset.manifests / "split.json"))
    if (manifest.dataset_id, manifest.protocol_id) != (dataset_id, protocol_id):
        raise ValueError("evaluation-state manifest identity differs")
    if manifest.canonical_data_sha256 != canonical.canonical_adata_sha256:
        raise ValueError("evaluation state and canonical data hashes differ")
    if manifest.split_content_sha256 != split.split_content_sha256:
        raise ValueError("evaluation state and split hashes differ")
    expected_conditions = [*split.val_conditions, *split.test_conditions]
    if manifest.condition_ids != expected_conditions:
        raise ValueError("evaluation-state conditions differ from the split")
    if sha256_file(layout.arrays) != manifest.arrays_sha256:
        raise ValueError("evaluation-state array artifact hash differs")
    with np.load(layout.arrays, allow_pickle=False) as payload:
        if set(payload.files) != {"systema_reference", "metric_control_means"}:
            raise ValueError("evaluation-state array keys differ")
        systema_reference = np.asarray(payload["systema_reference"])
        metric_control_means = np.asarray(payload["metric_control_means"])
    expected_gene_count = canonical.n_expression_genes
    if systema_reference.shape != (expected_gene_count,) or metric_control_means.shape != (
        len(expected_conditions),
        expected_gene_count,
    ):
        raise ValueError("evaluation-state array shapes differ")
    if (
        _sha256_array(systema_reference) != manifest.systema_reference_content_sha256
        or _sha256_array(metric_control_means) != manifest.metric_control_means_content_sha256
    ):
        raise ValueError("evaluation-state array content hash differs")
    if not np.isfinite(systema_reference).all() or not np.isfinite(metric_control_means).all():
        raise ValueError("evaluation-state arrays contain non-finite values")
    if any(
        indices and max(indices) >= expected_gene_count
        for values in (manifest.de_gene_indices, manifest.top_de_gene_indices)
        for indices in values.values()
    ):
        raise ValueError("evaluation-state DE index is outside expression axis")
    return LoadedEvaluationState(
        manifest=manifest,
        systema_reference=np.ascontiguousarray(systema_reference),
        metric_control_means=np.ascontiguousarray(metric_control_means),
        manifest_file_sha256=sha256_file(layout.manifest),
    )
