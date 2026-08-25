"""Evaluator-only access to frozen control draws and complete truth populations."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from gradpert.contracts import (
    CanonicalDataManifest,
    EvaluationControlManifest,
    SplitManifest,
)
from gradpert.data import DatasetLayout
from gradpert.hashing import sha256_file, sha256_json

if TYPE_CHECKING:
    from gradpert.training.inference import LoadedControlRows


@dataclass(frozen=True)
class LoadedTruthRows:
    condition_id: str
    ordered_row_ids: tuple[str, ...]
    expression: np.ndarray[Any, Any]


@dataclass
class EvaluationCacheStats:
    requested: bool = False
    active: bool = False
    cached_rows: int = 0
    cache_hits: int = 0
    fallback_reason: str | None = None

    def payload(self) -> dict[str, object]:
        return dict(self.__dict__)


class CanonicalEvaluationData:
    """Bind one validation/test evaluator to exact canonical receipts."""

    def __init__(
        self,
        *,
        dataset_id: str,
        protocol_id: str,
        split_name: Literal["val", "test"],
        data_root: str | Path,
    ) -> None:
        self.layout = DatasetLayout(Path(data_root), dataset_id, protocol_id)
        self.split_name = split_name
        self.manifest = CanonicalDataManifest.model_validate_json(
            (self.layout.manifests / "canonical.json").read_text(encoding="utf-8")
        )
        self.split = SplitManifest.model_validate_json(
            (self.layout.manifests / "split.json").read_text(encoding="utf-8")
        )
        if (self.manifest.dataset_id, self.manifest.protocol_id) != (
            dataset_id,
            protocol_id,
        ):
            raise ValueError("canonical evaluator identity differs from its request")
        if (self.split.dataset_id, self.split.protocol_id) != (dataset_id, protocol_id):
            raise ValueError("split evaluator identity differs from its request")
        if self.manifest.split_content_sha256 != self.split.split_content_sha256:
            raise ValueError("canonical evaluator split hash differs")
        val_path = self.layout.manifests / "evaluation_controls.val.json"
        test_path = self.layout.manifests / "evaluation_controls.test.json"
        observed_control_hash = sha256_json(
            {
                "val": sha256_file(val_path),
                "test": sha256_file(test_path),
            }
        )
        if observed_control_hash != self.manifest.evaluation_controls_sha256:
            raise ValueError("evaluation control manifests differ from canonical data")
        selected_path = val_path if split_name == "val" else test_path
        self.control_manifest = EvaluationControlManifest.model_validate_json(
            selected_path.read_text(encoding="utf-8")
        )
        expected_conditions = (
            self.split.val_conditions if split_name == "val" else self.split.test_conditions
        )
        if [draw.condition_id for draw in self.control_manifest.draws] != expected_conditions:
            raise ValueError("evaluation control conditions differ from the frozen split")
        if self.control_manifest.split_content_sha256 != self.split.split_content_sha256:
            raise ValueError("evaluation controls use a different split hash")
        self.control_manifest_file_sha256 = sha256_file(selected_path)

        self.expression_gene_ids = tuple(
            (self.layout.canonical / "expression_gene_ids.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        if (
            len(self.expression_gene_ids) != self.manifest.n_expression_genes
            or sha256_json(list(self.expression_gene_ids))
            != self.manifest.expression_gene_order_sha256
        ):
            raise ValueError("evaluator expression gene order differs")
        try:
            anndata = importlib.import_module("anndata")
        except ImportError as error:  # pragma: no cover - server evaluator environment
            raise RuntimeError("anndata is required for canonical evaluation") from error
        self._adata = anndata.read_h5ad(self.layout.canonical_adata, backed="r")
        if tuple(self._adata.shape) != (
            self.manifest.n_cells,
            self.manifest.n_graph_genes,
        ):
            self.close()
            raise ValueError("evaluator H5AD shape differs from canonical manifest")
        self.row_ids = tuple(str(value) for value in self._adata.obs_names)
        if sha256_json(list(self.row_ids)) != self.manifest.observation_order_sha256:
            self.close()
            raise ValueError("evaluator H5AD row order differs from canonical manifest")
        self._row_index = {row_id: index for index, row_id in enumerate(self.row_ids)}
        obs = self._adata.obs
        self.condition_ids = tuple(str(value) for value in obs["condition"])
        self.context_ids = tuple(
            f"{cell_type}::{batch}"
            for cell_type, batch in zip(obs["cell_type"], obs["batch"], strict=True)
        )
        self._control_mask = np.asarray(obs["control"], dtype=bool)
        if not np.array_equal(
            self._control_mask,
            np.asarray(self.condition_ids) == self.split.control_condition_id,
        ):
            self.close()
            raise ValueError("evaluator control flag and condition disagree")
        self._indices_by_condition = {
            condition: tuple(
                index
                for index, observed_condition in enumerate(self.condition_ids)
                if observed_condition == condition
            )
            for condition in expected_conditions
        }
        if any(not indices for indices in self._indices_by_condition.values()):
            self.close()
            raise ValueError("evaluator truth population is empty")
        self.cache_stats = EvaluationCacheStats()
        self._expression_cache: np.ndarray[Any, Any] | None = None
        self._cache_position: dict[int, int] = {}

    def close(self) -> None:
        file_manager = getattr(getattr(self, "_adata", None), "file", None)
        if file_manager is not None:
            file_manager.close()

    def __enter__(self) -> CanonicalEvaluationData:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_expression_indices_uncached(self, indices: tuple[int, ...]) -> np.ndarray[Any, Any]:
        if not indices:
            raise ValueError("evaluator expression request must be non-empty")
        values = np.asarray(indices, dtype=np.int64)
        unique_sorted, inverse = np.unique(values, return_inverse=True)
        matrix = self._adata[unique_sorted, : self.manifest.n_expression_genes].X
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        restored = np.asarray(dense, dtype=np.float32)[inverse]
        if restored.shape != (len(indices), self.manifest.n_expression_genes):
            raise ValueError("evaluator expression slice has an unexpected shape")
        if not np.isfinite(restored).all():
            raise ValueError("evaluator expression contains non-finite values")
        return cast(np.ndarray[Any, Any], np.ascontiguousarray(restored))

    def configure_expression_cache(self, *, enabled: bool) -> float:
        """Cache exactly validation truth and compatible controls, never test data."""

        import time

        self.cache_stats = EvaluationCacheStats(requested=enabled)
        started = time.perf_counter()
        if not enabled:
            return 0.0
        if self.split_name != "val":
            raise ValueError("expression caching is restricted to validation data")
        truth_indices = {
            index for indices in self._indices_by_condition.values() for index in indices
        }
        truth_contexts = {self.context_ids[index] for index in truth_indices}
        control_indices = {
            index
            for index, (is_control, context) in enumerate(
                zip(self._control_mask, self.context_ids, strict=True)
            )
            if is_control and context in truth_contexts
        }
        requested = tuple(sorted(truth_indices | control_indices))
        try:
            cache = self._read_expression_indices_uncached(requested)
        except (MemoryError, OSError) as error:
            self.cache_stats.fallback_reason = type(error).__name__
        else:
            self._expression_cache = cache
            self._cache_position = {index: position for position, index in enumerate(requested)}
            self.cache_stats.active = True
            self.cache_stats.cached_rows = len(requested)
        return (time.perf_counter() - started) * 1000.0

    def _read_expression_indices(self, indices: tuple[int, ...]) -> np.ndarray[Any, Any]:
        if self._expression_cache is not None:
            try:
                positions = [self._cache_position[index] for index in indices]
            except KeyError as error:
                raise RuntimeError(
                    f"validation cache lacks required canonical row {error.args[0]}"
                ) from error
            self.cache_stats.cache_hits += 1
            return cast(
                np.ndarray[Any, Any], np.ascontiguousarray(self._expression_cache[positions])
            )
        return self._read_expression_indices_uncached(indices)

    def load_control_rows(self, ordered_row_ids: tuple[str, ...]) -> LoadedControlRows:
        from gradpert.training.inference import LoadedControlRows

        try:
            indices = tuple(self._row_index[row_id] for row_id in ordered_row_ids)
        except KeyError as error:
            raise ValueError(f"evaluation control row is absent: {error.args[0]}") from error
        if any(not self._control_mask[index] for index in indices):
            raise ValueError("evaluation control draw contains a perturbed row")
        return LoadedControlRows(
            ordered_row_ids=ordered_row_ids,
            expression=self._read_expression_indices(indices),
        )

    def load_truth_rows(self, condition_id: str) -> LoadedTruthRows:
        try:
            indices = self._indices_by_condition[condition_id]
        except KeyError as error:
            raise ValueError(f"condition is outside evaluator split: {condition_id}") from error
        return LoadedTruthRows(
            condition_id=condition_id,
            ordered_row_ids=tuple(self.row_ids[index] for index in indices),
            expression=self._read_expression_indices(indices),
        )

    def compatible_control_pool_mean(self, condition_id: str) -> np.ndarray[Any, Any]:
        """Mean the complete union of same-context control rows for a condition."""

        try:
            truth_indices = self._indices_by_condition[condition_id]
        except KeyError as error:
            raise ValueError(f"condition is outside evaluator split: {condition_id}") from error
        contexts = {self.context_ids[index] for index in truth_indices}
        control_indices = tuple(
            index
            for index, (is_control, context) in enumerate(
                zip(self._control_mask, self.context_ids, strict=True)
            )
            if is_control and context in contexts
        )
        if not control_indices:
            raise ValueError(f"no compatible evaluator controls for {condition_id}")
        return cast(
            np.ndarray[Any, Any],
            np.asarray(
                self._read_expression_indices(control_indices).mean(axis=0),
                dtype=np.float32,
            ),
        )
