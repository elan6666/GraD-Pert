"""Manifest-bound canonical data access and deterministic native train batches."""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gradpert.contracts import CanonicalDataManifest, SplitManifest
from gradpert.data import DatasetLayout
from gradpert.hashing import sha256_json
from gradpert.training.controls import TrainingControlPairer
from gradpert.training.systems import DISABLED_NATIVE_SYSTEM_OPTIONS, NativeSystemOptions

if TYPE_CHECKING:
    import torch

    from gradpert.training.batch import GraDPertTrainingBatch


@dataclass(frozen=True)
class BaselineFitData:
    """Immutable training-only arrays consumed by nonlearned baselines."""

    perturbed_expression: np.ndarray[Any, Any]
    condition_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    batch_ids: tuple[str, ...]
    control_expression: np.ndarray[Any, Any]
    control_context_ids: tuple[str, ...]
    control_batch_ids: tuple[str, ...]


@dataclass
class TrainingPipelineStats:
    """Small runtime evidence for the optimized data path."""

    control_cache_requested: bool = False
    control_cache_active: bool = False
    control_cache_rows: int = 0
    control_cache_fallback_reason: str | None = None
    merged_read_batches: int = 0
    cached_control_batches: int = 0
    prefetch_requested: bool = False
    prefetch_active: bool = False
    prefetch_fallback_reason: str | None = None
    pin_memory_requested: bool = False
    pin_memory_active: bool = False
    pin_memory_fallback_reason: str | None = None
    nonblocking_transfer_requested: bool = False
    nonblocking_transfer_active: bool = False
    nonblocking_transfer_fallback_reason: str | None = None
    epoch_batch_identity_sha256: str | None = None
    first_perturbed_row_ids_sha256: str | None = None
    first_control_row_ids_sha256: str | None = None
    first_pretransfer_control_sha256: str | None = None
    first_pretransfer_target_sha256: str | None = None
    yielded_batches: int = 0

    def payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class _TrainingBatchSpec:
    perturbed_indices: tuple[int, ...]
    perturbed_row_ids: tuple[str, ...]
    control_indices: tuple[int, ...]
    control_row_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    anchors_by_condition: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class TrainingBatchIdentitySpec:
    """Expression-free ordered identity for one deterministic train batch."""

    perturbed_row_ids: tuple[str, ...]
    control_row_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    anchor_gene_ids_by_condition: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class _CpuTrainingBatch:
    spec: _TrainingBatchSpec
    control_expression: np.ndarray[Any, Any]
    target_expression: np.ndarray[Any, Any]
    data_read_ms: float


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], byteorder="big")


def condition_limited_epoch_batches(
    *,
    condition_ids: Sequence[str],
    run_seed: int,
    epoch: int,
    batch_size: int = 64,
    max_unique_conditions: int = 8,
) -> tuple[tuple[int, ...], ...]:
    """Shuffle every cell once, then form batches with bounded condition fanout.

    Per-condition queues are independently shuffled each epoch. The condition
    tie order is fixed by the run seed, so batch counts are identical across
    epochs. A partial batch is retained when eight remaining condition queues
    cannot fill 64 rows; only a final singleton is dropped because both native
    prediction modules contain BatchNorm.
    """

    if run_seed < 0 or epoch < 0:
        raise ValueError("run_seed and epoch must be nonnegative")
    if batch_size <= 1 or not 1 <= max_unique_conditions <= batch_size:
        raise ValueError("invalid batch size or unique-condition cap")
    if not condition_ids or any(not condition_id for condition_id in condition_ids):
        raise ValueError("condition IDs must be non-empty")

    by_condition: dict[str, list[int]] = {}
    for row_index, condition_id in enumerate(condition_ids):
        by_condition.setdefault(condition_id, []).append(row_index)
    queues: dict[str, deque[int]] = {}
    for condition_id in sorted(by_condition):
        values = np.asarray(by_condition[condition_id], dtype=np.int64)
        rng = np.random.Generator(
            np.random.PCG64(_stable_seed(run_seed, epoch, condition_id, "cell_order"))
        )
        queues[condition_id] = deque(int(index) for index in values[rng.permutation(len(values))])

    tie_rng = np.random.Generator(np.random.PCG64(_stable_seed(run_seed, "condition_order")))
    tie_order = {
        condition_id: rank
        for rank, condition_id in enumerate(
            np.asarray(sorted(queues), dtype=object)[tie_rng.permutation(len(queues))].tolist()
        )
    }
    batches: list[tuple[int, ...]] = []
    while any(queues.values()):
        active = [condition_id for condition_id, queue in queues.items() if queue]
        active.sort(key=lambda item: (-len(queues[item]), tie_order[item]))
        selected = active[:max_unique_conditions]
        target_size = min(batch_size, sum(len(queues[item]) for item in selected))

        batch: list[int] = []
        while len(batch) < target_size:
            progressed = False
            for condition_id in selected:
                queue = queues[condition_id]
                if queue and len(batch) < batch_size:
                    batch.append(queue.popleft())
                    progressed = True
            if not progressed:  # pragma: no cover - protected by the capacity check
                raise AssertionError("selected condition queues were exhausted early")
        if len(batch) > 1:
            batches.append(tuple(batch))
    return tuple(batches)


class CanonicalTrainingData:
    """Read only train perturbations and compatible controls from one sealed H5AD."""

    def __init__(
        self,
        *,
        dataset_id: str,
        protocol_id: str,
        data_root: str | Path,
        run_seed: int,
        graph_gene_ids_override: Sequence[str] | None = None,
        graph_manifest_path_override: str | Path | None = None,
    ) -> None:
        if run_seed < 0:
            raise ValueError("run_seed must be nonnegative")
        self.layout = DatasetLayout(Path(data_root), dataset_id, protocol_id)
        self.run_seed = run_seed
        manifest_path = self.layout.manifests / "canonical.json"
        split_path = self.layout.manifests / "split.json"
        if not manifest_path.is_file() or not split_path.is_file():
            raise ValueError("canonical and split manifests must be regular files")
        self.manifest = CanonicalDataManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        self.split = SplitManifest.model_validate_json(split_path.read_text(encoding="utf-8"))
        if (self.manifest.dataset_id, self.manifest.protocol_id) != (
            dataset_id,
            protocol_id,
        ):
            raise ValueError("canonical manifest identity differs from the data request")
        if (self.split.dataset_id, self.split.protocol_id) != (dataset_id, protocol_id):
            raise ValueError("split manifest identity differs from the data request")
        if self.manifest.split_content_sha256 != self.split.split_content_sha256:
            raise ValueError("canonical and split content hashes differ")

        self.expression_gene_ids = tuple(
            (self.layout.canonical / "expression_gene_ids.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        canonical_graph_gene_ids = tuple(
            (self.layout.canonical / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines()
        )
        if canonical_graph_gene_ids[: len(self.expression_gene_ids)] != self.expression_gene_ids:
            raise ValueError("expression genes must be the leading graph-axis prefix")
        if (
            len(self.expression_gene_ids) != self.manifest.n_expression_genes
            or len(canonical_graph_gene_ids) != self.manifest.n_graph_genes
            or sha256_json(list(self.expression_gene_ids))
            != self.manifest.expression_gene_order_sha256
            or sha256_json(list(canonical_graph_gene_ids)) != self.manifest.graph_gene_order_sha256
        ):
            raise ValueError("runtime gene axes differ from the canonical manifest")
        self.canonical_graph_gene_ids = canonical_graph_gene_ids
        self.graph_gene_ids = (
            canonical_graph_gene_ids
            if graph_gene_ids_override is None
            else tuple(graph_gene_ids_override)
        )
        if not self.graph_gene_ids or len(self.graph_gene_ids) != len(set(self.graph_gene_ids)):
            raise ValueError("runtime graph axis must contain unique gene IDs")
        self.runtime_graph_manifest_path = (
            self.layout.root / "graphs" / "manifest.json"
            if graph_manifest_path_override is None
            else Path(graph_manifest_path_override).resolve(strict=True)
        )

        try:
            anndata = importlib.import_module("anndata")
        except ImportError as error:  # pragma: no cover - server training environment
            raise RuntimeError("anndata is required for canonical training data") from error
        self._adata = anndata.read_h5ad(self.layout.canonical_adata, backed="r")
        if tuple(self._adata.shape) != (
            self.manifest.n_cells,
            self.manifest.n_graph_genes,
        ):
            self.close()
            raise ValueError("canonical H5AD shape differs from its manifest")
        observed_graph_genes = tuple(str(value) for value in self._adata.var["gene_name"])
        self.row_ids = tuple(str(value) for value in self._adata.obs_names)
        if (
            observed_graph_genes != self.canonical_graph_gene_ids
            or sha256_json(list(self.row_ids)) != self.manifest.observation_order_sha256
        ):
            self.close()
            raise ValueError("canonical H5AD order differs from sealed axes")

        obs = self._adata.obs
        required = {"condition", "cell_type", "batch", "control"}
        if not required.issubset(obs.columns):
            self.close()
            raise ValueError("canonical H5AD lacks required training metadata")
        self.condition_ids = tuple(str(value) for value in obs["condition"])
        self.context_ids = tuple(
            f"{cell_type}::{batch}"
            for cell_type, batch in zip(obs["cell_type"], obs["batch"], strict=True)
        )
        self.batch_ids = tuple(str(value) for value in obs["batch"])
        controls = np.asarray(obs["control"], dtype=bool)
        if not np.array_equal(
            controls,
            np.asarray(self.condition_ids) == self.split.control_condition_id,
        ):
            self.close()
            raise ValueError("canonical control flag and condition disagree")
        self._row_index = {row_id: index for index, row_id in enumerate(self.row_ids)}
        if len(self._row_index) != len(self.row_ids):
            self.close()
            raise ValueError("canonical row IDs are not unique")

        train_set = set(self.split.train_conditions)
        self.train_row_indices = tuple(
            index for index, condition in enumerate(self.condition_ids) if condition in train_set
        )
        if not self.train_row_indices:
            self.close()
            raise ValueError("canonical training partition contains no perturbed cells")
        observed_train = {self.condition_ids[index] for index in self.train_row_indices}
        if observed_train != train_set:
            self.close()
            raise ValueError("canonical H5AD lacks one or more frozen train conditions")

        pools: dict[str, list[str]] = {}
        for index, is_control in enumerate(controls):
            if is_control:
                pools.setdefault(self.context_ids[index], []).append(self.row_ids[index])
        train_contexts = {self.context_ids[index] for index in self.train_row_indices}
        missing_contexts = sorted(train_contexts - set(pools))
        if missing_contexts:
            self.close()
            raise ValueError(f"training contexts lack compatible controls: {missing_contexts}")
        self.control_pools: Mapping[str, tuple[str, ...]] = {
            context: tuple(rows) for context, rows in pools.items()
        }
        self.control_row_indices = tuple(
            index
            for index, is_control in enumerate(controls)
            if is_control and self.context_ids[index] in train_contexts
        )
        self.system_options = DISABLED_NATIVE_SYSTEM_OPTIONS
        self.pipeline_stats = TrainingPipelineStats()
        self._control_expression_cache: np.ndarray[Any, Any] | None = None
        self._control_cache_position: dict[int, int] = {}

        graph_index = {gene_id: index for index, gene_id in enumerate(self.graph_gene_ids)}
        all_conditions = (
            *self.split.train_conditions,
            *self.split.val_conditions,
            *self.split.test_conditions,
        )
        anchors: dict[str, tuple[int, ...]] = {}
        for condition in all_conditions:
            components = tuple(
                component
                for component in condition.split("+")
                if component != self.split.control_condition_id
            )
            if not components or len(components) != len(set(components)):
                self.close()
                raise ValueError(f"invalid perturbation condition: {condition}")
            try:
                anchors[condition] = tuple(graph_index[component] for component in components)
            except KeyError as error:
                self.close()
                raise ValueError(
                    f"condition target is absent from graph axis: {error.args[0]}"
                ) from error
        self.anchors_by_condition: Mapping[str, tuple[int, ...]] = anchors

    def close(self) -> None:
        file_manager = getattr(getattr(self, "_adata", None), "file", None)
        if file_manager is not None:
            file_manager.close()

    def __enter__(self) -> CanonicalTrainingData:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def require_experiment_data_contract(
        self,
        *,
        registry_version: str,
        split_policy: str,
    ) -> None:
        """Fail closed when a self-contained config names stale data policy."""

        if registry_version != "datasets-v2":
            raise ValueError("experiment config must use the datasets-v2 registry")
        if split_policy != self.split.policy_id:
            raise ValueError("experiment config split policy differs from canonical split")

    def _read_expression_indices(self, indices: Sequence[int]) -> np.ndarray[Any, Any]:
        if not indices:
            raise ValueError("expression row request must be non-empty")
        values = np.asarray(indices, dtype=np.int64)
        if int(values.min()) < 0 or int(values.max()) >= len(self.row_ids):
            raise ValueError("expression row index is outside canonical data")
        unique_sorted, inverse = np.unique(values, return_inverse=True)
        matrix = self._adata[unique_sorted, : self.manifest.n_expression_genes].X
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        restored = np.asarray(dense, dtype=np.float32)[inverse]
        if restored.shape != (len(indices), self.manifest.n_expression_genes):
            raise ValueError("canonical expression slice has an unexpected shape")
        if not np.isfinite(restored).all():
            raise ValueError("canonical expression slice contains non-finite values")
        return np.ascontiguousarray(restored)

    def configure_system_optimizations(self, options: NativeSystemOptions) -> float:
        """Build bounded caches and return their wall time in milliseconds."""

        if self.pipeline_stats.yielded_batches:
            raise RuntimeError("system optimizations must be configured before iteration")
        self.system_options = options
        self.pipeline_stats = TrainingPipelineStats(
            control_cache_requested=options.control_expression_cache,
            prefetch_requested=options.background_prefetch,
            pin_memory_requested=options.pin_memory,
            nonblocking_transfer_requested=options.nonblocking_transfer,
        )
        started = time.perf_counter()
        if options.control_expression_cache:
            try:
                cache = self._read_expression_indices(self.control_row_indices)
            except (MemoryError, OSError) as error:
                self._control_expression_cache = None
                self._control_cache_position = {}
                self.pipeline_stats.control_cache_fallback_reason = type(error).__name__
            else:
                self._control_expression_cache = cache
                self._control_cache_position = {
                    row_index: position
                    for position, row_index in enumerate(self.control_row_indices)
                }
                self.pipeline_stats.control_cache_active = True
                self.pipeline_stats.control_cache_rows = len(self.control_row_indices)
        return (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _array_sha256(array: np.ndarray[Any, Any]) -> str:
        contiguous = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
        return digest.hexdigest()

    def _batch_specs(
        self,
        *,
        epoch: int,
        batch_size: int,
        max_unique_conditions: int,
    ) -> tuple[_TrainingBatchSpec, ...]:
        train_conditions = tuple(self.condition_ids[index] for index in self.train_row_indices)
        relative_batches = condition_limited_epoch_batches(
            condition_ids=train_conditions,
            run_seed=self.run_seed,
            epoch=epoch,
            batch_size=batch_size,
            max_unique_conditions=max_unique_conditions,
        )
        pairer = TrainingControlPairer(run_seed=self.run_seed)
        specs: list[_TrainingBatchSpec] = []
        identity_rows: list[dict[str, object]] = []
        for relative_indices in relative_batches:
            perturbed_indices = tuple(self.train_row_indices[index] for index in relative_indices)
            perturbed_row_ids = tuple(self.row_ids[index] for index in perturbed_indices)
            contexts = tuple(self.context_ids[index] for index in perturbed_indices)
            pairing = pairer.pair_epoch(
                epoch=epoch,
                perturbed_row_ids=perturbed_row_ids,
                context_ids=contexts,
                control_pools=self.control_pools,
            )
            control_indices = tuple(self._row_index[row_id] for row_id in pairing.control_row_ids)
            condition_ids = tuple(self.condition_ids[index] for index in perturbed_indices)
            unique_conditions = tuple(dict.fromkeys(condition_ids))
            spec = _TrainingBatchSpec(
                perturbed_indices=perturbed_indices,
                perturbed_row_ids=perturbed_row_ids,
                control_indices=control_indices,
                control_row_ids=pairing.control_row_ids,
                condition_ids=condition_ids,
                anchors_by_condition={
                    condition: self.anchors_by_condition[condition]
                    for condition in unique_conditions
                },
            )
            specs.append(spec)
            identity_rows.append(
                {
                    "perturbed_row_ids_sha256": sha256_json(list(perturbed_row_ids)),
                    "control_row_ids_sha256": sha256_json(list(pairing.control_row_ids)),
                }
            )
        self.pipeline_stats.epoch_batch_identity_sha256 = sha256_json(identity_rows)
        return tuple(specs)

    def training_batch_identity_specs(
        self,
        *,
        epoch: int,
        batch_size: int = 64,
        max_unique_conditions: int = 8,
    ) -> tuple[TrainingBatchIdentitySpec, ...]:
        """Return the deterministic batch schedule without reading expression arrays."""

        identities: list[TrainingBatchIdentitySpec] = []
        for spec in self._batch_specs(
            epoch=epoch,
            batch_size=batch_size,
            max_unique_conditions=max_unique_conditions,
        ):
            anchor_gene_ids: dict[str, tuple[str, ...]] = {}
            for condition, anchors in spec.anchors_by_condition.items():
                if any(anchor < 0 or anchor >= len(self.graph_gene_ids) for anchor in anchors):
                    raise ValueError("training batch anchor is outside the runtime graph axis")
                anchor_gene_ids[condition] = tuple(
                    self.graph_gene_ids[anchor] for anchor in anchors
                )
            identities.append(
                TrainingBatchIdentitySpec(
                    perturbed_row_ids=spec.perturbed_row_ids,
                    control_row_ids=spec.control_row_ids,
                    condition_ids=spec.condition_ids,
                    anchor_gene_ids_by_condition=anchor_gene_ids,
                )
            )
        return tuple(identities)

    def _materialize_cpu_batch(self, spec: _TrainingBatchSpec) -> _CpuTrainingBatch:
        started = time.perf_counter()
        if self._control_expression_cache is not None:
            positions = [self._control_cache_position[index] for index in spec.control_indices]
            control = np.ascontiguousarray(self._control_expression_cache[positions])
            target = self._read_expression_indices(spec.perturbed_indices)
            self.pipeline_stats.cached_control_batches += 1
        elif self.system_options.merged_hdf5_reads:
            merged = self._read_expression_indices((*spec.control_indices, *spec.perturbed_indices))
            boundary = len(spec.control_indices)
            control = np.ascontiguousarray(merged[:boundary])
            target = np.ascontiguousarray(merged[boundary:])
            self.pipeline_stats.merged_read_batches += 1
        else:
            control = self._read_expression_indices(spec.control_indices)
            target = self._read_expression_indices(spec.perturbed_indices)
        return _CpuTrainingBatch(
            spec=spec,
            control_expression=control,
            target_expression=target,
            data_read_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _iter_cpu_batches(
        self, specs: tuple[_TrainingBatchSpec, ...]
    ) -> Iterator[_CpuTrainingBatch]:
        if not self.system_options.background_prefetch:
            yield from (self._materialize_cpu_batch(spec) for spec in specs)
            return
        try:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gradpert-prefetch")
        except (OSError, RuntimeError) as error:
            self.pipeline_stats.prefetch_fallback_reason = type(error).__name__
            yield from (self._materialize_cpu_batch(spec) for spec in specs)
            return
        self.pipeline_stats.prefetch_active = True
        futures: deque[Future[_CpuTrainingBatch]] = deque()
        next_index = 0
        try:
            while next_index < min(self.system_options.prefetch_depth, len(specs)):
                futures.append(executor.submit(self._materialize_cpu_batch, specs[next_index]))
                next_index += 1
            while futures:
                future = futures.popleft()
                if next_index < len(specs):
                    futures.append(executor.submit(self._materialize_cpu_batch, specs[next_index]))
                    next_index += 1
                yield future.result()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def _to_training_batch(
        self,
        cpu_batch: _CpuTrainingBatch,
        *,
        device: torch.device,
    ) -> GraDPertTrainingBatch:
        import torch

        from gradpert.training.batch import GraDPertTrainingBatch

        transfer_started = time.perf_counter()
        control_cpu = torch.from_numpy(cpu_batch.control_expression)
        target_cpu = torch.from_numpy(cpu_batch.target_expression)
        pin_active = False
        if self.system_options.pin_memory and device.type == "cuda":
            try:
                control_cpu = control_cpu.pin_memory()
                target_cpu = target_cpu.pin_memory()
            except RuntimeError as error:
                self.pipeline_stats.pin_memory_fallback_reason = type(error).__name__
                control_cpu = torch.from_numpy(cpu_batch.control_expression)
                target_cpu = torch.from_numpy(cpu_batch.target_expression)
            else:
                pin_active = True
                self.pipeline_stats.pin_memory_active = True
        elif self.system_options.pin_memory:
            self.pipeline_stats.pin_memory_fallback_reason = "non_cuda_device"
        nonblocking = self.system_options.nonblocking_transfer and pin_active
        if self.system_options.nonblocking_transfer and not nonblocking:
            self.pipeline_stats.nonblocking_transfer_fallback_reason = (
                self.pipeline_stats.pin_memory_fallback_reason or "pin_memory_inactive"
            )
        if nonblocking:
            self.pipeline_stats.nonblocking_transfer_active = True
        control_tensor = control_cpu.to(device=device, non_blocking=nonblocking)
        target_tensor = target_cpu.to(device=device, non_blocking=nonblocking)
        host_to_device_ms = (time.perf_counter() - transfer_started) * 1000.0

        first = self.pipeline_stats.yielded_batches == 0
        perturbed_hash = sha256_json(list(cpu_batch.spec.perturbed_row_ids)) if first else None
        control_ids_hash = sha256_json(list(cpu_batch.spec.control_row_ids)) if first else None
        control_hash = self._array_sha256(cpu_batch.control_expression) if first else None
        target_hash = self._array_sha256(cpu_batch.target_expression) if first else None
        if first:
            self.pipeline_stats.first_perturbed_row_ids_sha256 = perturbed_hash
            self.pipeline_stats.first_control_row_ids_sha256 = control_ids_hash
            self.pipeline_stats.first_pretransfer_control_sha256 = control_hash
            self.pipeline_stats.first_pretransfer_target_sha256 = target_hash
        self.pipeline_stats.yielded_batches += 1
        return GraDPertTrainingBatch(
            control_expression=control_tensor,
            target_expression=target_tensor,
            condition_ids=cpu_batch.spec.condition_ids,
            anchors_by_condition=cpu_batch.spec.anchors_by_condition,
            perturbed_row_ids=cpu_batch.spec.perturbed_row_ids,
            control_row_ids=cpu_batch.spec.control_row_ids,
            perturbed_row_ids_sha256=perturbed_hash,
            control_row_ids_sha256=control_ids_hash,
            pretransfer_control_sha256=control_hash,
            pretransfer_target_sha256=target_hash,
            data_read_ms=cpu_batch.data_read_ms,
            host_to_device_ms=host_to_device_ms,
        )

    def load_control_rows(self, ordered_row_ids: Sequence[str]) -> np.ndarray[Any, Any]:
        try:
            indices = [self._row_index[row_id] for row_id in ordered_row_ids]
        except KeyError as error:
            raise ValueError(f"requested control row is absent: {error.args[0]}") from error
        if any(self.condition_ids[index] != self.split.control_condition_id for index in indices):
            raise ValueError("control-row request contains a perturbed cell")
        return self._read_expression_indices(indices)

    def load_baseline_fit_data(self) -> BaselineFitData:
        """Materialize only train perturbations and controls in train contexts."""

        perturbation_indices = self.train_row_indices
        control_indices = self.control_row_indices
        return BaselineFitData(
            perturbed_expression=self._read_expression_indices(perturbation_indices),
            condition_ids=tuple(self.condition_ids[index] for index in perturbation_indices),
            context_ids=tuple(self.context_ids[index] for index in perturbation_indices),
            batch_ids=tuple(self.batch_ids[index] for index in perturbation_indices),
            control_expression=self._read_expression_indices(control_indices),
            control_context_ids=tuple(self.context_ids[index] for index in control_indices),
            control_batch_ids=tuple(self.batch_ids[index] for index in control_indices),
        )

    def steps_per_epoch(
        self,
        *,
        batch_size: int = 64,
        max_unique_conditions: int = 8,
    ) -> int:
        if batch_size <= 1:
            raise ValueError("batch_size must exceed one")
        train_conditions = tuple(self.condition_ids[index] for index in self.train_row_indices)
        steps = len(
            condition_limited_epoch_batches(
                condition_ids=train_conditions,
                run_seed=self.run_seed,
                epoch=0,
                batch_size=batch_size,
                max_unique_conditions=max_unique_conditions,
            )
        )
        if steps <= 0:
            raise ValueError("training partition cannot form one full batch")
        return steps

    def iter_train_epoch(
        self,
        *,
        epoch: int,
        device: torch.device,
        batch_size: int = 64,
        max_unique_conditions: int = 8,
    ) -> Iterator[GraDPertTrainingBatch]:
        specs = self._batch_specs(
            epoch=epoch,
            batch_size=batch_size,
            max_unique_conditions=max_unique_conditions,
        )
        for cpu_batch in self._iter_cpu_batches(specs):
            yield self._to_training_batch(cpu_batch, device=device)


def write_training_data_receipt(data: CanonicalTrainingData, path: str | Path) -> None:
    """Write the small immutable data identity consumed by a native run."""

    payload = {
        "schema_version": "native-training-data-v1",
        "dataset_id": data.manifest.dataset_id,
        "protocol_id": data.manifest.protocol_id,
        "canonical_data_sha256": data.manifest.canonical_adata_sha256,
        "split_content_sha256": data.split.split_content_sha256,
        "expression_gene_order_sha256": data.manifest.expression_gene_order_sha256,
        "graph_gene_order_sha256": data.manifest.graph_gene_order_sha256,
        "canonical_graph_gene_order_sha256": data.manifest.graph_gene_order_sha256,
        "runtime_graph_gene_order_sha256": sha256_json(list(data.graph_gene_ids)),
        "runtime_graph_gene_count": len(data.graph_gene_ids),
        "train_perturbed_cell_count": len(data.train_row_indices),
        "drop_last": False,
        "singleton_tail_policy": "drop_for_batchnorm",
        "batch_order": "condition_limited_seeded_v1",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
