"""Row-bound training inputs for the frozen K562 leave-one-line-out fold."""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gradpert.data import DatasetLayout
from gradpert.data._io import atomic_json
from gradpert.graphs.materialization import (
    _atomic_graph,
    _load_pruned_graph,
    _read_filtered_edges,
    verify_graph_source_checkout,
)
from gradpert.graphs.pruning import prune_incoming_edges
from gradpert.graphs.registry import load_graph_source_registry
from gradpert.graphs.views import GraphTopology
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.data import CanonicalTrainingData, TrainingPipelineStats
from gradpert.training.systems import DISABLED_NATIVE_SYSTEM_OPTIONS

DATASET = "crosscell_k562"
PROTOCOL = "txpert_cache_fixed_axis_leave_one_line_out_v1"
SCHEMA = "gradpert-crosscell-k562-training-v1"
SOURCE_SHA = "1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8"
FOLDS_SHA = "a8525e167fd3605418d1d92a15bb027e1890504d7449d98e647bdf768863aebf"
ROLES = ("train", "val", "source_control", "test", "target_control")


def file_identity(path: Path) -> list[int]:
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def frozen_rows(path: Path, fold: dict, obs: Any) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ROLES):
            raise ValueError("unexpected fold row roles")
        rows = {role: archive[role] for role in ROLES}
    lines = obs["cell_line"].astype(str).to_numpy()
    conditions = obs["condition"].astype(str).to_numpy()
    is_control = obs["control"].to_numpy()
    if not np.array_equal(is_control, conditions == "ctrl"):
        raise ValueError("control flags disagree with conditions")
    source_line = np.isin(lines, fold["train_cell_lines"])
    target_line = lines == "K562"
    for role, ids in rows.items():
        target = role in ("test", "target_control")
        control = role.endswith("control")
        if (
            ids.ndim != 1
            or ids.dtype.kind != "i"
            or not len(ids)
            or ids[0] < 0
            or ids[-1] >= len(obs)
            or np.any(np.diff(ids) <= 0)
            or len(ids) != fold["counts"][role]
            or sha256_json(ids.tolist()) != fold["row_index_hashes"][role]
            or not np.all((target_line if target else source_line)[ids])
            or not np.all(is_control[ids] == control)
        ):
            raise ValueError(f"invalid frozen rows for {role}")
    expected = np.flatnonzero(source_line | target_line)
    if not np.array_equal(np.sort(np.concatenate(list(rows.values()))), expected):
        raise ValueError("fold roles overlap or omit eligible rows")
    return rows


def seal_inputs(*, source: Path, folds: Path, checkout: Path, registry_path: Path, output: Path):
    if any(
        not path.resolve().is_relative_to("/data/yilangliu")
        for path in (source, folds, checkout, output)
    ):
        raise ValueError("cross-cell scientific inputs stay on the server")
    stat = file_identity(source)
    if sha256_file(folds) != FOLDS_SHA or sha256_file(source) != SOURCE_SHA:
        raise ValueError("upstream cross-cell identity mismatch")
    manifest = json.loads(folds.read_text())
    fold = manifest["folds"]["K562"]
    index = folds.parent / fold["row_index_file"]
    if sha256_file(index) != fold["row_index_sha256"]:
        raise ValueError("frozen K562 row archive changed")
    adata = importlib.import_module("anndata").read_h5ad(source, backed="r")
    try:
        expression = list(map(str, adata.var_names))
        if sha256_json(expression) != manifest["gene_order_sha256"]:
            raise ValueError("fixed expression axis changed")
        rows = frozen_rows(index, fold, adata.obs)
        conditions = adata.obs["condition"].astype(str).to_numpy()
        perturbations = set(conditions[np.concatenate([rows[k] for k in ("train", "val", "test")])])
    finally:
        adata.file.close()
    if stat != file_identity(source):
        raise ValueError("H5AD changed during checkout verification")
    targets = {part for c in perturbations for part in c.split("+") if part != "ctrl"}
    genes = tuple(expression + sorted(targets - set(expression)))
    registry = load_graph_source_registry(registry_path)
    upstream = verify_graph_source_checkout(checkout, registry)
    output.parent.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name, path in upstream.items():
        edges, _, _ = _read_filtered_edges(path, registry.sources[name], genes)
        graph = prune_incoming_edges(source_name=name, gene_ids=genes, weighted_edges=edges)
        artifact = output.parent / f"{name}.npz"
        _atomic_graph(artifact, graph)
        sources[name] = {"sha256": sha256_file(artifact), "edges": len(graph.edges)}
    atomic_json(
        output,
        {
            "schema": SCHEMA,
            "h5ad": str(source.resolve(strict=True)),
            "h5ad_sha256": SOURCE_SHA,
            "h5ad_file_identity": stat,
            "fold_index": str(index.resolve(strict=True)),
            "fold_manifest_sha256": FOLDS_SHA,
            "fold": fold,
            "gene_ids": list(genes),
            "expression_gene_count": len(expression),
            "expression_gene_order_sha256": manifest["gene_order_sha256"],
            "graph_gene_order_sha256": sha256_json(list(genes)),
            "graph_source_commit": registry.commit,
            "graph_source_registry_sha256": sha256_file(registry_path),
            "sources": sources,
        },
    )


def load_crosscell_topology(path: Path) -> tuple[dict, GraphTopology]:
    seal = json.loads(path.read_text())
    genes = tuple(seal["gene_ids"])
    if (
        seal["schema"] != SCHEMA
        or seal["h5ad_sha256"] != SOURCE_SHA
        or seal["fold_manifest_sha256"] != FOLDS_SHA
        or seal["fold"]["test_cell_line"] != "K562"
        or set(seal["fold"]["train_cell_lines"]) != {"RPE1", "jurkat", "hepg2"}
        or set(seal["sources"]) != {"go", "string"}
        or sha256_json(list(genes)) != seal["graph_gene_order_sha256"]
    ):
        raise ValueError("cross-cell input seal differs from the frozen protocol")
    sources = {}
    for name, entry in seal["sources"].items():
        graph = path.parent / f"{name}.npz"
        if sha256_file(graph) != entry["sha256"]:
            raise ValueError(f"cross-cell {name} graph changed")
        sources[name] = _load_pruned_graph(graph, source_name=name, gene_ids=genes)
    return seal, GraphTopology(genes, sources)


@dataclass(frozen=True)
class RowManifest:
    dataset_id: str
    protocol_id: str
    canonical_adata_sha256: str
    n_expression_genes: int
    expression_gene_order_sha256: str
    graph_gene_order_sha256: str


@dataclass(frozen=True)
class RowSplit:
    policy_id: str
    split_content_sha256: str
    train_conditions: tuple[str, ...]
    val_conditions: tuple[str, ...]
    test_conditions: tuple[str, ...]
    control_condition_id: str = "ctrl"


class CrossCellTrainingData(CanonicalTrainingData):
    """Reuse native batches with explicit row permissions and context controls."""

    def __init__(self, *, seal: dict, graph_manifest: Path, data_root: Path, run_seed: int):
        self.layout = DatasetLayout(data_root, DATASET, PROTOCOL)
        self.run_seed = run_seed
        self.runtime_graph_manifest_path = graph_manifest
        self.manifest = RowManifest(  # type: ignore[assignment]
            DATASET,
            PROTOCOL,
            SOURCE_SHA,
            seal["expression_gene_count"],
            seal["expression_gene_order_sha256"],
            seal["graph_gene_order_sha256"],
        )
        h5ad = Path(seal["h5ad"]).resolve(strict=True)
        index = Path(seal["fold_index"]).resolve(strict=True)
        if (
            not h5ad.is_relative_to("/data/yilangliu")
            or not index.is_relative_to("/data/yilangliu")
            or file_identity(h5ad) != seal["h5ad_file_identity"]
            or sha256_file(index) != seal["fold"]["row_index_sha256"]
        ):
            raise ValueError("cross-cell row sources differ from checked server inputs")
        self._adata = importlib.import_module("anndata").read_h5ad(h5ad, backed="r")
        try:
            self._bind_rows(frozen_rows(index, seal["fold"], self._adata.obs), seal)
        except BaseException:
            self.close()
            raise

    def _bind_rows(self, rows: dict[str, np.ndarray], seal: dict) -> None:
        obs = self._adata.obs
        self.expression_gene_ids = tuple(map(str, self._adata.var_names))
        self.graph_gene_ids = self.canonical_graph_gene_ids = tuple(seal["gene_ids"])
        if (
            sha256_json(list(self.expression_gene_ids))
            != self.manifest.expression_gene_order_sha256
            or len(self.expression_gene_ids) != self.manifest.n_expression_genes
            or self.expression_gene_ids != self.graph_gene_ids[: len(self.expression_gene_ids)]
        ):
            raise ValueError("cross-cell expression axis differs from graph prefix")
        self.row_ids = tuple(map(str, range(len(obs))))
        self._row_index = {row: i for i, row in enumerate(self.row_ids)}
        self.condition_ids = tuple(map(str, obs["condition"]))
        self.batch_ids = tuple(map(str, obs["batch"]))
        self.context_ids = tuple(
            f"{line}::{batch}" for line, batch in zip(obs["cell_line"], self.batch_ids, strict=True)
        )

        def conditions(role: str) -> tuple[str, ...]:
            return tuple(sorted({self.condition_ids[i] for i in rows[role]}))

        self.split = RowSplit(  # type: ignore[assignment]
            PROTOCOL,
            seal["fold"]["row_index_sha256"],
            conditions("train"),
            conditions("val"),
            conditions("test"),
        )
        self.train_row_indices = tuple(map(int, rows["train"]))
        self._val_row_indices = tuple(map(int, rows["val"]))
        self.control_row_indices = tuple(map(int, rows["source_control"]))
        self._permitted = np.zeros(len(obs), dtype=bool)
        pools: dict[str, list[str]] = {}
        for i in self.control_row_indices:
            pools.setdefault(self.context_ids[i], []).append(self.row_ids[i])
        for role in ("train", "source_control"):
            self._permitted[rows[role]] = True
        if any(self.context_ids[i] not in pools for i in self.train_row_indices):
            raise ValueError("source training context lacks controls")
        self.control_pools = {context: tuple(ids) for context, ids in pools.items()}
        self.system_options = DISABLED_NATIVE_SYSTEM_OPTIONS
        self.pipeline_stats = TrainingPipelineStats()
        self._control_expression_cache = None
        self._control_cache_position = {}
        graph_index = {gene: i for i, gene in enumerate(self.graph_gene_ids)}
        self.anchors_by_condition = {
            condition: tuple(graph_index[g] for g in condition.split("+") if g != "ctrl")
            for condition in set(self.split.train_conditions + self.split.val_conditions)
        }

    def _read_expression_indices(self, indices: Sequence[int]) -> np.ndarray[Any, Any]:
        values = np.asarray(indices, dtype=np.int64)
        if (
            not len(values)
            or values.min() < 0
            or values.max() >= len(self.row_ids)
            or not np.all(self._permitted[values])
        ):
            raise ValueError("expression request outside frozen training/source controls")
        return super()._read_expression_indices(indices)

    def preflight_controls(self):
        for idx in self._val_row_indices:
            pool = self.control_pools.get(self.context_ids[idx])
            if pool:
                rng = np.random.default_rng(self.run_seed)
                picked = tuple(pool[i] for i in rng.integers(len(pool), size=300))
                controls = self._read_expression_indices([self._row_index[r] for r in picked])
                return self.condition_ids[idx], self.context_ids[idx], picked, controls
        raise ValueError("no source validation context has compatible controls")
