"""Direct Top-500 HVG recomputation and isolated reduced-graph materialization."""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from gradpert.contracts.base import NonEmpty, Sha256, StrictManifest
from gradpert.contracts.manifests import CanonicalDataManifest, SplitManifest
from gradpert.data import DatasetLayout
from gradpert.data._io import atomic_json, atomic_text
from gradpert.data.preprocessing import (
    canonicalize_metadata,
    filter_cells_by_perturbation_effect,
)
from gradpert.data.schema import DatasetRegistryEntry
from gradpert.graphs.materialization import (
    _atomic_graph,
    _load_pruned_graph,
    _read_filtered_edges,
    verify_graph_source_checkout,
)
from gradpert.graphs.pruning import prune_incoming_edges
from gradpert.graphs.registry import GraphSourceRegistry
from gradpert.graphs.views import GraphTopology
from gradpert.hashing import sha256_file, sha256_json


class ReducedGraphManifest(StrictManifest):
    schema_version: Literal["recomputed-top500-graph-v1"]
    dataset_id: Literal["nadig_jurkat"]
    protocol_id: NonEmpty
    canonical_data_sha256: Sha256
    split_content_sha256: Sha256
    source_h5ad_sha256: Sha256
    source_registry_sha256: Sha256
    hvg_method: Literal["scanpy.pp.highly_variable_genes"]
    normalize_total: Literal[4000]
    log1p: Literal[True]
    requested_hvg_count: Literal[500]
    expression_gene_count: Literal[5000]
    direct_top500_gene_ids: list[NonEmpty]
    direct_top500_gene_order_sha256: Sha256
    frozen_rank_top500_gene_order_sha256: Sha256
    candidate_target_count: int = Field(ge=1)
    graph_gene_ids: list[NonEmpty]
    graph_gene_order_sha256: Sha256
    graph_gene_count: int = Field(ge=1)
    source_artifact_sha256: dict[Literal["go", "string"], Sha256]
    source_pruned_nonself_edge_count: dict[Literal["go", "string"], int]
    topology_content_sha256: Sha256
    materialization_wall_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def enforce_content(self) -> ReducedGraphManifest:
        if len(self.direct_top500_gene_ids) != 500:
            raise ValueError("direct Top-500 receipt must contain exactly 500 genes")
        if len(self.graph_gene_ids) != self.graph_gene_count:
            raise ValueError("reduced graph gene count differs from its axis")
        if len(self.graph_gene_ids) != len(set(self.graph_gene_ids)):
            raise ValueError("reduced graph axis contains duplicate genes")
        if sha256_json(self.direct_top500_gene_ids) != self.direct_top500_gene_order_sha256:
            raise ValueError("direct Top-500 gene hash differs")
        if self.direct_top500_gene_order_sha256 != self.frozen_rank_top500_gene_order_sha256:
            raise ValueError("direct Top-500 differs from the frozen dispersion ranking")
        if sha256_json(self.graph_gene_ids) != self.graph_gene_order_sha256:
            raise ValueError("reduced graph gene hash differs")
        expected_topology = sha256_json(
            {
                "graph_gene_order_sha256": self.graph_gene_order_sha256,
                "sources": self.source_artifact_sha256,
            }
        )
        if expected_topology != self.topology_content_sha256:
            raise ValueError("reduced topology content hash differs")
        return self


def _rank_selected_hvgs(adata: Any, *, expected_count: int) -> tuple[str, ...]:
    required = {"gene_name", "highly_variable", "dispersions_norm"}
    if not required.issubset(adata.var.columns):
        raise ValueError("HVG ranking requires gene_name/highly_variable/dispersions_norm")
    genes = tuple(str(value) for value in adata.var["gene_name"])
    selected = np.asarray(adata.var["highly_variable"], dtype=bool)
    dispersions = np.asarray(adata.var["dispersions_norm"], dtype=np.float64)
    if int(selected.sum()) != expected_count or not np.isfinite(dispersions[selected]).all():
        raise ValueError("HVG selection count or normalized dispersions are invalid")
    positions = np.flatnonzero(selected)
    ranked = sorted(positions.tolist(), key=lambda position: (-dispersions[position], position))
    return tuple(genes[position] for position in ranked)


def _direct_top500(
    source: Any,
    entry: DatasetRegistryEntry,
) -> tuple[str, ...]:
    filtered, _ = filter_cells_by_perturbation_effect(source, entry)
    canonical, _ = canonicalize_metadata(filtered, entry)
    scanpy = importlib.import_module("scanpy")
    scanpy.pp.normalize_total(canonical, target_sum=4000)
    scanpy.pp.log1p(canonical)
    scanpy.pp.highly_variable_genes(canonical, n_top_genes=500, subset=False)
    return _rank_selected_hvgs(canonical, expected_count=500)


def _candidate_targets(split: SplitManifest) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                component
                for condition in (
                    *split.train_conditions,
                    *split.val_conditions,
                    *split.test_conditions,
                )
                for component in condition.split("+")
                if component != split.control_condition_id
            }
        )
    )


def materialize_recomputed_top500_graph(
    *,
    entry: DatasetRegistryEntry,
    data_root: str | Path,
    destination: str | Path,
    source_registry_path: str | Path,
    source_registry: GraphSourceRegistry,
    official_checkout: str | Path,
) -> ReducedGraphManifest:
    """Build a separate Top-500-union-target graph without changing canonical data."""

    if entry.dataset_id != "nadig_jurkat" or entry.source.semantics != "raw_single_cell":
        raise ValueError("the current graph-axis pilot is frozen to raw Nadig Jurkat")
    materialization_started = time.perf_counter()
    layout = DatasetLayout(Path(data_root), entry.dataset_id, entry.protocol_id)
    target = Path(destination)
    manifest_path = target / "manifest.json"
    if target.exists() and any(target.iterdir()):
        return load_reduced_graph_topology(target)[1]
    target.mkdir(parents=True, exist_ok=True)
    canonical = CanonicalDataManifest.model_validate_json(
        layout.canonical_manifest.read_text(encoding="utf-8")
    )
    split = SplitManifest.model_validate_json(
        (layout.manifests / "split.json").read_text(encoding="utf-8")
    )
    source_path = layout.source / entry.source.filename
    if entry.source.checksum.algorithm != "sha256":
        raise ValueError("the raw Nadig Jurkat pilot requires a frozen SHA-256 source")
    if sha256_file(source_path, chunk_size=8 * 1024 * 1024) != entry.source.checksum.value:
        raise ValueError("raw source H5AD differs from the frozen registry")
    anndata = importlib.import_module("anndata")
    source = anndata.read_h5ad(source_path)
    direct_top500 = _direct_top500(source, entry)
    if getattr(source, "isbacked", False):
        source.file.close()

    frozen = anndata.read_h5ad(layout.canonical_adata, backed="r")
    frozen_top500 = _rank_selected_hvgs(frozen, expected_count=5000)[:500]
    frozen.file.close()
    if direct_top500 != frozen_top500:
        raise ValueError("direct Top-500 does not equal the frozen dispersions_norm ranking")

    canonical_graph_genes = tuple(
        (layout.canonical / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    candidate_targets = _candidate_targets(split)
    missing_targets = sorted(set(candidate_targets) - set(canonical_graph_genes))
    if missing_targets:
        raise ValueError(f"candidate targets are absent from canonical graph: {missing_targets}")
    target_set = set(candidate_targets)
    direct_set = set(direct_top500)
    graph_gene_ids = direct_top500 + tuple(
        gene for gene in canonical_graph_genes if gene in target_set and gene not in direct_set
    )

    source_paths = verify_graph_source_checkout(official_checkout, source_registry)
    artifact_hashes: dict[Literal["go", "string"], str] = {}
    edge_counts: dict[Literal["go", "string"], int] = {}
    for source_name in ("go", "string"):
        edges, _, _ = _read_filtered_edges(
            source_paths[source_name],
            source_registry.sources[source_name],
            graph_gene_ids,
        )
        graph = prune_incoming_edges(
            source_name=source_name,
            gene_ids=graph_gene_ids,
            weighted_edges=edges,
            top_k=20,
        )
        artifact_path = target / f"{source_name}.npz"
        _atomic_graph(artifact_path, graph)
        artifact_hashes[source_name] = sha256_file(artifact_path)
        edge_counts[source_name] = len(graph.edges)
    atomic_text(target / "graph_gene_ids.txt", "\n".join(graph_gene_ids) + "\n")
    graph_hash = sha256_json(list(graph_gene_ids))
    manifest = ReducedGraphManifest(
        schema_version="recomputed-top500-graph-v1",
        dataset_id="nadig_jurkat",
        protocol_id=entry.protocol_id,
        canonical_data_sha256=canonical.canonical_adata_sha256,
        split_content_sha256=split.split_content_sha256,
        source_h5ad_sha256=sha256_file(source_path, chunk_size=8 * 1024 * 1024),
        source_registry_sha256=sha256_file(source_registry_path),
        hvg_method="scanpy.pp.highly_variable_genes",
        normalize_total=4000,
        log1p=True,
        requested_hvg_count=500,
        expression_gene_count=5000,
        direct_top500_gene_ids=list(direct_top500),
        direct_top500_gene_order_sha256=sha256_json(list(direct_top500)),
        frozen_rank_top500_gene_order_sha256=sha256_json(list(frozen_top500)),
        candidate_target_count=len(candidate_targets),
        graph_gene_ids=list(graph_gene_ids),
        graph_gene_order_sha256=graph_hash,
        graph_gene_count=len(graph_gene_ids),
        source_artifact_sha256=artifact_hashes,
        source_pruned_nonself_edge_count=edge_counts,
        topology_content_sha256=sha256_json(
            {"graph_gene_order_sha256": graph_hash, "sources": artifact_hashes}
        ),
        materialization_wall_ms=(time.perf_counter() - materialization_started) * 1000.0,
    )
    atomic_json(manifest_path, manifest.model_dump(mode="json"))
    load_reduced_graph_topology(target)
    return manifest


def load_reduced_graph_topology(
    root: str | Path,
) -> tuple[GraphTopology, ReducedGraphManifest]:
    """Load and fully hash-check an isolated reduced graph artifact."""

    source = Path(root)
    manifest = ReducedGraphManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    gene_ids = tuple((source / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines())
    if sha256_json(list(gene_ids)) != manifest.graph_gene_order_sha256:
        raise ValueError("reduced runtime graph axis differs from its manifest")
    graphs = {}
    for source_name in ("go", "string"):
        artifact = source / f"{source_name}.npz"
        if sha256_file(artifact) != manifest.source_artifact_sha256[source_name]:
            raise ValueError(f"reduced graph artifact hash mismatch: {source_name}")
        graphs[source_name] = _load_pruned_graph(
            artifact,
            source_name=source_name,
            gene_ids=gene_ids,
        )
    return GraphTopology(gene_ids=gene_ids, sources=graphs), manifest
