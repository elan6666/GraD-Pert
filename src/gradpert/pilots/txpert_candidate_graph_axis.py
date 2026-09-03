"""Receipt-backed TxPert candidate-gene runtime graph for the H4 ablation."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, model_validator

from gradpert.contracts.base import NonEmpty, Sha256, StrictManifest
from gradpert.contracts.manifests import CanonicalDataManifest, SplitManifest
from gradpert.data import DatasetLayout
from gradpert.data._io import atomic_json, atomic_text
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
from gradpert.pilots.graph_axis import _candidate_targets

TXPERT_PUBLIC_COMMIT: Final[Literal["08d82eea86746b044cf7531f4ec8c5f60e1cb73f"]] = (
    "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
)
TXPERT_CANDIDATE_GENE_SET_SHA256: Final[
    Literal["7e2be69a204b72349b793cc6723a5f88419f1ca6472ea5e28c5f7d623ee8e23d"]
] = "7e2be69a204b72349b793cc6723a5f88419f1ca6472ea5e28c5f7d623ee8e23d"
TXPERT_CANDIDATE_GENE_COUNT: Final[Literal[9853]] = 9853


class TxPertCandidateGraphManifest(StrictManifest):
    """Exact candidate-gene universe and pruned STRING+GO topology for H4."""

    schema_version: Literal["txpert-candidate-gene-graph-v1"]
    dataset_id: Literal["nadig_jurkat"]
    protocol_id: NonEmpty
    canonical_data_sha256: Sha256
    split_content_sha256: Sha256
    source_h5ad_sha256: Sha256
    source_registry_sha256: Sha256
    graph_axis_policy: Literal["txpert_candidate_gene_universe"]
    selection_method: Literal["frozen_txpert_gears_gene_set_order"]
    txpert_public_commit: Literal["08d82eea86746b044cf7531f4ec8c5f60e1cb73f"]
    candidate_gene_set_path: NonEmpty
    candidate_gene_set_sha256: Literal[
        "7e2be69a204b72349b793cc6723a5f88419f1ca6472ea5e28c5f7d623ee8e23d"
    ]
    requested_gene_count: Literal[9853]
    expression_gene_count: Literal[5000]
    candidate_gene_ids: list[NonEmpty]
    candidate_gene_order_sha256: Sha256
    candidate_target_ids: list[NonEmpty]
    candidate_target_order_sha256: Sha256
    graph_gene_ids: list[NonEmpty]
    graph_gene_order_sha256: Sha256
    graph_gene_count: Literal[9853]
    source_artifact_sha256: dict[Literal["go", "string"], Sha256]
    source_pruned_nonself_edge_count: dict[Literal["go", "string"], int]
    topology_content_sha256: Sha256
    top_k_incoming_per_source: Literal[20]
    control_graph_node_included: Literal[False]
    gene_feature_policy: Literal["learned_id"]
    materialization_wall_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def enforce_content(self) -> TxPertCandidateGraphManifest:
        if len(self.candidate_gene_ids) != self.requested_gene_count:
            raise ValueError("TxPert candidate-gene receipt count differs")
        if len(self.candidate_gene_ids) != len(set(self.candidate_gene_ids)):
            raise ValueError("TxPert candidate-gene receipt contains duplicates")
        if self.graph_gene_ids != self.candidate_gene_ids:
            raise ValueError("H4 graph axis must equal the frozen TxPert candidate order")
        if self.graph_gene_count != len(self.graph_gene_ids):
            raise ValueError("H4 graph count differs from its ordered axis")
        if sha256_json(self.candidate_gene_ids) != self.candidate_gene_order_sha256:
            raise ValueError("TxPert candidate-gene order hash differs")
        if sha256_json(self.graph_gene_ids) != self.graph_gene_order_sha256:
            raise ValueError("H4 graph-gene order hash differs")
        if sha256_json(self.candidate_target_ids) != self.candidate_target_order_sha256:
            raise ValueError("H4 perturbation-target order hash differs")
        missing_targets = sorted(set(self.candidate_target_ids) - set(self.graph_gene_ids))
        if missing_targets:
            raise ValueError(f"TxPert candidate universe omits targets: {missing_targets}")
        expected_topology = sha256_json(
            {
                "graph_gene_order_sha256": self.graph_gene_order_sha256,
                "sources": self.source_artifact_sha256,
            }
        )
        if expected_topology != self.topology_content_sha256:
            raise ValueError("H4 topology content hash differs")
        return self


def _read_txpert_candidate_genes(path: Path) -> tuple[str, ...]:
    if sha256_file(path) != TXPERT_CANDIDATE_GENE_SET_SHA256:
        raise ValueError("TxPert candidate-gene set SHA-256 differs from the frozen public file")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) != {"", "0"}:
        raise ValueError("TxPert candidate-gene CSV schema differs")
    expected_ids = list(range(TXPERT_CANDIDATE_GENE_COUNT))
    try:
        observed_ids = [int(row[""]) for row in rows]
        genes = tuple(row["0"] for row in rows)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("TxPert candidate-gene CSV content is malformed") from error
    if observed_ids != expected_ids:
        raise ValueError("TxPert candidate-gene IDs are not the exact ordered 0..9852 axis")
    if len(genes) != TXPERT_CANDIDATE_GENE_COUNT or any(not gene for gene in genes):
        raise ValueError("TxPert candidate-gene count/content differs")
    if len(set(genes)) != len(genes):
        raise ValueError("TxPert candidate-gene CSV contains duplicate symbols")
    return genes


def materialize_txpert_candidate_graph(
    *,
    entry: DatasetRegistryEntry,
    data_root: str | Path,
    destination: str | Path,
    candidate_gene_set_path: str | Path,
    source_registry_path: str | Path,
    source_registry: GraphSourceRegistry,
    official_checkout: str | Path,
) -> TxPertCandidateGraphManifest:
    """Build H4 without changing the canonical 5,000-gene expression axis."""

    if entry.dataset_id != "nadig_jurkat" or entry.source.semantics != "raw_single_cell":
        raise ValueError("the H4 candidate graph is frozen to raw Nadig Jurkat")
    started = time.perf_counter()
    layout = DatasetLayout(Path(data_root), entry.dataset_id, entry.protocol_id)
    target = Path(destination)
    canonical = CanonicalDataManifest.model_validate_json(
        layout.canonical_manifest.read_text(encoding="utf-8")
    )
    split = SplitManifest.model_validate_json(
        (layout.manifests / "split.json").read_text(encoding="utf-8")
    )
    source_path = layout.source / entry.source.filename
    if entry.source.checksum.algorithm != "sha256":
        raise ValueError("raw Nadig Jurkat source requires a frozen SHA-256")
    source_h5ad_sha256 = sha256_file(source_path, chunk_size=8 * 1024 * 1024)
    if source_h5ad_sha256 != entry.source.checksum.value:
        raise ValueError("raw source H5AD differs from the frozen registry")
    candidate_path = Path(candidate_gene_set_path).resolve(strict=True)
    candidate_genes = _read_txpert_candidate_genes(candidate_path)
    candidate_targets = _candidate_targets(split)
    missing_targets = sorted(set(candidate_targets) - set(candidate_genes))
    if missing_targets:
        raise ValueError(f"TxPert candidate universe omits Nadig targets: {missing_targets}")
    source_registry_sha256 = sha256_file(source_registry_path)

    if target.exists() and any(target.iterdir()):
        _, existing = load_txpert_candidate_graph(target)
        expected = {
            "canonical_data_sha256": canonical.canonical_adata_sha256,
            "split_content_sha256": split.split_content_sha256,
            "source_h5ad_sha256": source_h5ad_sha256,
            "source_registry_sha256": source_registry_sha256,
            "candidate_gene_set_path": str(candidate_path),
        }
        differing = [name for name, value in expected.items() if getattr(existing, name) != value]
        if differing:
            raise ValueError("existing H4 graph lineage differs: " + ", ".join(differing))
        return existing
    target.mkdir(parents=True, exist_ok=True)

    source_paths = verify_graph_source_checkout(official_checkout, source_registry)
    artifact_hashes: dict[Literal["go", "string"], str] = {}
    edge_counts: dict[Literal["go", "string"], int] = {}
    for source_name in ("go", "string"):
        edges, _, _ = _read_filtered_edges(
            source_paths[source_name], source_registry.sources[source_name], candidate_genes
        )
        graph = prune_incoming_edges(
            source_name=source_name,
            gene_ids=candidate_genes,
            weighted_edges=edges,
            top_k=20,
        )
        artifact_path = target / f"{source_name}.npz"
        _atomic_graph(artifact_path, graph)
        artifact_hashes[source_name] = sha256_file(artifact_path)
        edge_counts[source_name] = len(graph.edges)

    graph_hash = sha256_json(list(candidate_genes))
    atomic_text(target / "graph_gene_ids.txt", "\n".join(candidate_genes) + "\n")
    manifest = TxPertCandidateGraphManifest(
        schema_version="txpert-candidate-gene-graph-v1",
        dataset_id="nadig_jurkat",
        protocol_id=entry.protocol_id,
        canonical_data_sha256=canonical.canonical_adata_sha256,
        split_content_sha256=split.split_content_sha256,
        source_h5ad_sha256=source_h5ad_sha256,
        source_registry_sha256=source_registry_sha256,
        graph_axis_policy="txpert_candidate_gene_universe",
        selection_method="frozen_txpert_gears_gene_set_order",
        txpert_public_commit=TXPERT_PUBLIC_COMMIT,
        candidate_gene_set_path=str(candidate_path),
        candidate_gene_set_sha256=TXPERT_CANDIDATE_GENE_SET_SHA256,
        requested_gene_count=TXPERT_CANDIDATE_GENE_COUNT,
        expression_gene_count=5000,
        candidate_gene_ids=list(candidate_genes),
        candidate_gene_order_sha256=graph_hash,
        candidate_target_ids=list(candidate_targets),
        candidate_target_order_sha256=sha256_json(list(candidate_targets)),
        graph_gene_ids=list(candidate_genes),
        graph_gene_order_sha256=graph_hash,
        graph_gene_count=TXPERT_CANDIDATE_GENE_COUNT,
        source_artifact_sha256=artifact_hashes,
        source_pruned_nonself_edge_count=edge_counts,
        topology_content_sha256=sha256_json(
            {"graph_gene_order_sha256": graph_hash, "sources": artifact_hashes}
        ),
        top_k_incoming_per_source=20,
        control_graph_node_included=False,
        gene_feature_policy="learned_id",
        materialization_wall_ms=(time.perf_counter() - started) * 1000.0,
    )
    atomic_json(target / "manifest.json", manifest.model_dump(mode="json"))
    load_txpert_candidate_graph(target)
    return manifest


def load_txpert_candidate_graph(
    root: str | Path,
) -> tuple[GraphTopology, TxPertCandidateGraphManifest]:
    source = Path(root)
    manifest = TxPertCandidateGraphManifest.model_validate_json(
        (source / "manifest.json").read_text(encoding="utf-8")
    )
    candidate_path = Path(manifest.candidate_gene_set_path).resolve(strict=True)
    if sha256_file(candidate_path) != manifest.candidate_gene_set_sha256:
        raise ValueError("live TxPert candidate-gene source differs from the H4 manifest")
    gene_ids = tuple((source / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines())
    if list(gene_ids) != manifest.graph_gene_ids:
        raise ValueError("H4 runtime graph axis differs from its manifest")
    graphs = {}
    for source_name in ("go", "string"):
        artifact = source / f"{source_name}.npz"
        if sha256_file(artifact) != manifest.source_artifact_sha256[source_name]:
            raise ValueError(f"H4 graph artifact hash mismatch: {source_name}")
        graphs[source_name] = _load_pruned_graph(
            artifact, source_name=source_name, gene_ids=gene_ids
        )
    return GraphTopology(gene_ids=gene_ids, sources=graphs), manifest
