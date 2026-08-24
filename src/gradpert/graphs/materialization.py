"""Receipt-backed materialization of dataset-specific GO and STRING graphs."""

from __future__ import annotations

import csv
import importlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gradpert.contracts import (
    CanonicalDataManifest,
    DatasetGraphManifest,
    GraphSourceArtifact,
    SplitManifest,
)
from gradpert.graphs.pruning import DirectedEdge, PrunedSourceGraph, prune_incoming_edges
from gradpert.graphs.registry import GraphSourceFile, GraphSourceRegistry
from gradpert.graphs.views import GraphTopology
from gradpert.hashing import sha256_file, sha256_json


@dataclass(frozen=True)
class DatasetGraphLayout:
    data_root: Path
    dataset_id: str
    protocol_id: str

    @property
    def dataset_root(self) -> Path:
        return self.data_root / self.dataset_id / self.protocol_id

    @property
    def canonical_root(self) -> Path:
        return self.dataset_root / "canonical"

    @property
    def canonical_manifest(self) -> Path:
        return self.dataset_root / "manifests" / "canonical.json"

    @property
    def split_manifest(self) -> Path:
        return self.dataset_root / "manifests" / "split.json"

    @property
    def root(self) -> Path:
        return self.dataset_root / "graphs"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def coverage_root(self) -> Path:
        return self.root / "graph_coverage"

    def source_artifact(self, source_name: str) -> Path:
        return self.root / f"{source_name}.npz"


def _read_json(path: Path) -> object:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required graph input must be a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_graph(path: Path, graph: PrunedSourceGraph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        edge_index = np.asarray(
            [
                [edge.source for edge in graph.edges],
                [edge.target for edge in graph.edges],
            ],
            dtype=np.int64,
        )
        edge_weight = np.asarray([edge.weight for edge in graph.edges], dtype=np.float64)
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _git(checkout_root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(checkout_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def verify_graph_source_checkout(
    checkout_root: str | Path,
    registry: GraphSourceRegistry,
) -> dict[str, Path]:
    root = Path(checkout_root).resolve(strict=True)
    if not root.is_dir() or not (root / ".git").exists():
        raise ValueError("graph source checkout must be a Git worktree")
    observed_commit = _git(root, "rev-parse", "HEAD")
    if observed_commit != registry.commit:
        raise ValueError(
            f"graph source checkout commit mismatch: {observed_commit} != {registry.commit}"
        )
    if _git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("graph source checkout must be clean")
    license_path = root / registry.license_notice_path
    if not license_path.is_file() or license_path.is_symlink():
        raise ValueError("graph source checkout lacks its frozen license notice")
    paths: dict[str, Path] = {}
    for source_name, source in registry.sources.items():
        path = root.joinpath(*source.relative_path.split("/"))
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"graph source file is missing: {source_name}")
        if path.stat().st_size != source.size_bytes:
            raise ValueError(f"graph source file size mismatch: {source_name}")
        if sha256_file(path, chunk_size=8 * 1024 * 1024) != source.sha256:
            raise ValueError(f"graph source file checksum mismatch: {source_name}")
        paths[source_name] = path
    return paths


def _read_filtered_edges(
    path: Path,
    source: GraphSourceFile,
    gene_ids: Sequence[str],
) -> tuple[list[tuple[str, str, float]], set[str], int]:
    try:
        pd = importlib.import_module("pandas")
    except ImportError as error:  # pragma: no cover - server data environment
        raise RuntimeError("pandas is required for graph materialization") from error
    columns = [source.source_column, source.target_column, source.weight_column]
    if source.format == "csv":
        frame = pd.read_csv(path, usecols=columns)
    else:
        try:
            frame = pd.read_parquet(path, columns=columns)
        except ImportError as error:  # pragma: no cover - server data environment
            raise RuntimeError(
                "pyarrow is required by the data extra for STRING Parquet"
            ) from error
    if list(frame.columns) != columns:
        raise ValueError("graph source columns differ from the frozen registry order")
    if bool(frame.isna().any().any()):
        raise ValueError("graph source contains null endpoint or weight values")
    source_values = frame[source.source_column].astype(str)
    target_values = frame[source.target_column].astype(str)
    nonempty = (source_values.str.len() > 0) & (target_values.str.len() > 0)
    dropped_empty_count = int((~nonempty).sum())
    if dropped_empty_count != source.expected_empty_endpoint_rows:
        raise ValueError(
            "graph source empty-endpoint count differs from the frozen registry: "
            f"{dropped_empty_count} != {source.expected_empty_endpoint_rows}"
        )
    weights = np.asarray(frame[source.weight_column], dtype=np.float64)
    if not np.isfinite(weights).all():
        raise ValueError("graph source contains non-finite weights")
    genes = set(gene_ids)
    mask = (
        nonempty
        & source_values.isin(genes)
        & target_values.isin(genes)
        & (source_values != target_values)
    )
    filtered_sources = source_values[mask].tolist()
    filtered_targets = target_values[mask].tolist()
    filtered_weights = weights[np.asarray(mask, dtype=bool)]
    edges = [
        (str(source_gene), str(target_gene), float(weight))
        for source_gene, target_gene, weight in zip(
            filtered_sources,
            filtered_targets,
            filtered_weights,
            strict=True,
        )
    ]
    covered = set(filtered_sources) | set(filtered_targets)
    return edges, covered, dropped_empty_count


def _candidate_targets(split: SplitManifest, graph_gene_ids: Sequence[str]) -> tuple[str, ...]:
    targets = {
        component
        for condition in [
            *split.train_conditions,
            *split.val_conditions,
            *split.test_conditions,
        ]
        for component in condition.split("+")
        if component != split.control_condition_id
    }
    missing = sorted(targets - set(graph_gene_ids))
    if missing:
        raise ValueError(f"candidate perturbation targets are absent from graph axis: {missing}")
    return tuple(sorted(targets))


def _degree_rows(
    gene_ids: Sequence[str],
    graphs: Mapping[str, PrunedSourceGraph],
    candidate_targets: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_name in sorted(graphs):
        graph = graphs[source_name]
        incoming = np.zeros(graph.n_nodes, dtype=np.int64)
        outgoing = np.zeros(graph.n_nodes, dtype=np.int64)
        for edge in graph.edges:
            outgoing[edge.source] += 1
            incoming[edge.target] += 1
        for node_id, gene_id in enumerate(gene_ids):
            rows.append(
                {
                    "gene_id": gene_id,
                    "source": source_name,
                    "incoming_nonself": int(incoming[node_id]),
                    "outgoing_nonself": int(outgoing[node_id]),
                    "self_loop_only": bool(incoming[node_id] + outgoing[node_id] == 0),
                    "candidate_target": gene_id in candidate_targets,
                }
            )
    return rows


def _conventional_artifact_path(layout: DatasetGraphLayout, filename: str) -> str:
    return str(Path("data") / layout.dataset_id / layout.protocol_id / "graphs" / filename)


def materialize_dataset_graphs(
    *,
    dataset_id: str,
    protocol_id: str,
    data_root: str | Path,
    source_registry_path: str | Path,
    source_registry: GraphSourceRegistry,
    official_checkout: str | Path,
) -> DatasetGraphManifest:
    """Build and seal one dataset's independent Top-20 GO/STRING artifacts."""

    layout = DatasetGraphLayout(Path(data_root), dataset_id, protocol_id)
    if layout.manifest.is_file():
        return verify_dataset_graphs(
            dataset_id=dataset_id,
            protocol_id=protocol_id,
            data_root=data_root,
            source_registry_path=source_registry_path,
            source_registry=source_registry,
            official_checkout=official_checkout,
        )
    canonical = CanonicalDataManifest.model_validate(_read_json(layout.canonical_manifest))
    split = SplitManifest.model_validate(_read_json(layout.split_manifest))
    if (canonical.dataset_id, canonical.protocol_id) != (dataset_id, protocol_id):
        raise ValueError("canonical data identity differs from graph request")
    if (split.dataset_id, split.protocol_id) != (dataset_id, protocol_id):
        raise ValueError("split identity differs from graph request")
    graph_gene_ids = tuple(
        (layout.canonical_root / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    if (
        len(graph_gene_ids) != canonical.n_graph_genes
        or sha256_json(list(graph_gene_ids)) != canonical.graph_gene_order_sha256
    ):
        raise ValueError("canonical graph gene axis no longer matches its manifest")
    candidate_targets = _candidate_targets(split, graph_gene_ids)
    source_paths = verify_graph_source_checkout(official_checkout, source_registry)
    graphs: dict[str, PrunedSourceGraph] = {}
    covered_by_source: dict[str, set[str]] = {}
    filtered_counts: dict[str, int] = {}
    dropped_empty_counts: dict[str, int] = {}
    for source_name in ("go", "string"):
        edges, covered, dropped_empty_count = _read_filtered_edges(
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
        _atomic_graph(layout.source_artifact(source_name), graph)
        graphs[source_name] = graph
        covered_by_source[source_name] = covered
        filtered_counts[source_name] = len(edges)
        dropped_empty_counts[source_name] = dropped_empty_count

    candidate_set = set(candidate_targets)
    gene_set = set(graph_gene_ids)
    missing_by_source = {
        source_name: gene_set - covered for source_name, covered in covered_by_source.items()
    }
    both_missing = missing_by_source["go"] & missing_by_source["string"]
    both_missing_targets = tuple(sorted(candidate_set & both_missing))
    missing_rows = [
        {
            "gene_id": gene_id,
            "source": source_name,
            "candidate_target": gene_id in candidate_set,
        }
        for source_name in ("go", "string")
        for gene_id in sorted(missing_by_source[source_name])
    ]
    isolated_rows = _degree_rows(graph_gene_ids, graphs, candidate_set)
    coverage_path = layout.coverage_root / "graph_coverage.json"
    missing_path = layout.coverage_root / "missing_genes.csv"
    isolated_path = layout.coverage_root / "isolated_genes.csv"
    coverage_payload: dict[str, object] = {
        "schema_version": "graph-coverage-v1",
        "dataset_id": dataset_id,
        "protocol_id": protocol_id,
        "graph_gene_count": len(graph_gene_ids),
        "candidate_target_count": len(candidate_targets),
        "candidate_targets": list(candidate_targets),
        "source_coverage": {
            source_name: {
                "covered_gene_count": len(covered_by_source[source_name]),
                "missing_gene_count": len(missing_by_source[source_name]),
                "dropped_empty_endpoint_row_count": dropped_empty_counts[source_name],
                "filtered_nonself_edge_count": filtered_counts[source_name],
                "pruned_nonself_edge_count": len(graphs[source_name].edges),
                "top_k_incoming": 20,
            }
            for source_name in ("go", "string")
        },
        "both_sources_missing_gene_count": len(both_missing),
        "both_sources_missing_target_count": len(both_missing_targets),
        "both_sources_missing_targets": list(both_missing_targets),
    }
    _atomic_json(coverage_path, coverage_payload)
    _atomic_csv(missing_path, ("gene_id", "source", "candidate_target"), missing_rows)
    _atomic_csv(
        isolated_path,
        (
            "gene_id",
            "source",
            "incoming_nonself",
            "outgoing_nonself",
            "self_loop_only",
            "candidate_target",
        ),
        isolated_rows,
    )
    source_artifacts = [
        GraphSourceArtifact(
            source_name=source_name,
            upstream_relative_path=source_registry.sources[source_name].relative_path,
            upstream_file_sha256=source_registry.sources[source_name].sha256,
            upstream_size_bytes=source_registry.sources[source_name].size_bytes,
            dropped_empty_endpoint_row_count=dropped_empty_counts[source_name],
            filtered_nonself_edge_count=filtered_counts[source_name],
            pruned_nonself_edge_count=len(graphs[source_name].edges),
            covered_gene_count=len(covered_by_source[source_name]),
            artifact_path=_conventional_artifact_path(layout, f"{source_name}.npz"),
            artifact_sha256=sha256_file(layout.source_artifact(source_name)),
        )
        for source_name in ("go", "string")
    ]
    topology_content_sha256 = sha256_json(
        {
            "graph_gene_order_sha256": canonical.graph_gene_order_sha256,
            "sources": {item.source_name: item.artifact_sha256 for item in source_artifacts},
        }
    )
    manifest = DatasetGraphManifest(
        schema_version="dataset-graph-v1",
        dataset_id=canonical.dataset_id,
        protocol_id=canonical.protocol_id,
        state="graph_ready",
        source_registry_sha256=sha256_file(source_registry_path),
        source_repository=source_registry.repository,
        source_commit=source_registry.commit,
        canonical_data_sha256=canonical.canonical_adata_sha256,
        graph_gene_order_sha256=canonical.graph_gene_order_sha256,
        graph_gene_count=canonical.n_graph_genes,
        candidate_target_count=len(candidate_targets),
        both_sources_missing_target_count=len(both_missing_targets),
        topology_content_sha256=topology_content_sha256,
        sources=source_artifacts,
        coverage_report_path=_conventional_artifact_path(
            layout, "graph_coverage/graph_coverage.json"
        ),
        coverage_report_sha256=sha256_file(coverage_path),
        missing_genes_path=_conventional_artifact_path(layout, "graph_coverage/missing_genes.csv"),
        missing_genes_sha256=sha256_file(missing_path),
        isolated_genes_path=_conventional_artifact_path(
            layout, "graph_coverage/isolated_genes.csv"
        ),
        isolated_genes_sha256=sha256_file(isolated_path),
    )
    _atomic_json(layout.manifest, manifest.model_dump(mode="json"))
    return verify_dataset_graphs(
        dataset_id=dataset_id,
        protocol_id=protocol_id,
        data_root=data_root,
        source_registry_path=source_registry_path,
        source_registry=source_registry,
        official_checkout=official_checkout,
    )


def _load_pruned_graph(
    path: Path,
    *,
    source_name: str,
    gene_ids: tuple[str, ...],
) -> PrunedSourceGraph:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"graph artifact must be a regular file: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"edge_index", "edge_weight"}:
            raise ValueError("graph artifact contains unexpected arrays")
        edge_index = np.asarray(payload["edge_index"])
        edge_weight = np.asarray(payload["edge_weight"])
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("graph edge_index must have shape [2,E]")
    if edge_weight.shape != (edge_index.shape[1],):
        raise ValueError("graph edge weights must align with edge_index")
    if edge_index.dtype.kind not in {"i", "u"} or not np.isfinite(edge_weight).all():
        raise ValueError("graph artifact contains invalid index/weight values")
    if edge_index.size and (int(edge_index.min()) < 0 or int(edge_index.max()) >= len(gene_ids)):
        raise ValueError("graph artifact contains an out-of-range node ID")
    edges = tuple(
        DirectedEdge(int(source), int(target), float(weight))
        for source, target, weight in zip(edge_index[0], edge_index[1], edge_weight, strict=True)
    )
    if tuple(sorted(edges, key=lambda edge: (edge.target, edge.source))) != edges:
        raise ValueError("graph artifact edges are not in canonical order")
    if any(edge.source == edge.target for edge in edges):
        raise ValueError("base graph artifact must not contain self-loops")
    return PrunedSourceGraph(
        source_name=source_name,
        n_nodes=len(gene_ids),
        gene_ids=gene_ids,
        edges=edges,
        top_k_incoming=20,
    )


def verify_dataset_graphs(
    *,
    dataset_id: str,
    protocol_id: str,
    data_root: str | Path,
    source_registry_path: str | Path,
    source_registry: GraphSourceRegistry,
    official_checkout: str | Path,
) -> DatasetGraphManifest:
    """Verify upstream files, every small receipt, and both graph artifacts."""

    layout = DatasetGraphLayout(Path(data_root), dataset_id, protocol_id)
    manifest = DatasetGraphManifest.model_validate(_read_json(layout.manifest))
    canonical = CanonicalDataManifest.model_validate(_read_json(layout.canonical_manifest))
    if (manifest.dataset_id, manifest.protocol_id) != (dataset_id, protocol_id):
        raise ValueError("dataset graph manifest identity differs from its directory")
    if manifest.canonical_data_sha256 != canonical.canonical_adata_sha256:
        raise ValueError("dataset graph and canonical data hashes differ")
    if manifest.source_registry_sha256 != sha256_file(source_registry_path):
        raise ValueError("dataset graph source registry hash differs")
    verify_graph_source_checkout(official_checkout, source_registry)
    gene_ids = tuple(
        (layout.canonical_root / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    if len(gene_ids) != manifest.graph_gene_count or sha256_json(list(gene_ids)) != (
        manifest.graph_gene_order_sha256
    ):
        raise ValueError("dataset graph gene order differs")
    source_by_name = {source.source_name: source for source in manifest.sources}
    graphs: dict[str, PrunedSourceGraph] = {}
    for source_name in ("go", "string"):
        artifact = source_by_name[source_name]
        path = layout.source_artifact(source_name)
        if sha256_file(path) != artifact.artifact_sha256:
            raise ValueError(f"dataset graph artifact hash mismatch: {source_name}")
        graph = _load_pruned_graph(path, source_name=source_name, gene_ids=gene_ids)
        if len(graph.edges) != artifact.pruned_nonself_edge_count:
            raise ValueError(f"dataset graph edge count mismatch: {source_name}")
        graphs[source_name] = graph
    coverage_paths = {
        "coverage": layout.coverage_root / "graph_coverage.json",
        "missing": layout.coverage_root / "missing_genes.csv",
        "isolated": layout.coverage_root / "isolated_genes.csv",
    }
    expected_hashes = {
        "coverage": manifest.coverage_report_sha256,
        "missing": manifest.missing_genes_sha256,
        "isolated": manifest.isolated_genes_sha256,
    }
    for name, path in coverage_paths.items():
        if sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"dataset graph coverage receipt hash mismatch: {name}")
    observed_topology_hash = sha256_json(
        {
            "graph_gene_order_sha256": manifest.graph_gene_order_sha256,
            "sources": {name: source_by_name[name].artifact_sha256 for name in ("go", "string")},
        }
    )
    if observed_topology_hash != manifest.topology_content_sha256:
        raise ValueError("dataset graph topology content hash differs")
    GraphTopology(gene_ids=gene_ids, sources=graphs)
    return manifest


def load_dataset_graph_topology(
    *,
    dataset_id: str,
    protocol_id: str,
    data_root: str | Path,
) -> GraphTopology:
    """Load a previously verified topology without requiring the upstream checkout."""

    layout = DatasetGraphLayout(Path(data_root), dataset_id, protocol_id)
    manifest = DatasetGraphManifest.model_validate(_read_json(layout.manifest))
    gene_ids = tuple(
        (layout.canonical_root / "graph_gene_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    if sha256_json(list(gene_ids)) != manifest.graph_gene_order_sha256:
        raise ValueError("runtime graph gene axis differs from its sealed manifest")
    source_by_name = {source.source_name: source for source in manifest.sources}
    graphs: dict[str, PrunedSourceGraph] = {}
    for source_name in ("go", "string"):
        path = layout.source_artifact(source_name)
        if sha256_file(path) != source_by_name[source_name].artifact_sha256:
            raise ValueError(f"runtime graph artifact hash mismatch: {source_name}")
        graphs[source_name] = _load_pruned_graph(
            path,
            source_name=source_name,
            gene_ids=gene_ids,
        )
    return GraphTopology(gene_ids=gene_ids, sources=graphs)
