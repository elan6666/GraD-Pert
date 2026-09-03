"""Explicit, receipt-backed performance pilots outside the frozen benchmark matrix."""

import json
from pathlib import Path

from gradpert.graphs import GraphTopology
from gradpert.pilots.graph_axis import (
    ReducedGraphManifest,
    load_reduced_graph_topology,
    materialize_recomputed_top500_graph,
)
from gradpert.pilots.txpert_candidate_graph_axis import (
    TXPERT_CANDIDATE_GENE_COUNT,
    TXPERT_CANDIDATE_GENE_SET_SHA256,
    TxPertCandidateGraphManifest,
    load_txpert_candidate_graph,
    materialize_txpert_candidate_graph,
)
from gradpert.pilots.vnext_graph_axis import (
    GenePTAvailabilityReceipt,
    GenePTSeedAvailabilityReceipt,
    VNextGraphManifest,
    load_vnext_graph_topology,
    materialize_genept_vnext_graph,
    materialize_vnext_hvg512_graph,
    preflight_genept_seed_vnext,
)


def load_vnext_runtime_graph_topology(
    root: str | Path,
) -> tuple[GraphTopology, VNextGraphManifest | TxPertCandidateGraphManifest]:
    """Dispatch a sealed H graph without weakening either manifest schema."""

    source = Path(root)
    payload = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") == "txpert-candidate-gene-graph-v1":
        return load_txpert_candidate_graph(source)
    return load_vnext_graph_topology(source)


__all__ = [
    "TXPERT_CANDIDATE_GENE_COUNT",
    "TXPERT_CANDIDATE_GENE_SET_SHA256",
    "GenePTAvailabilityReceipt",
    "GenePTSeedAvailabilityReceipt",
    "ReducedGraphManifest",
    "TxPertCandidateGraphManifest",
    "VNextGraphManifest",
    "load_reduced_graph_topology",
    "load_txpert_candidate_graph",
    "load_vnext_graph_topology",
    "load_vnext_runtime_graph_topology",
    "materialize_genept_vnext_graph",
    "materialize_recomputed_top500_graph",
    "materialize_txpert_candidate_graph",
    "materialize_vnext_hvg512_graph",
    "preflight_genept_seed_vnext",
]
