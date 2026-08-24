"""Independent deterministic Top-K incoming pruning for one graph source."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class DirectedEdge:
    source: int
    target: int
    weight: float


@dataclass(frozen=True)
class PrunedSourceGraph:
    source_name: str
    n_nodes: int
    gene_ids: tuple[str, ...]
    edges: tuple[DirectedEdge, ...]
    top_k_incoming: int


def prune_incoming_edges(
    *,
    source_name: str,
    gene_ids: Sequence[str],
    weighted_edges: Iterable[tuple[str, str, float]],
    top_k: int = 20,
) -> PrunedSourceGraph:
    """Filter, deduplicate, and prune one source without adding self-loops."""

    if not source_name:
        raise ValueError("source_name must be non-empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ordered_gene_ids = tuple(gene_ids)
    if not ordered_gene_ids or any(not gene_id for gene_id in ordered_gene_ids):
        raise ValueError("gene_ids must be non-empty strings")
    if len(ordered_gene_ids) != len(set(ordered_gene_ids)):
        raise ValueError("gene_ids must be unique")
    node_by_gene = {gene_id: index for index, gene_id in enumerate(ordered_gene_ids)}

    deduplicated: dict[tuple[int, int], float] = {}
    for source_gene, target_gene, weight in weighted_edges:
        if source_gene not in node_by_gene or target_gene not in node_by_gene:
            continue
        source = node_by_gene[source_gene]
        target = node_by_gene[target_gene]
        if source == target:
            continue
        numeric_weight = float(weight)
        key = (source, target)
        previous = deduplicated.get(key)
        if previous is None or numeric_weight > previous:
            deduplicated[key] = numeric_weight

    incoming: dict[int, list[DirectedEdge]] = defaultdict(list)
    for (source, target), weight in deduplicated.items():
        incoming[target].append(DirectedEdge(source=source, target=target, weight=weight))

    retained: list[DirectedEdge] = []
    for target in sorted(incoming):
        ranked = sorted(
            incoming[target],
            key=lambda edge: (
                -edge.weight,
                ordered_gene_ids[edge.source],
                ordered_gene_ids[edge.target],
            ),
        )
        retained.extend(ranked[:top_k])
    retained.sort(key=lambda edge: (edge.target, edge.source))
    return PrunedSourceGraph(
        source_name=source_name,
        n_nodes=len(ordered_gene_ids),
        gene_ids=ordered_gene_ids,
        edges=tuple(retained),
        top_k_incoming=top_k,
    )
