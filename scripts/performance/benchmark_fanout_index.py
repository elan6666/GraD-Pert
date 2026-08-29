"""Measure reference versus preindexed Fanout view construction locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections.abc import Mapping
from typing import Any

from gradpert.graphs import (
    GraphTopology,
    build_incoming_edge_index,
    build_training_graph_views,
)
from gradpert.graphs.pruning import DirectedEdge, PrunedSourceGraph


def _topology() -> GraphTopology:
    node_count = 2_809
    gene_ids = tuple(f"G{node_id}" for node_id in range(node_count))

    def source(name: str, incoming_degree: int) -> PrunedSourceGraph:
        edges = tuple(
            DirectedEdge(
                source=(target - offset) % node_count,
                target=target,
                weight=float(offset),
            )
            for target in range(node_count)
            for offset in range(1, incoming_degree + 1)
        )
        return PrunedSourceGraph(
            source_name=name,
            n_nodes=node_count,
            gene_ids=gene_ids,
            edges=edges,
            top_k_incoming=20,
        )

    return GraphTopology(
        gene_ids=gene_ids,
        sources={"string": source("string", 18), "go": source("go", 14)},
        active_sources=("string", "go"),
    )


def _local_view_sha256(views: Any) -> str:
    hasher = hashlib.sha256()
    for condition_id in sorted(views.locals_by_condition):
        for view in views.locals_by_condition[condition_id]:
            hasher.update(condition_id.encode())
            hasher.update(view.view_id.encode())
            hasher.update(str(view.node_ids).encode())
            for source_name in sorted(view.edges_by_source):
                hasher.update(source_name.encode())
                hasher.update(
                    str(
                        tuple(
                            (edge.source, edge.target, edge.weight)
                            for edge in view.edges_by_source[source_name]
                        )
                    ).encode()
                )
            hasher.update(str(view.masked_node_ids).encode())
            hasher.update(str(view.masked_anchor_ids).encode())
            hasher.update(str(view.warnings).encode())
    return hasher.hexdigest()


def _measure(
    topology: GraphTopology,
    *,
    condition_count: int,
    node_budget: int,
    repeats: int,
    incoming_edges: Mapping[int, tuple[tuple[str, DirectedEdge], ...]] | None,
) -> tuple[list[float], str]:
    anchors = {
        f"condition-{index}": ((index * 443) % topology.n_nodes,)
        for index in range(condition_count)
    }
    timings = []
    final = None
    for repeat in range(repeats + 1):
        started = time.perf_counter()
        result = build_training_graph_views(
            topology,
            anchors_by_condition=anchors,
            heldout_target_ids=(),
            run_seed=1,
            global_step=repeat,
            local_count=8,
            local_node_budget=node_budget,
            local_builder="fanout",
            local_fanouts=(20, 10, 5, 5),
            local_anchor_mask_count=0,
            incoming_edges=incoming_edges,
        )
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if repeat:
            timings.append(elapsed_ms)
        final = result
    if final is None:
        raise RuntimeError("benchmark produced no view result")
    return timings, _local_view_sha256(final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    topology = _topology()
    incoming_edges = build_incoming_edge_index(topology)
    rows = []
    for condition_count in (1, 5, 8):
        for node_budget in (256, 512):
            reference, reference_sha = _measure(
                topology,
                condition_count=condition_count,
                node_budget=node_budget,
                repeats=args.repeats,
                incoming_edges=None,
            )
            optimized, optimized_sha = _measure(
                topology,
                condition_count=condition_count,
                node_budget=node_budget,
                repeats=args.repeats,
                incoming_edges=incoming_edges,
            )
            reference_median = statistics.median(reference)
            optimized_median = statistics.median(optimized)
            rows.append(
                {
                    "condition_count": condition_count,
                    "local_view_count": condition_count * 8,
                    "node_budget": node_budget,
                    "reference_ms": reference,
                    "optimized_ms": optimized,
                    "reference_median_ms": reference_median,
                    "optimized_median_ms": optimized_median,
                    "median_speedup": reference_median / optimized_median,
                    "median_reduction_pct": 100.0 * (1.0 - optimized_median / reference_median),
                    "exact_local_view_sha256_equal": reference_sha == optimized_sha,
                    "local_view_sha256": reference_sha,
                }
            )
    print(
        json.dumps(
            {
                "schema_version": "fanout-index-microbenchmark-v1",
                "topology_node_count": topology.n_nodes,
                "topology_nonself_edge_count": sum(
                    len(graph.edges) for graph in topology.sources.values()
                ),
                "repeats_after_warmup": args.repeats,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
