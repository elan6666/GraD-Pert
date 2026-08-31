"""Benchmark exact reference versus preindexed RingInduced view construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from gradpert.graphs import (
    GraphTopology,
    build_induced_edge_index,
    build_training_graph_views,
)
from gradpert.graphs.views import InducedEdgeIndex
from gradpert.pilots.vnext_graph_axis import load_vnext_graph_topology


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
            hasher.update(str(view.masked_anchor_ids).encode())
            hasher.update(str(view.warnings).encode())
    return hasher.hexdigest()


def _build(
    topology: GraphTopology,
    *,
    condition_count: int,
    local_count: int,
    node_budget: int,
    induced_edges: InducedEdgeIndex | None,
) -> Any:
    anchors = {
        f"condition-{index}": ((index * 443) % topology.n_nodes,)
        for index in range(condition_count)
    }
    return build_training_graph_views(
        topology,
        anchors_by_condition=anchors,
        heldout_target_ids=(),
        run_seed=1,
        global_step=31,
        local_count=local_count,
        local_node_budget=node_budget,
        local_builder="ring_induced",
        local_anchor_mask_count=0,
        induced_edges=induced_edges,
    )


def _time(call: Any) -> tuple[float, Any]:
    started = time.perf_counter()
    result = call()
    return (time.perf_counter() - started) * 1_000.0, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--condition-count", type=int, default=8)
    parser.add_argument("--local-count", type=int, default=4)
    parser.add_argument("--node-budget", type=int, default=1404)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.condition_count, args.local_count, args.node_budget, args.repeats) <= 0:
        raise SystemExit("benchmark counts and budget must be positive")

    topology, manifest = load_vnext_graph_topology(args.graph_root)
    induced_edges = build_induced_edge_index(topology)
    reference_call = lambda: _build(  # noqa: E731
        topology,
        condition_count=args.condition_count,
        local_count=args.local_count,
        node_budget=args.node_budget,
        induced_edges=None,
    )
    indexed_call = lambda: _build(  # noqa: E731
        topology,
        condition_count=args.condition_count,
        local_count=args.local_count,
        node_budget=args.node_budget,
        induced_edges=induced_edges,
    )
    reference_call()
    indexed_call()
    reference_ms: list[float] = []
    indexed_ms: list[float] = []
    reference_result = indexed_result = None
    for repeat in range(args.repeats):
        ordered = (
            ((reference_ms, reference_call), (indexed_ms, indexed_call))
            if repeat % 2 == 0
            else ((indexed_ms, indexed_call), (reference_ms, reference_call))
        )
        for values, call in ordered:
            elapsed, result = _time(call)
            values.append(elapsed)
            if call is reference_call:
                reference_result = result
            else:
                indexed_result = result
    if reference_result is None or indexed_result is None:
        raise RuntimeError("benchmark produced no views")
    reference_sha = _local_view_sha256(reference_result)
    indexed_sha = _local_view_sha256(indexed_result)
    reference_median = statistics.median(reference_ms)
    indexed_median = statistics.median(indexed_ms)
    print(
        json.dumps(
            {
                "schema_version": "ring-induced-index-microbenchmark-v1",
                "topology_node_count": topology.n_nodes,
                "topology_content_sha256": manifest.topology_content_sha256,
                "condition_count": args.condition_count,
                "local_count": args.local_count,
                "node_budget": args.node_budget,
                "repeats_after_warmup": args.repeats,
                "reference_ms": reference_ms,
                "indexed_ms": indexed_ms,
                "reference_median_ms": reference_median,
                "indexed_median_ms": indexed_median,
                "median_speedup": reference_median / indexed_median,
                "median_reduction_pct": 100.0 * (1.0 - indexed_median / reference_median),
                "exact_local_view_sha256_equal": reference_sha == indexed_sha,
                "local_view_sha256": reference_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
