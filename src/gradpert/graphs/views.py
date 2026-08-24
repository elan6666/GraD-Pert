"""Prediction, global, and incoming-ring-induced graph views."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from gradpert.graphs.pruning import DirectedEdge, PrunedSourceGraph


@dataclass(frozen=True)
class GraphTopology:
    gene_ids: tuple[str, ...]
    sources: Mapping[str, PrunedSourceGraph]

    def __post_init__(self) -> None:
        if tuple(sorted(self.sources)) != ("go", "string"):
            raise ValueError("v1 topology requires exactly separate go and string sources")
        for name, graph in self.sources.items():
            if graph.source_name != name or graph.gene_ids != self.gene_ids:
                raise ValueError("source graph identity/gene order mismatch")
            if graph.top_k_incoming != 20:
                raise ValueError("v1 topology requires Top-20 incoming pruning per source")

    @property
    def n_nodes(self) -> int:
        return len(self.gene_ids)


@dataclass(frozen=True)
class GraphView:
    view_id: str
    node_ids: tuple[int, ...]
    edges_by_source: Mapping[str, tuple[DirectedEdge, ...]]
    masked_node_ids: tuple[int, ...]
    masked_anchor_ids: tuple[int, ...]
    warnings: tuple[str, ...]

    def local_edge_index(self, source_name: str) -> np.ndarray[Any, np.dtype[np.int64]]:
        local_by_global = {node_id: index for index, node_id in enumerate(self.node_ids)}
        edges = self.edges_by_source[source_name]
        return np.asarray(
            [
                [local_by_global[edge.source] for edge in edges],
                [local_by_global[edge.target] for edge in edges],
            ],
            dtype=np.int64,
        )


@dataclass(frozen=True)
class GraphViewBatch:
    prediction: GraphView
    globals: tuple[GraphView, GraphView]
    locals: tuple[GraphView, ...]
    masked_global_index: int
    masked_local_indices: tuple[int, int, int, int]
    global_mask_ratio: float


@dataclass(frozen=True)
class GraDPertTrainingViews:
    """One batch-shared global/prediction graph plus condition-specific locals."""

    prediction: GraphView
    globals: tuple[GraphView, GraphView]
    locals_by_condition: Mapping[str, tuple[GraphView, ...]]
    anchors_by_condition: Mapping[str, tuple[int, ...]]
    masked_global_index: int
    masked_local_indices_by_condition: Mapping[str, tuple[int, int, int, int]]
    global_mask_ratio: float


def clean_graph_view(view: GraphView) -> GraphView:
    """Preserve topology while removing Student-only input masks for Teacher."""

    return GraphView(
        view_id=view.view_id,
        node_ids=view.node_ids,
        edges_by_source=view.edges_by_source,
        masked_node_ids=(),
        masked_anchor_ids=(),
        warnings=view.warnings,
    )


def stable_view_seed(
    *,
    run_seed: int,
    global_step: int,
    condition_id: str,
    namespace: str,
    view_index: int,
) -> int:
    if global_step < 0 or view_index < 0 or not condition_id or not namespace:
        raise ValueError("view seed inputs must be non-negative/non-empty")
    payload = (f"{run_seed}::{global_step}::{condition_id}::{namespace}::{view_index}").encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], byteorder="big")


def _self_loops(node_ids: Iterable[int]) -> tuple[DirectedEdge, ...]:
    return tuple(DirectedEdge(node_id, node_id, 1.0) for node_id in sorted(node_ids))


def _base_edges_with_self_loops(
    graph: PrunedSourceGraph,
    *,
    node_ids: set[int] | None = None,
) -> tuple[DirectedEdge, ...]:
    selected = set(range(graph.n_nodes)) if node_ids is None else node_ids
    edges = [edge for edge in graph.edges if edge.source in selected and edge.target in selected]
    edges.extend(_self_loops(selected))
    return tuple(sorted(edges, key=lambda edge: (edge.target, edge.source)))


def _drop_edges(
    graph: PrunedSourceGraph,
    *,
    probability: float,
    rng: np.random.Generator,
) -> tuple[DirectedEdge, ...]:
    if not 0 <= probability < 1:
        raise ValueError("DropEdge probability must be in [0, 1)")
    kept = [edge for edge in graph.edges if float(rng.random()) >= probability]
    kept.extend(_self_loops(range(graph.n_nodes)))
    return tuple(sorted(kept, key=lambda edge: (edge.target, edge.source)))


def _nonself_incident_nodes(edges_by_source: Mapping[str, tuple[DirectedEdge, ...]]) -> set[int]:
    nodes: set[int] = set()
    for edges in edges_by_source.values():
        for edge in edges:
            if edge.source != edge.target:
                nodes.add(edge.source)
                nodes.add(edge.target)
    return nodes


def _base_nonself_incident_nodes(topology: GraphTopology) -> set[int]:
    return {
        node_id
        for graph in topology.sources.values()
        for edge in graph.edges
        for node_id in (edge.source, edge.target)
        if edge.source != edge.target
    }


def _incoming_neighbors(topology: GraphTopology) -> dict[int, set[int]]:
    incoming: dict[int, set[int]] = {node_id: set() for node_id in range(topology.n_nodes)}
    for graph in topology.sources.values():
        for edge in graph.edges:
            incoming[edge.target].add(edge.source)
    return incoming


def build_ring_induced_view(
    topology: GraphTopology,
    *,
    anchors: Iterable[int],
    node_budget: int,
    seed: int,
    view_id: str,
    mask_anchors: bool,
) -> GraphView:
    anchor_ids = tuple(sorted(set(anchors)))
    if not anchor_ids:
        raise ValueError("local view requires at least one active anchor")
    if any(anchor < 0 or anchor >= topology.n_nodes for anchor in anchor_ids):
        raise ValueError("anchor is outside the graph universe")
    if node_budget < len(anchor_ids):
        raise ValueError("local node budget cannot retain all active anchors")

    incoming = _incoming_neighbors(topology)
    selected = set(anchor_ids)
    frontier = set(anchor_ids)
    rng = np.random.Generator(np.random.PCG64(seed))
    while frontier and len(selected) < node_budget:
        boundary = set().union(*(incoming[node_id] for node_id in frontier)) - selected
        if not boundary:
            break
        remaining = node_budget - len(selected)
        if len(boundary) <= remaining:
            selected.update(boundary)
            frontier = boundary
            continue
        candidates = np.asarray(sorted(boundary), dtype=np.int64)
        sampled = rng.choice(candidates, size=remaining, replace=False)
        selected.update(int(node_id) for node_id in sampled)
        break

    edges_by_source = {
        name: _base_edges_with_self_loops(graph, node_ids=selected)
        for name, graph in topology.sources.items()
    }
    incident_nodes = _nonself_incident_nodes(edges_by_source)
    eligible_anchors = tuple(anchor for anchor in anchor_ids if anchor in incident_nodes)
    isolated_anchors = tuple(anchor for anchor in anchor_ids if anchor not in incident_nodes)
    warnings = tuple(
        f"self_loop_only_anchor:{topology.gene_ids[anchor]}" for anchor in isolated_anchors
    )
    return GraphView(
        view_id=view_id,
        node_ids=tuple(sorted(selected)),
        edges_by_source=edges_by_source,
        masked_node_ids=(),
        masked_anchor_ids=eligible_anchors if mask_anchors else (),
        warnings=warnings,
    )


def _prediction_view(topology: GraphTopology) -> GraphView:
    return GraphView(
        view_id="prediction",
        node_ids=tuple(range(topology.n_nodes)),
        edges_by_source={
            name: _base_edges_with_self_loops(graph) for name, graph in topology.sources.items()
        },
        masked_node_ids=(),
        masked_anchor_ids=(),
        warnings=(),
    )


def build_prediction_graph_view(topology: GraphTopology) -> GraphView:
    """Return the fixed full graph used by prediction and inference only."""

    return _prediction_view(topology)


def _build_global_views(
    topology: GraphTopology,
    *,
    protected_anchor_ids: tuple[int, ...],
    heldout_target_ids: Iterable[int],
    run_seed: int,
    global_step: int,
    batch_key: str,
    drop_edge_probability: float = 0.1,
) -> tuple[tuple[GraphView, GraphView], int, float]:
    heldout = set(heldout_target_ids)

    global_views: list[GraphView] = []
    for view_index in range(2):
        edges_by_source = {}
        for source_index, (name, graph) in enumerate(sorted(topology.sources.items())):
            seed = stable_view_seed(
                run_seed=run_seed,
                global_step=global_step,
                condition_id=batch_key,
                namespace=f"global_dropedge_{name}",
                view_index=view_index * 2 + source_index,
            )
            edges_by_source[name] = _drop_edges(
                graph,
                probability=drop_edge_probability,
                rng=np.random.Generator(np.random.PCG64(seed)),
            )
        global_views.append(
            GraphView(
                view_id=f"global_{view_index}",
                node_ids=tuple(range(topology.n_nodes)),
                edges_by_source=edges_by_source,
                masked_node_ids=(),
                masked_anchor_ids=(),
                warnings=(),
            )
        )

    mask_rng = np.random.Generator(
        np.random.PCG64(
            stable_view_seed(
                run_seed=run_seed,
                global_step=global_step,
                condition_id=batch_key,
                namespace="global_node_mask",
                view_index=0,
            )
        )
    )
    masked_global_index = int(mask_rng.integers(0, 2))
    ratio = float(mask_rng.uniform(0.1, 0.5))
    base_incident = _base_nonself_incident_nodes(topology)
    eligible = sorted(base_incident - set(protected_anchor_ids) - heldout)
    mask_count = min(len(eligible), max(1, math.floor(ratio * len(eligible)))) if eligible else 0
    masked_nodes = (
        tuple(sorted(int(node_id) for node_id in mask_rng.choice(eligible, mask_count, False)))
        if mask_count
        else ()
    )
    chosen_global = global_views[masked_global_index]
    global_views[masked_global_index] = GraphView(
        view_id=chosen_global.view_id,
        node_ids=chosen_global.node_ids,
        edges_by_source=chosen_global.edges_by_source,
        masked_node_ids=masked_nodes,
        masked_anchor_ids=(),
        warnings=chosen_global.warnings,
    )
    return (global_views[0], global_views[1]), masked_global_index, ratio


def _build_local_views(
    topology: GraphTopology,
    *,
    anchor_ids: tuple[int, ...],
    run_seed: int,
    global_step: int,
    condition_id: str,
    local_count: int,
    local_node_budget: int,
) -> tuple[tuple[GraphView, ...], tuple[int, int, int, int]]:
    if local_count != 8:
        raise ValueError("v1 requires exactly eight local views")

    local_mask_rng = np.random.Generator(
        np.random.PCG64(
            stable_view_seed(
                run_seed=run_seed,
                global_step=global_step,
                condition_id=condition_id,
                namespace="local_anchor_mask_assignment",
                view_index=0,
            )
        )
    )
    masked_local_indices = tuple(
        sorted(int(index) for index in local_mask_rng.choice(local_count, 4, replace=False))
    )
    locals_ = tuple(
        build_ring_induced_view(
            topology,
            anchors=anchor_ids,
            node_budget=local_node_budget,
            seed=stable_view_seed(
                run_seed=run_seed,
                global_step=global_step,
                condition_id=condition_id,
                namespace="local_boundary",
                view_index=view_index,
            ),
            view_id=f"local_{view_index}",
            mask_anchors=view_index in masked_local_indices,
        )
        for view_index in range(local_count)
    )
    return locals_, cast(tuple[int, int, int, int], masked_local_indices)


def build_training_graph_views(
    topology: GraphTopology,
    *,
    anchors_by_condition: Mapping[str, Iterable[int]],
    heldout_target_ids: Iterable[int],
    run_seed: int,
    global_step: int,
    drop_edge_probability: float = 0.1,
    local_count: int = 8,
    local_node_budget: int = 512,
) -> GraDPertTrainingViews:
    """Build one batch-level global pair and locals once per unique condition."""

    if not anchors_by_condition:
        raise ValueError("training views require at least one condition")
    normalized: dict[str, tuple[int, ...]] = {}
    for condition_id in sorted(anchors_by_condition):
        anchors = tuple(sorted(set(anchors_by_condition[condition_id])))
        if not condition_id or not anchors:
            raise ValueError("condition IDs and active anchors must be non-empty")
        if any(anchor < 0 or anchor >= topology.n_nodes for anchor in anchors):
            raise ValueError("active anchor is outside the graph universe")
        normalized[condition_id] = anchors

    protected = tuple(sorted({anchor for anchors in normalized.values() for anchor in anchors}))
    batch_key = "batch[" + "|".join(normalized) + "]"
    globals_, masked_global_index, ratio = _build_global_views(
        topology,
        protected_anchor_ids=protected,
        heldout_target_ids=heldout_target_ids,
        run_seed=run_seed,
        global_step=global_step,
        batch_key=batch_key,
        drop_edge_probability=drop_edge_probability,
    )
    locals_by_condition: dict[str, tuple[GraphView, ...]] = {}
    masked_locals: dict[str, tuple[int, int, int, int]] = {}
    for condition_id, anchors in normalized.items():
        local_views, masked_indices = _build_local_views(
            topology,
            anchor_ids=anchors,
            run_seed=run_seed,
            global_step=global_step,
            condition_id=condition_id,
            local_count=local_count,
            local_node_budget=local_node_budget,
        )
        locals_by_condition[condition_id] = local_views
        masked_locals[condition_id] = masked_indices
    return GraDPertTrainingViews(
        prediction=_prediction_view(topology),
        globals=globals_,
        locals_by_condition=locals_by_condition,
        anchors_by_condition=normalized,
        masked_global_index=masked_global_index,
        masked_local_indices_by_condition=masked_locals,
        global_mask_ratio=ratio,
    )


def build_graph_view_batch(
    topology: GraphTopology,
    *,
    anchors: Iterable[int],
    heldout_target_ids: Iterable[int],
    run_seed: int,
    global_step: int,
    condition_id: str,
    drop_edge_probability: float = 0.1,
    local_count: int = 8,
    local_node_budget: int = 512,
) -> GraphViewBatch:
    """Compatibility wrapper for a one-condition training-view batch."""

    training = build_training_graph_views(
        topology,
        anchors_by_condition={condition_id: anchors},
        heldout_target_ids=heldout_target_ids,
        run_seed=run_seed,
        global_step=global_step,
        drop_edge_probability=drop_edge_probability,
        local_count=local_count,
        local_node_budget=local_node_budget,
    )
    return GraphViewBatch(
        prediction=training.prediction,
        globals=training.globals,
        locals=training.locals_by_condition[condition_id],
        masked_global_index=training.masked_global_index,
        masked_local_indices=training.masked_local_indices_by_condition[condition_id],
        global_mask_ratio=training.global_mask_ratio,
    )
