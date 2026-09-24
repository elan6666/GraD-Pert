"""Deterministic graph/cell views on an explicit frozen gene order."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from gradpert.config.v2 import V2Options
from gradpert.graphs import GraphTopology
from gradpert.modeling.v2.model import GraphContext
from gradpert.training.batch import GraDPertTrainingBatch

from .objective import CellView, GraphView, TrainingBatch


class NeighborhoodIndex:
    """Deduplicated incoming Top20/source plus deterministic expander/self edges."""

    def __init__(
        self,
        topology: GraphTopology,
        degree: int,
        seed: int,
        *,
        expander_type: str = "permutation",
        propagated: bool = False,
    ) -> None:
        if degree < 0:
            raise ValueError("expander degree cannot be negative")
        if expander_type not in ("permutation", "hamiltonian"):
            raise ValueError("unknown expander type")
        n = topology.n_nodes
        rows: list[dict[int, list[bool]]] = [{} for _ in range(n)]

        def add(query: int, memory: int, source: int) -> None:
            rows[query].setdefault(memory, [False] * 4)[source] = True

        for name, slot in (("go", 0), ("string", 1)):
            for edge in topology.sources[name].edges:
                add(edge.target, edge.source, slot)
        rng = np.random.default_rng(seed)
        # Fixed random edges depend only on the frozen gene order and seed.
        for _ in range(degree):
            permutation = rng.permutation(n)
            if expander_type == "permutation":
                for query, memory in enumerate(permutation):
                    add(query, int(memory), 2)
            elif n > 1:
                for i, query in enumerate(permutation):
                    memory = int(permutation[(i + 1) % n])
                    add(int(query), memory, 2)
                    add(memory, int(query), 2)
        for query in range(n):
            add(query, query, 3)
        self.rows, self.n_nodes = rows, n
        self.gene_ids = topology.gene_ids
        self.propagated = propagated

    def view(
        self,
        ids: np.ndarray,
        targets: Sequence[Sequence[int]],
        *,
        rng: np.random.Generator,
        device: torch.device,
        induced: bool,
        edge_dropout: float = 0.0,
        mask_ratio: float = 0.0,
    ) -> GraphView:
        position = {int(g): i for i, g in enumerate(ids)}

        def sampled_edges(genes: np.ndarray) -> list[list[tuple[int, list[bool]]]]:
            result = []
            for gene in genes:
                row = []
                for memory, membership in sorted(self.rows[int(gene)].items()):
                    if induced and memory not in position:
                        continue
                    if memory != gene and rng.random() < edge_dropout:
                        continue
                    row.append((memory, membership))
                result.append(row)
            return result

        def pack_edges(
            edge_rows: list[list[tuple[int, list[bool]]]],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            max_neighbors = max(map(len, edge_rows))
            neighbors = np.zeros((len(edge_rows), max_neighbors), dtype=np.int64)
            valid = np.zeros_like(neighbors, dtype=bool)
            sources = np.zeros((*neighbors.shape, 4), dtype=bool)
            for i, row in enumerate(edge_rows):
                for j, (memory, membership) in enumerate(row):
                    neighbors[i, j], valid[i, j], sources[i, j] = memory, True, membership
            return neighbors, valid, sources

        edges = sampled_edges(ids)
        neighbors, valid, sources = pack_edges(edges)
        context_data = None
        if self.propagated:
            context_ids = np.unique(np.concatenate((ids, neighbors[valid].astype(np.int64))))
            selected_edges = {int(gene): row for gene, row in zip(ids, edges, strict=True)}
            context_edges = []
            for gene in context_ids:
                if int(gene) in selected_edges:
                    context_edges.append(selected_edges[int(gene)])
                else:
                    context_edges.extend(sampled_edges(np.array([gene], dtype=np.int64)))
            context_neighbors, context_valid, context_sources = pack_edges(context_edges)
            query_neighbors = np.where(valid, np.searchsorted(context_ids, neighbors), 0)
            if (query_neighbors[valid] >= len(context_ids)).any():
                raise ValueError("graph context is missing a query neighbor")
            context_data = (
                context_ids,
                context_neighbors,
                context_valid,
                context_sources,
                np.searchsorted(context_ids, ids),
                query_neighbors,
            )
        target_positions = np.zeros((len(targets), max(map(len, targets))), dtype=np.int64)
        target_valid = np.zeros_like(target_positions, dtype=bool)
        for i, target in enumerate(targets):
            for j, gene in enumerate(target):
                target_positions[i, j], target_valid[i, j] = position[gene], True
        count = int(len(ids) * mask_ratio)
        masked = np.sort(rng.choice(len(ids), count, replace=False))

        def tensor(a: np.ndarray) -> Tensor:
            return torch.from_numpy(a).to(device)

        view = GraphView(
            *(
                tensor(a)
                for a in (ids, neighbors, valid, sources, target_positions, target_valid, masked)
            )
        )
        if context_data is not None:
            view.context = GraphContext(*(tensor(a) for a in context_data))
        return view

    def local_nodes(
        self, anchors: Sequence[int], budget: int, rng: np.random.Generator
    ) -> np.ndarray:
        selected = set(anchors)
        if len(selected) > budget:
            raise ValueError("local budget smaller than forced target set")
        frontier = set(selected)
        while frontier and len(selected) < budget:
            candidates = sorted({k for q in frontier for k in self.rows[q]} - selected)
            rng.shuffle(candidates)
            frontier = set(candidates[: budget - len(selected)])
            selected.update(frontier)
        if len(selected) < budget:
            remaining = np.array(sorted(set(range(self.n_nodes)) - selected))
            selected.update(map(int, rng.choice(remaining, budget - len(selected), replace=False)))
        return np.array(sorted(selected), dtype=np.int64)


def assemble_batch(
    raw: GraDPertTrainingBatch,
    index: NeighborhoodIndex,
    options: V2Options,
    rng: np.random.Generator,
    *,
    allowed_expression_ids: np.ndarray | None = None,
) -> TrainingBatch:
    device = raw.control_expression.device
    n = raw.control_expression.shape[1]
    pool = np.arange(n) if allowed_expression_ids is None else allowed_expression_ids
    if (
        pool.ndim != 1
        or pool.dtype.kind not in "iu"
        or not len(pool)
        or (pool < 0).any()
        or (pool >= n).any()
        or len(np.unique(pool)) != len(pool)
    ):
        raise ValueError("allowed expression IDs must be unique in-axis integers")
    if options.query_count > len(pool):
        raise ValueError("requested query count exceeds allowed expression axis")
    queries = np.sort(rng.choice(pool, options.query_count, replace=False))
    conditions = sorted(raw.anchors_by_condition)
    targets = [raw.anchors_by_condition[c] for c in conditions]
    anchors = sorted({g for target in targets for g in target})
    ids = np.union1d(queries, anchors)
    graph = index.view(ids, targets, rng=rng, device=device, induced=False)
    query_positions = torch.tensor(np.searchsorted(ids, queries), device=device)
    selection = torch.tensor(queries, device=device)
    control, truth = raw.control_expression[:, selection], raw.target_expression[:, selection]
    graph_views = []
    if options.lambda1:
        for i in range(2 + options.local_views):
            nodes = (
                np.arange(index.n_nodes)
                if i < 2
                else index.local_nodes(anchors, max(len(anchors), index.n_nodes // 2), rng)
            )
            graph_views.append(
                index.view(
                    nodes,
                    targets,
                    rng=rng,
                    device=device,
                    induced=True,
                    edge_dropout=options.graph_edge_dropout,
                    mask_ratio=options.graph_mask_ratio if i < 2 else 0,
                )
            )
    cell_views = []
    if options.lambda2:
        for i in range(2 + options.local_views):
            lo, hi = (
                (options.global_min_ratio, options.global_max_ratio)
                if i < 2
                else (options.local_min_ratio, options.local_max_ratio)
            )
            size = max(1, int(options.query_count * rng.uniform(lo, hi)))
            positions = np.sort(rng.choice(options.query_count, size, replace=False))
            mask = np.zeros((len(control), size), dtype=bool)
            for row in range(len(control)):
                if i < 2 and rng.random() < options.mask_probability:
                    masked = max(
                        1, int(size * rng.uniform(options.mask_min_ratio, options.mask_max_ratio))
                    )
                    mask[row, rng.choice(size, masked, replace=False)] = True
            cell_views.append(
                CellView(torch.tensor(positions, device=device), torch.tensor(mask, device=device))
            )
    condition_map = {c: i for i, c in enumerate(conditions)}
    condition_index = torch.tensor([condition_map[c] for c in raw.condition_ids], device=device)
    first: dict[tuple[str, str], int] = {}
    for i, key in enumerate(zip(raw.control_row_ids, raw.condition_ids, strict=True)):
        first.setdefault(key, i)
    return TrainingBatch(
        graph,
        query_positions,
        control,
        truth,
        condition_index,
        tuple(graph_views),
        tuple(cell_views),
        torch.tensor(list(first.values()), device=device),
    )
