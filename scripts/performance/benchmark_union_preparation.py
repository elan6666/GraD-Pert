"""CPU-only attribution microbenchmark; never a training/speed acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradpert.graphs import build_induced_edge_index, build_training_graph_views
from gradpert.modeling.encoders import _expander_pairs, _is_undirected
from gradpert.pilots.vnext_graph_axis import load_vnext_graph_topology
from gradpert.training.data import CanonicalTrainingData


def prepare(node_count: int, sources: Any, *, array_native: bool) -> Any:
    """Compare only the measured tuple/channel preparation, not union sorting."""
    channels = []
    if array_native:
        for name, pairs in sources:
            array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
            if array.size and (array.min() < 0 or array.max() >= node_count):
                raise ValueError("edge outside node axis")
            channels.append((name, array))
            if not _is_undirected(pairs):
                channels.append((name + ":reverse", array[:, ::-1]))
        nodes = np.arange(node_count, dtype=np.int64)
        channels.append(("self", np.column_stack((nodes, nodes))))
        channels.append(("expander", np.asarray(_expander_pairs(node_count, 3), dtype=np.int64)))
    else:
        for _, pairs in sources:
            if any(a < 0 or a >= node_count or b < 0 or b >= node_count for a, b in pairs):
                raise ValueError("edge outside node axis")
        for name, pairs in sources:
            channels.append((name, pairs))
            if not _is_undirected(pairs):
                channels.append((name + ":reverse", tuple((b, a) for a, b in pairs)))
        channels.append(("self", tuple((i, i) for i in range(node_count))))
        channels.append(("expander", _expander_pairs(node_count, 3)))
        channels = [
            (name, np.asarray(pairs, dtype=np.int64).reshape(-1, 2)) for name, pairs in channels
        ]
    # Include the identical downstream concatenation and channel allocation costs.
    pairs = np.concatenate([array for _, array in channels])
    ids = np.concatenate(
        [np.full(len(array), i, dtype=np.int64) for i, (_, array) in enumerate(channels)]
    )
    return tuple(name for name, _ in channels), pairs, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite evidence")
    source = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source):
        raise SystemExit("source must be clean")
    config = source / "configs/r50/batch512/gradpert_b2/nadig_jurkat.yaml"
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    if config_sha != "839c7518d79bafd592ca3587d07442179590976da2bd5405ba4af3541ba78505":
        raise SystemExit("wrong frozen E3 coordinate")
    graph_root = args.data_root / "vnext/graph_axes/nadig_jurkat/hvg512_plus_targets"
    topology, manifest = load_vnext_graph_topology(graph_root)
    if topology.n_nodes != 2809:
        raise SystemExit("wrong graph node count")
    torch.set_num_threads(1)
    with CanonicalTrainingData(
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
        data_root=args.data_root,
        run_seed=1,
        graph_gene_ids_override=topology.gene_ids,
        graph_manifest_path_override=graph_root / "manifest.json",
    ) as data:
        specs = data._batch_specs(epoch=0, batch_size=512, max_unique_conditions=8)[:3]
        heldout = tuple(
            sorted(
                {
                    anchor
                    for condition in (*data.split.val_conditions, *data.split.test_conditions)
                    for anchor in data.anchors_by_condition[condition]
                }
            )
        )
        groups = []
        for step, spec in enumerate(specs):
            views = build_training_graph_views(
                topology,
                anchors_by_condition=spec.anchors_by_condition,
                heldout_target_ids=heldout,
                run_seed=1,
                global_step=step,
                local_count=4,
                local_node_budget=1404,
                local_builder="ring_induced",
                local_anchor_mask_count=0,
                induced_edges=build_induced_edge_index(topology),
            )
            # Unique global/local view construction, not a claimed encoder-call count.
            group = []
            for view in (
                *views.globals,
                *(v for vs in views.locals_by_condition.values() for v in vs),
            ):
                local = {node: i for i, node in enumerate(view.node_ids)}
                pairs = tuple(
                    (
                        name,
                        tuple(
                            (local[e.source], local[e.target]) for e in view.edges_by_source[name]
                        ),
                    )
                    for name in topology.active_sources
                )
                group.append((len(view.node_ids), pairs))
            groups.append(group)
    frozen_sha = hashlib.sha256(repr(groups).encode()).hexdigest()
    rng = torch.get_rng_state().clone()
    for group in groups:
        for n, pairs in group:
            a, b = prepare(n, pairs, array_native=False), prepare(n, pairs, array_native=True)
            assert a[0] == b[0]
            np.testing.assert_array_equal(a[1], b[1])
            np.testing.assert_array_equal(a[2], b[2])
    assert torch.equal(rng, torch.get_rng_state())
    timings: dict[str, list[float]] = {"reference": [], "array_native": []}
    for repeat in range(6):
        for mode in (False, True) if repeat % 2 == 0 else (True, False):
            start = time.perf_counter()
            for group in groups:
                for n, pairs in group:
                    prepare(n, pairs, array_native=mode)
            elapsed = (time.perf_counter() - start) * 1000 / len(groups)
            if repeat:
                timings["array_native" if mode else "reference"].append(elapsed)
    payload = {
        "scientific_completion": False,
        "acceptance_timing": False,
        "scope": (
            "CPU channel preparation only; three actual epoch0 batches; "
            "no expression/evaluation materialization"
        ),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "source_clean": True,
        "config_sha256": config_sha,
        "graph_manifest_sha256": hashlib.sha256(
            (graph_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "topology_sha256": manifest.topology_content_sha256,
        "frozen_ordered_pairs_sha256": frozen_sha,
        "view_counts": [len(g) for g in groups],
        "channel_arrays_exact": True,
        "global_torch_rng_exact": True,
        "milliseconds_per_batch": timings,
        "medians_ms": {k: statistics.median(v) for k, v in timings.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
