"""CPU ABBA of immutable full-context topology reuse with exact view/RNG parity."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from capacity_probe import state_fingerprint

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.execution.identity import inspect_source_identity
from gradpert.graphs.materialization import load_dataset_graph_topology
from gradpert.hashing import sha256_file
from gradpert.training.v2.views import NeighborhoodIndex


def execute(args):
    if args.output.exists() or not args.output.resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("fresh server-only engineering output required")
    config = load_experiment_config(args.config)
    arch, options = V2Options.parse_parameters(config.model.parameters)
    source = inspect_source_identity(
        Path(__file__).resolve().parents[2],
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    if (
        not arch.graph_read_mode == "relay"
        or sha256_file(Path(options.graph_manifest_path)) != options.graph_manifest_sha256
    ):
        raise ValueError("frozen relay graph configuration required")
    topology = load_dataset_graph_topology(
        dataset_id=config.dataset_id, protocol_id=config.data.protocol_id, data_root=args.data_root
    )
    rng = np.random.default_rng(17)
    queries = [
        np.sort(rng.choice(topology.n_nodes, options.query_count, replace=False)) for _ in range(4)
    ]
    results = []
    expected = None
    for reuse in (False, True, True, False):
        start = time.perf_counter()
        index = NeighborhoodIndex(
            topology,
            options.graph_expander_degree,
            options.graph_expander_seed,
            expander_type=options.graph_expander_type,
            relay=True,
            cache_full_context=reuse,
        )
        initialization = time.perf_counter() - start
        rng = np.random.default_rng(311)
        timings, signatures = [], []
        for ids in queries:
            start = time.perf_counter()
            view = index.view(
                ids,
                [(int(ids[0]),)],
                rng=rng,
                device=torch.device("cpu"),
                induced=False,
                mask_ratio=0.25,
            )
            timings.append(time.perf_counter() - start)
            signatures.append(
                state_fingerprint(
                    {
                        f.name: (
                            vars(view.context) if f.name == "context" else getattr(view, f.name)
                        )
                        for f in fields(view)
                    }
                )
            )
        identity = dict(views=signatures, rng=state_fingerprint(rng.bit_generator.state))
        if expected is not None and identity != expected:
            raise ValueError("view identity/tensor/RNG changed")
        expected = identity
        results.append(
            dict(
                reuse=reuse,
                initialization_seconds=initialization,
                view_seconds=timings,
                warm_view_median_seconds=float(np.median(timings[1:])),
                context_cache_builds=index.context_cache_builds,
                context_cache_hits=index.context_cache_hits,
                **identity,
            )
        )
    reference = float(np.median([r["warm_view_median_seconds"] for r in results if not r["reuse"]]))
    candidate = float(np.median([r["warm_view_median_seconds"] for r in results if r["reuse"]]))
    receipt = dict(
        schema="context-topology-cache-abba-1",
        passed=True,
        source=source.payload(),
        config_sha256=sha256_file(args.config),
        graph_manifest_sha256=options.graph_manifest_sha256,
        nodes=topology.n_nodes,
        queries=options.query_count,
        exact_tensor_and_rng_parity=True,
        measurements=results,
        reference_warm_seconds=reference,
        candidate_warm_seconds=candidate,
        component_speedup=reference / candidate,
        limitation=(
            "CPU graph-view construction only; no expression input, "
            "model update or end-to-end speed claim"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "data-root", "output", "publication"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--publication-sha256", required=True)
    print(json.dumps(execute(parser.parse_args()), indent=2))
