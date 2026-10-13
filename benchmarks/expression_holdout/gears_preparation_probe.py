"""Server-only real-input parity/timing probe of the GEARS data adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.common.official_checkout import official_module_session
from benchmarks.expression_holdout.config import load_config, require_data_identity
from benchmarks.expression_holdout.data import prepare_data
from benchmarks.expression_holdout.gears import (
    OFFICIAL_SHA,
    _ranking_metadata,
    context_graph_builder,
    official_control_metadata,
    reuse_ranking_metadata,
)
from gradpert.data._io import atomic_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("old-source", "runtime", "cache-parent", "output", "config"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--cells", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.cells <= 20:
        raise ValueError("fresh bounded preparation probe required")
    if any(
        not p.resolve().is_relative_to("/data/yilangliu")
        for p in (args.old_source, args.runtime, args.cache_parent, args.output)
    ):
        raise ValueError("real data/probe stays on the server")
    runtime = json.loads(args.runtime.read_text())
    source = inspect_source_identity(
        ROOT,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=runtime["publication_receipt"],
        expected_publication_receipt_sha256=runtime["publication_sha256"],
    )
    legacy_source = inspect_source_identity(
        args.old_source,
        formal=False,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
    )
    if legacy_source.dirty:
        raise ValueError("old reference must be clean")
    legacy_path = args.old_source / "benchmarks/expression_holdout/gears.py"
    spec = importlib.util.spec_from_file_location("gears_preparation_reference", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    method_sha = hashlib.sha256(inspect.getsource(_ranking_metadata).encode()).hexdigest()
    if (
        method_sha
        != hashlib.sha256(inspect.getsource(legacy._ranking_metadata).encode()).hexdigest()
    ):
        raise ValueError("ranking method changed; cached ranking not reusable")
    import torch

    cache_path = args.cache_parent / "fit/official_data/expression_holdout/perturb_processed.h5ad"
    parent_run = json.loads((args.cache_parent / "RUN.json").read_text())
    cache = {
        "path": str(cache_path),
        "sha256": sha256_file(cache_path),
        "parent_run_root": str(args.cache_parent),
        "parent_run_sha256": sha256_file(args.cache_parent / "RUN.json"),
        "data_receipt_sha256": sha256_file(args.cache_parent / "data-receipt.json"),
        "ranking_method_sha256": method_sha,
        "source_commit": parent_run["source_commit"],
        "config_sha256": parent_run["config_sha256"],
    }
    start = time.monotonic()
    config = load_config(args.config)
    prepared = prepare_data(config, Path(runtime["data_root"]))
    require_data_identity(prepared, config)
    allowed = np.asarray(prepared.allowed_indices, dtype=int)
    selected = tuple(
        map(
            str,
            prepared.adata.obs_names[prepared.adata.obs.condition.astype(str) != "ctrl"][
                : args.cells
            ],
        )
    )
    observations = {}
    with official_module_session(
        checkout_root=runtime["official_checkout"],
        expected_commit=OFFICIAL_SHA,
        module_names=("gears", "gears.gears"),
    ):
        import gears
        from gears import data_utils

        adata = data_utils.get_DE_genes(
            official_control_metadata(prepared.adata.copy()), skip_calc_de=True
        )
        ranking = reuse_ranking_metadata(adata, allowed, {**runtime, "gears_ranking_cache": cache})
        view = adata[list(selected)]
        controls = adata[adata.obs.condition.astype(str) == "ctrl"]
        instance = gears.PertData(str(args.cache_parent / "fit/official_data"))
        instance.adata = adata
        instance.set_pert_genes()
        instance.ctrl_adata = controls
        load_seconds = time.monotonic() - start
        upstream = instance.create_cell_graph_dataset
        for label, factory in (
            ("original", legacy.context_graph_builder),
            ("cached", context_graph_builder),
        ):
            np.random.seed(71)
            torch.manual_seed(71)
            audit = []
            builder = factory(upstream, prepared, audit)
            graphs = []
            start = time.monotonic()
            for condition in view.obs.condition.astype(str).unique():
                graphs.extend(builder(instance, view, str(condition), 2))
            elapsed = time.monotonic() - start
            np_state = np.random.get_state()
            rng = sha256_json(
                {"algorithm": np_state[0], "keys": np_state[1].tolist(), "rest": list(np_state[2:])}
            )
            tensors = []
            for graph in graphs:
                tensors.append(
                    {
                        "x": graph.x.detach().cpu().numpy().tobytes().hex(),
                        "y": graph.y.detach().cpu().numpy().tobytes().hex(),
                        "de_idx": np.asarray(graph.de_idx).tolist(),
                        "pert_idx": np.asarray(graph.pert_idx).tolist(),
                        "pert": graph.pert,
                    }
                )
            observations[label] = {
                "seconds": elapsed,
                "pair_sha256": sha256_json(audit),
                "graph_sha256": sha256_json(tensors),
                "numpy_rng_sha256": rng,
                "torch_rng_sha256": hashlib.sha256(
                    torch.get_rng_state().numpy().tobytes()
                ).hexdigest(),
                "graph_count": len(graphs),
                "cache_statistics": getattr(builder, "cache_statistics", None),
            }
    for key in (
        "pair_sha256",
        "graph_sha256",
        "numpy_rng_sha256",
        "torch_rng_sha256",
        "graph_count",
    ):
        if observations["original"][key] != observations["cached"][key]:
            raise ValueError(f"real GEARS graph/RNG parity failed: {key}")
    args.output.mkdir()
    atomic_json(
        args.output / "receipt.json",
        {
            "schema": "gears-preparation-parity-1",
            "status": "passed",
            "source_commit": source.commit,
            "source_clean": True,
            "old_source_commit": legacy_source.commit,
            "old_adapter_sha256": sha256_file(legacy_path),
            "official_commit": OFFICIAL_SHA,
            "cache_spec": cache,
            "ranking_cache_validated": ranking,
            "input_load_validate_seconds": load_seconds,
            "sample_cells": len(selected),
            "sample_rows_sha256": sha256_json(list(selected)),
            "modes": observations,
            "adapter_speedup": observations["original"]["seconds"]
            / observations["cached"]["seconds"],
            "scientific_complete": False,
            "optimizer_steps": 0,
            "zero_pkl": True,
        },
    )


if __name__ == "__main__":
    main()
