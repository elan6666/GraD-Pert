"""Synthetic server CPU integration through actual frozen upstream packages."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np

from benchmarks.common.official_checkout import official_module_session
from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file

from .config import load_config
from .data import build_prepared
from .runner import restore_last


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to("/data/yilangliu") or a.output.exists():
        raise ValueError("fresh server-only CPU integration output required")
    config = load_config(a.config)
    cfg = deepcopy(config)
    runtime = json.loads(a.runtime.read_text())
    with Path(runtime["gears_assets"]).joinpath("gene2go_all.pkl").open("rb") as h:
        go = pickle.load(h)
    with Path(runtime["prior_path"]).open("rb") as h:
        prior = pickle.load(h)
    genes = tuple(sorted(set(go) & set(prior))[:32])
    if len(genes) != 32:
        raise ValueError("official graph/prior lacks synthetic integration identities")
    conditions = (genes[0] + "+ctrl", genes[1] + "+ctrl", genes[2] + "+ctrl")
    vals = (genes[3] + "+ctrl",)
    row_ids = tuple(
        ["c0", "c1", "c2", "c3"] + [f"t{i}" for i in range(24)] + ["v0", "v1", "v2", "v3"]
    )
    labels = tuple(["ctrl"] * 4 + [conditions[i % 3] for i in range(24)] + [vals[0]] * 4)
    contexts = tuple(
        ["J::0", "J::0", "J::1", "J::1"]
        + ["J::" + str(i % 2) for i in range(24)]
        + ["J::0", "J::0", "J::1", "J::1"]
    )
    allowed = tuple(range(30))
    prepared = build_prepared(
        matrix=np.random.default_rng(1).uniform(0.01, 2, size=(32, 30)),
        allowed_indices=allowed,
        gene_ids=genes,
        graph_gene_ids=genes,
        row_ids=row_ids,
        conditions=labels,
        contexts=contexts,
        train_row_ids=row_ids[4:28],
        val_row_ids=row_ids[28:],
        control_row_ids=row_ids[:4],
        train_conditions=conditions,
        val_conditions=vals,
    )
    runtime.update(device="cpu", seed=1, required_target_genes=list(genes[:4]))
    runtime.pop("prior_aliases", None)
    runtime.pop("prior_aliases_sha256", None)
    cfg["training"]["batch_size"] = 4
    a.output.mkdir(parents=True)
    adapter = importlib.import_module(f"benchmarks.expression_holdout.{cfg['model_id']}")
    names = {"scouter": ("scouter",), "gears": ("gears", "gears.gears")}
    session = (
        official_module_session(
            checkout_root=runtime["official_checkout"],
            expected_commit=cfg["source_code"]["commit"],
            module_names=names[cfg["model_id"]],
        )
        if cfg["model_id"] in names
        else contextlib.nullcontext()
    )
    with session:
        fitted = adapter.fit(prepared, cfg, runtime, a.output / "fit", max_updates=10)
        # Fresh restored weights must produce the same finite prediction with the
        # same official noise and gene identity, including restored excluded input.
        controls = np.random.default_rng(2).uniform(0.01, 2, (6, 32)).astype(np.float32)
        targets = (genes[0],)
        before = adapter.predict(
            fitted, controls, targets, genes, cell_batch=3, query_count=16, seed=2
        )
        restore_last(fitted, cfg["model_id"], a.output / "fit/last.pt")
        after = adapter.predict(
            fitted, controls, targets, genes, cell_batch=3, query_count=16, seed=2
        )
        np.testing.assert_array_equal(before, after)
        if before.shape != controls.shape or not np.isfinite(before).all():
            raise ValueError("actual restored upstream prediction is incomplete/nonfinite")
        count = fitted.receipt.get(
            "completed_updates", fitted.receipt.get("diagnostic_updates_completed")
        )
        if count != 10:
            raise ValueError("actual official update gate is incomplete")
        receipt = {
            "schema": "official-upstream-cpu-integration-v1",
            "model": cfg["model_id"],
            "official_commit": cfg["source_code"]["commit"],
            "config_sha256": sha256_file(a.config),
            "optimizer_updates": count,
            "restored_prediction_exact": True,
            "prediction_shape": list(after.shape),
            "zero_pkl": not list(a.output.rglob("*.pkl")),
            "diagnostic_deviations": ["synthetic32genes/32rows; batch4; CPU;10updates;query16"],
            "formal_capacity_passed": False,
            "scientific_complete": False,
            "fit": fitted.receipt,
        }
        if not receipt["zero_pkl"]:
            raise ValueError("CPU integration leaves scratchPKL")
        atomic_json(a.output / "receipt.json", receipt)
        print(json.dumps(receipt, default=str), flush=True)


if __name__ == "__main__":
    main()
