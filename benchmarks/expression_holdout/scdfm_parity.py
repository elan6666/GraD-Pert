"""Real frozen scDFM GPU updates: repeatability before layout-preserving offload."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from gradpert.data._io import atomic_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file

from .config import load_config
from .data import build_prepared
from .scdfm import fit, predict


def compare(left, right, path="", stats=None):
    if stats is None:
        stats = {"tensors": 0, "max_abs": 0.0, "failures": []}
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            stats["failures"].append(path + ": shape/dtype")
        else:
            stats["tensors"] += 1
            if left.is_floating_point():
                difference = float((left - right).abs().max()) if left.numel() else 0.0
                stats["max_abs"] = max(stats["max_abs"], difference)
                if not torch.allclose(left, right, rtol=1e-6, atol=1e-7):
                    stats["failures"].append(path + ": tolerance")
            elif not torch.equal(left, right):
                stats["failures"].append(path + ": integer")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            stats["failures"].append(path + ": keys")
        else:
            for key in left:
                compare(left[key], right[key], path + "/" + str(key), stats)
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            stats["failures"].append(path + ": length")
        else:
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                compare(a, b, path + "/" + str(index), stats)
    elif isinstance(left, np.ndarray):
        if not np.allclose(left, right, rtol=1e-6, atol=1e-7):
            stats["failures"].append(path + ": prediction")
    elif left != right:
        stats["failures"].append(path + ": value")
    return stats


def run(config_path: Path, runtime_path: Path, output: Path) -> dict:
    if output.exists() or not output.resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("fresh server parity output required")
    runtime = json.loads(runtime_path.read_text())
    source = inspect_source_identity(
        Path(__file__).resolve().parents[2],
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=runtime["publication_receipt"],
        expected_publication_receipt_sha256=runtime["publication_sha256"],
    )
    if source.commit != runtime["source_commit"] or runtime["device"] != "cuda:0":
        raise ValueError("parity source/device binding changed")
    config = load_config(config_path)
    names = tuple("g" + str(i) for i in range(32))
    rows = tuple(["c0", "c1", "c2", "c3"] + [f"t{i}" for i in range(24)] + ["v0", "v1", "v2", "v3"])
    labels = tuple(
        ["ctrl"] * 4 + [names[i % 3] + "+ctrl" for i in range(24)] + [names[3] + "+ctrl"] * 4
    )
    prepared = build_prepared(
        matrix=np.random.default_rng(1).uniform(0.01, 2, (32, 30)),
        allowed_indices=tuple(range(30)),
        gene_ids=names,
        graph_gene_ids=names,
        row_ids=rows,
        conditions=labels,
        contexts=("J::0",) * 32,
        train_row_ids=rows[4:28],
        val_row_ids=rows[28:],
        control_row_ids=rows[:4],
        train_conditions=tuple(names[i] + "+ctrl" for i in range(3)),
        val_conditions=(names[3] + "+ctrl",),
    )
    torch.cuda.set_per_process_memory_fraction(runtime["cuda_memory_fraction"])
    output.mkdir()
    snapshots, timing = {}, {}
    for name, storage in (("none-a", "none"), ("none-b", "none"), ("cpu", "cpu")):
        cfg = copy.deepcopy(config)
        cfg["training"].update(batch_size=4, infer_top_gene=16, activation_offload=storage)
        torch.cuda.reset_peak_memory_stats()
        fitted = fit(prepared, cfg, runtime, output / name, max_updates=3)
        snapshots[name] = {
            "checkpoint": torch.load(
                output / name / "last.pt", map_location="cpu", weights_only=False
            ),
            "gradients": {
                n: p.grad.detach().cpu().clone()
                for n, p in fitted.model.vf.named_parameters()
                if p.grad is not None
            },
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": torch.cuda.get_rng_state().cpu().clone(),
        }
        controls = np.random.default_rng(2).uniform(0.01, 2, (6, 32)).astype("float32")
        snapshots[name]["prediction"] = predict(
            fitted, controls, (names[0],), names, cell_batch=3, query_count=16, seed=2
        )
        timing[name] = {
            "updates": fitted.receipt["completed_updates"],
            "last_loss": fitted.receipt["last_loss"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        del fitted
        torch.cuda.empty_cache()
    pairs = {name: compare(snapshots["none-a"], snapshots[name]) for name in ("none-b", "cpu")}
    receipt = {
        "schema": "scdfm-layout-offload-parity-1",
        "source_commit": source.commit,
        "source_clean": not source.dirty,
        "config_sha256": sha256_file(config_path),
        "runtime_sha256": sha256_file(runtime_path),
        "official_commit": config["source_code"]["commit"],
        "status": "passed" if all(not row["failures"] for row in pairs.values()) else "failed",
        "rtol": 1e-6,
        "atol": 1e-7,
        "pairs": pairs,
        "modes": timing,
        "scope": "official512x4_model_synthetic32genes_batch4_query16_three_updates",
        "formal_capacity_passed": False,
        "scientific_complete": False,
        "zero_pkl": not list(output.rglob("*.pkl")),
    }
    atomic_json(output / "receipt.json", receipt)
    if receipt["status"] != "passed":
        raise RuntimeError("scDFM repeatability/offload parity failed at original tolerance")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.runtime, args.output)))
