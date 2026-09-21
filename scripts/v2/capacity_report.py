"""Collect hash-bound engineering probes without declaring an unmeasured maximum."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.hashing import sha256_file


def collect(receipt_path: Path, config_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    config = load_experiment_config(config_path)
    architecture, options = V2Options.parse_parameters(config.model.parameters)
    if receipt.get("config_sha256") != sha256_file(config_path):
        raise ValueError("capacity config checksum mismatch")
    if receipt.get("kind") != "capacity_only" or receipt.get("status") != "passed":
        raise ValueError("only completed engineering probes are eligible")
    if receipt.get("steps_completed", 0) < 128:
        raise ValueError("capacity requires at least 128 sustained updates")
    source = receipt.get("source", {})
    if source.get("dirty") is not False or source.get("commit") != source.get("published_commit"):
        raise ValueError("probe must identify a clean published source")
    if not source.get("formal_eligible") or not receipt.get("resume_checkpoint_sha256"):
        raise ValueError("source or checkpoint continuation evidence missing")
    shape = receipt.get("inference_shape", [])
    if len(shape) != 2 or shape[0] != 300 or shape[1] < 1:
        raise ValueError("300-control inference evidence missing")
    if receipt.get("world_size", 1) != options.world_size:
        raise ValueError("probe and config world sizes differ")
    durations = receipt.get("measured_update_seconds", [])
    if len(durations) < 120 or any(not math.isfinite(t) or t <= 0 for t in durations):
        raise ValueError("sustained update timings missing or invalid")
    rate = receipt.get("cells_per_second", 0)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("invalid update throughput")
    ranks = receipt.get("rank_measurements", [])
    communication = [r.get("gradient_reduction_seconds", []) for r in ranks]
    if communication and any(len(v) != len(durations) for v in communication):
        raise ValueError("communication timing population mismatch")
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "config": str(config_path.resolve()),
        "config_sha256": receipt["config_sha256"],
        "training_sha": source["commit"],
        "dataset": config.dataset_id,
        "physical_gpus": receipt["gpu"],
        "world_size": options.world_size,
        "microbatch": options.microbatch,
        "accumulation": options.accumulation,
        "effective_batch": config.training.train_batch_size.value,
        "width": architecture.width,
        "lambda1": options.lambda1,
        "lambda2": options.lambda2,
        "steps_completed": receipt["steps_completed"],
        "cells_per_second": rate,
        "end_to_end_cells_per_second": receipt.get("end_to_end_training_cells_per_second"),
        "step_median_seconds": statistics.median(durations),
        "step_max_seconds": max(durations),
        "peak_allocated_bytes_per_gpu": receipt["peak_allocated_bytes"],
        "peak_reserved_bytes_per_gpu": receipt["peak_reserved_bytes"],
        "gradient_reduction_mean_seconds": (
            statistics.mean(max(v) for v in zip(*communication, strict=True))
            if communication
            else None
        ),
        "inference_control_count": shape[0],
        "inference_gene_count": shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", nargs=2, type=Path, action="append", required=True, metavar=("RECEIPT", "CONFIG")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [collect(*pair) for pair in args.probe]
    if len({r["receipt_sha256"] for r in rows}) != len(rows):
        parser.error("duplicate probe receipts")
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": "gradpert-v2-capacity-report-1",
        "probes": rows,
        "scope": "Measured points only; no maximum, chosen batch, or scientific result implied.",
        "comparison": (
            "Source SHA and execution/loss settings remain visible; "
            "unequal settings are not controlled speedups."
        ),
        "communication_scope": (
            "Gradient reduction includes packing and rank wait; other collectives excluded."
        ),
    }
    (args.output / "capacity.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output / "capacity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output.resolve()), "verified_probes": len(rows)}))


if __name__ == "__main__":
    main()
