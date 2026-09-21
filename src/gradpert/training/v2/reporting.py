"""Small, source-bound epoch curves from committed v2 lifecycle history."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file
from gradpert.training.curves import render_curves

METRICS = ("txpert_macro_pearson_delta", "trishift_pearson_delta", "systema_pearson")


def export_curves(root: Path) -> dict[str, Any]:
    journal = read_json(root / "epoch_state.json")
    history = read_json(root / "history.json")
    if not history or len(history) != journal["epoch"]:
        raise ValueError("curves require committed epoch history matching the journal")
    if [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("epoch history is not contiguous")
    identity = journal["identity"]
    source = identity["source"]["commit"]
    rows = []
    for record in history:
        validation = record["validation"]
        if validation["split"] != "val":
            raise ValueError("validation curves cannot contain test results")
        metrics = {m["metric_id"]: m["macro_mean"] for m in validation["metrics"]}
        if set(metrics) != set(METRICS):
            raise ValueError("v2 validation must report all three metric roles")
        rows.append(
            {
                "epoch": record["epoch"],
                "optimizer_steps": record["optimizer_steps"],
                "source_commit": source,
                "run_id": identity["run_id"],
                "train_prediction_step_mean": record["training"]["prediction"],
                "train_joint_step_mean": record["training"]["joint_loss"],
                "validation_prediction_loss": validation["prediction_loss"],
                **metrics,
            }
        )
    output = root / "epoch_curves.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Adapt committed v2 epoch summaries to the existing native curve interface.
    # These are update means, not invented per-step observations.
    shared = root / "curves"
    shared.mkdir(exist_ok=True)
    training = shared / "train_steps.csv"
    fields = ["global_step", "prediction_loss_update_mean", "joint_loss_update_mean"]
    with training.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "global_step": row["optimizer_steps"],
                    "prediction_loss_update_mean": row["train_prediction_step_mean"],
                    "joint_loss_update_mean": row["train_joint_step_mean"],
                }
            )
            atomic_json(
                shared / f"validation.epoch-{row['epoch'] - 1:04d}.json",
                {
                    "epoch": row["epoch"] - 1,
                    "global_step": row["optimizer_steps"],
                    "run_id": row["run_id"],
                    "source_commit": source,
                    "prediction_loss": row["validation_prediction_loss"],
                    **{metric: row[metric] for metric in METRICS},
                },
            )
    expected = {f"validation.epoch-{row['epoch'] - 1:04d}.json" for row in rows}
    if {p.name for p in shared.glob("validation.epoch-*.json")} != expected:
        raise ValueError("curve adapter contains validation epochs outside committed history")
    render_curves(shared)
    outputs = [output, shared / "curves_manifest.json"]
    outputs.extend(shared / name for name in read_json(shared / "curves_manifest.json")["outputs"])
    receipt = {
        "source_commit": source,
        "run_id": identity["run_id"],
        "history_sha256": sha256_file(root / "history.json"),
        "journal_sha256": sha256_file(root / "epoch_state.json"),
        "outputs": {str(p.relative_to(root)): sha256_file(p) for p in outputs},
        "training_reduction": "arithmetic_mean_over_optimizer_updates",
        "missing_metric_policy": "blank_csv_nan_plot_gap; reasons retained in history.json",
    }
    atomic_json(root / "curves_receipt.json", receipt)
    return receipt
