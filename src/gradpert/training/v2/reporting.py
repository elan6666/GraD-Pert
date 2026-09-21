"""Small, source-bound epoch curves from committed v2 lifecycle history."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Any

from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file

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
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    epochs = [r["epoch"] for r in rows]
    fields = [
        ("train_prediction_step_mean", "Train prediction (mean over updates)"),
        ("train_joint_step_mean", "Train joint objective (mean over updates)"),
        ("validation_prediction_loss", "Validation (condition means)"),
    ]
    for axis, (field, label) in zip(list(axes.flat)[:3], fields, strict=True):
        axis.plot(epochs, [r[field] for r in rows])
        axis.set(xlabel="Epoch", ylabel="Loss", title=label)
    for axis, metric in zip(list(axes.flat)[3:], METRICS, strict=True):
        axis.plot(epochs, [r[metric] if r[metric] is not None else float("nan") for r in rows])
        axis.set(xlabel="Epoch", ylabel="Pearson", title=metric)
        if all(r[metric] is None for r in rows):
            axis.text(
                0.5,
                0.5,
                "No finite values\nSee history for reasons",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
    for axis in axes.flat:
        axis.set_xlim(epochs[0] - 0.1, epochs[-1] + 0.1)
        axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    figure.tight_layout()
    outputs = [output]
    for extension in ("png", "pdf"):
        destination = root / f"epoch_curves.{extension}"
        figure.savefig(destination)
        outputs.append(destination)
    plt.close(figure)
    receipt = {
        "source_commit": source,
        "run_id": identity["run_id"],
        "history_sha256": sha256_file(root / "history.json"),
        "journal_sha256": sha256_file(root / "epoch_state.json"),
        "outputs": {p.name: sha256_file(p) for p in outputs},
        "training_reduction": "arithmetic_mean_over_optimizer_updates",
        "missing_metric_policy": "blank_csv_nan_plot_gap; reasons retained in history.json",
    }
    atomic_json(root / "curves_receipt.json", receipt)
    return receipt
