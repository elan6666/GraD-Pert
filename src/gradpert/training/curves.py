"""Render training/validation curves from small immutable numeric receipts."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file


def render_curves(small_root: Path) -> None:
    """Keep raw values and independent metric axes; never replace missing with zero."""
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    training = small_root / "train_steps.csv"
    with training.open() as stream:
        steps = list(csv.DictReader(stream))
    files = sorted(small_root.glob("validation.epoch-*.json"))
    records = [json.loads(f.read_text()) for f in files]
    if not steps or not records:
        raise ValueError("curves require training and validation records")
    metrics = [
        "prediction_loss",
        *(["joint_loss"] if any("joint_loss" in record for record in records) else []),
        "txpert_macro_pearson_delta",
        "trishift_pearson_delta",
        "systema_pearson",
    ]
    columns = ["epoch", "global_step", "run_id", "source_commit", *metrics]
    csv_path = small_root / "validation_curves.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: record.get(key) for key in columns} for record in records)
    outputs = []
    fig, ax = plt.subplots(figsize=(10, 5))
    fields = [key for key in steps[0] if "loss" in key]
    if not fields:
        raise ValueError("training receipt has no loss fields")
    for key in fields:
        ax.plot(
            [int(r["global_step"]) for r in steps],
            [float(r[key]) for r in steps],
            label=key,
            linewidth=0.7,
        )
    ax.set(xlabel="Global step", ylabel="Training loss", title="Training losses (raw)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = small_root / f"training_loss.{ext}"
        fig.savefig(path)
        outputs.append(path)
    plt.close(fig)
    columns_count = 2
    rows_count = (len(metrics) + columns_count - 1) // columns_count
    fig, axes = plt.subplots(rows_count, columns_count, figsize=(11, 3.5 * rows_count))
    for ax, key in zip(axes.flat, metrics, strict=False):
        ax.plot(
            [r["epoch"] + 1 for r in records],
            [r.get(key) if r.get(key) is not None else float("nan") for r in records],
        )
        ax.set(xlabel="Epoch (1-based)", ylabel=key, title=key)
    for ax in list(axes.flat)[len(metrics) :]:
        ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = small_root / f"validation_curves.{ext}"
        fig.savefig(path)
        outputs.append(path)
    plt.close(fig)
    atomic_json(
        small_root / "curves_manifest.json",
        {
            "schema": "training-validation-curves-v1",
            "source_commit": records[-1].get("source_commit"),
            "run_id": records[-1].get("run_id"),
            "inputs": {f.name: sha256_file(f) for f in [training, *files]},
            "outputs": {f.name: sha256_file(f) for f in [csv_path, *outputs]},
            "missing_policy": "blank_csv_nan_plot_gap; reasons in per-epoch JSON",
        },
    )
