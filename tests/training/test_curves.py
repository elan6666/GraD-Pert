"""Curve artifacts retain numeric provenance and unavailable-value gaps."""

import csv
import json

from gradpert.hashing import sha256_file
from gradpert.training.curves import render_curves


def test_curve_outputs_and_missing_values(tmp_path):
    (tmp_path / "train_steps.csv").write_text(
        "global_step,total_loss,prediction_loss\n0,2,1\n1,1,0.5\n"
    )
    for epoch in range(2):
        (tmp_path / f"validation.epoch-{epoch:03d}.json").write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "global_step": epoch + 1,
                    "run_id": "fixture",
                    "source_commit": "a" * 40,
                    "prediction_loss": 1 / (epoch + 1),
                    "txpert_macro_pearson_delta": 0.2,
                    "trishift_pearson_delta": None,
                    "systema_pearson": 0.1,
                }
            )
        )
    render_curves(tmp_path)
    manifest = json.loads((tmp_path / "curves_manifest.json").read_text())
    assert len(manifest["outputs"]) == 5
    assert all(sha256_file(tmp_path / f) == h for f, h in manifest["outputs"].items())
    with (tmp_path / "validation_curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["trishift_pearson_delta"] == ""
    assert rows[0]["prediction_loss"] == "1.0"
    assert (tmp_path / "training_loss.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "validation_curves.pdf").read_bytes().startswith(b"%PDF")
