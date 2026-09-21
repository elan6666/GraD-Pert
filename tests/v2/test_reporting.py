import csv

import pytest

from gradpert.data._io import atomic_json
from gradpert.training.v2.reporting import METRICS, export_curves


def test_curves_preserve_source_epoch_and_missing_metrics(tmp_path):
    atomic_json(
        tmp_path / "epoch_state.json",
        {"epoch": 2, "identity": {"source": {"commit": "a" * 40}, "run_id": "synthetic"}},
    )
    atomic_json(
        tmp_path / "history.json",
        [
            {
                "epoch": i,
                "optimizer_steps": i * 5,
                "training": {"prediction": 0.3 / i, "joint_loss": 0.5 / i},
                "validation": {
                    "split": "val",
                    "prediction_loss": 0.4 / i,
                    "metrics": [
                        {"metric_id": metric, "macro_mean": None if j == 1 else 0.1 * i}
                        for j, metric in enumerate(METRICS)
                    ],
                },
            }
            for i in (1, 2)
        ],
    )
    receipt = export_curves(tmp_path)
    with (tmp_path / "epoch_curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert rows[0]["trishift_pearson_delta"] == ""
    assert rows[0]["source_commit"] == "a" * 40
    assert len(receipt["outputs"]) == 7
    assert (tmp_path / "curves" / "training_loss.png").stat().st_size > 1000
    assert (tmp_path / "curves" / "validation_curves.pdf").stat().st_size > 1000


def test_uncommitted_history_is_rejected(tmp_path):
    atomic_json(tmp_path / "epoch_state.json", {"epoch": 1})
    atomic_json(tmp_path / "history.json", [])
    with pytest.raises(ValueError, match="committed"):
        export_curves(tmp_path)


def test_adapter_uses_native_curve_epoch_convention(tmp_path):
    test_curves_preserve_source_epoch_and_missing_metrics(tmp_path)
    with (tmp_path / "curves" / "validation_curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["epoch"] for r in rows] == ["0", "1"]
    assert [r["global_step"] for r in rows] == ["5", "10"]
    assert rows[0]["trishift_pearson_delta"] == ""
    with (tmp_path / "curves" / "train_steps.csv").open() as stream:
        training = list(csv.DictReader(stream))
    assert float(training[0]["prediction_loss_update_mean"]) == 0.3
