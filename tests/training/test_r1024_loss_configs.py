"""New selection configs preserve scientific factors and exact fifty epochs."""

import json
from pathlib import Path

import pytest

from gradpert.config.schema import ExperimentConfig


@pytest.mark.parametrize("row", ["t2", "p1", "p2", "c1", "l1", "l2", "l3"])
def test_only_selection_and_output_change(row):
    root = Path(__file__).resolve().parents[2] / "configs/r50"
    old = json.loads((root / f"r1024_{row}/gradpert_b2/nadig_jurkat.yaml").read_text())
    new = json.loads((root / f"r1024_loss_{row}_v1/gradpert_b2/nadig_jurkat.yaml").read_text())
    parsed = ExperimentConfig.model_validate(new)
    assert parsed.training.max_epochs.value == 50
    assert parsed.training.early_stopping is False
    assert parsed.training.monitor == "val/prediction_loss"
    assert parsed.training.monitor_mode == "min"
    new["training"]["monitor"] = old["training"]["monitor"]
    new["training"]["monitor_mode"] = old["training"]["monitor_mode"]
    new["artifacts"]["root"] = old["artifacts"]["root"]
    assert new == old


def test_reject_early_stop():
    root = Path(__file__).resolve().parents[2]
    d = json.loads(
        (root / "configs/r50/r1024_loss_t2_v1/gradpert_b2/nadig_jurkat.yaml").read_text()
    )
    d["training"]["early_stopping"] = True
    with pytest.raises(ValueError, match="without early stopping"):
        ExperimentConfig.model_validate(d)
