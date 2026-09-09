from pathlib import Path

import pytest
import yaml

from gradpert.config.loader import load_experiment_config
from gradpert.config.schema import ExperimentConfig

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("row,lr", [("ref", 0.001), ("lr_low", 0.0001), ("lr_mid", 0.0003)])
def test_r50_preserves_e3_except_budget_and_declared_lr(row, lr):
    old = load_experiment_config(
        ROOT / "configs/ablations/nadig_jurkat/e3_genept_initialized/gradpert_b2/nadig_jurkat.yaml"
    )
    new = load_experiment_config(ROOT / f"configs/r50/{row}/gradpert_b2/nadig_jurkat.yaml")
    a = {k: v.value for k, v in old.model.parameters.items()}
    b = {k: v.value for k, v in new.model.parameters.items()}
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {"performance_pilot_variant"}
    assert old.data == new.data and old.evaluation == new.evaluation
    assert new.training.formal_run_policy == "r50_selection"
    assert new.training.max_epochs.value == 50 and not new.training.early_stopping
    assert new.training.learning_rate.value == lr
    assert new.training.train_batch_size.value == 256
    assert new.training.scheduler.value == "none"
    assert [
        b[k + "_loss_weight"]
        for k in ["prediction", "condition_consistency", "masked_node", "spread"]
    ] == [1, 0.8, 0.4, 0.1]
    assert new.artifacts.result_mode == "metrics_only"


@pytest.mark.parametrize("key,value", [("early_stopping", True), ("run_seeds", [2])])
def test_r50_rejects_unregistered_screening_changes(key, value):
    raw = yaml.safe_load((ROOT / "configs/r50/ref/gradpert_b2/nadig_jurkat.yaml").read_text())
    raw["training"][key] = value
    with pytest.raises(ValueError, match="R50"):
        ExperimentConfig.model_validate(raw)


def test_r50_rejects_different_epoch_budget():
    raw = yaml.safe_load((ROOT / "configs/r50/ref/gradpert_b2/nadig_jurkat.yaml").read_text())
    raw["training"]["max_epochs"]["value"] = 10
    with pytest.raises(ValueError, match="exactly 50"):
        ExperimentConfig.model_validate(raw)
