from pathlib import Path

import pytest

from gradpert.config.loader import load_experiment_config
from gradpert.config.step_schedule import EndpointLRWarmupCosine, load_training_schedule


@pytest.mark.parametrize(
    "row,changes",
    [
        ("g1_muon", {"optimizer"}),
        ("g2_schedule", {"scheduler"}),
        ("g3_combined", {"optimizer", "scheduler"}),
    ],
)
def test_glm_configs_change_only_registered_factors(row, changes):
    root = Path(__file__).resolve().parents[2] / "configs/r50"
    parent = load_experiment_config(root / "batch512/gradpert_b2/nadig_jurkat.yaml")
    candidate = load_experiment_config(root / row / "gradpert_b2/nadig_jurkat.yaml")
    before, after = parent.training.model_dump(), candidate.training.model_dump()
    assert {k for k in before if before[k] != after[k]} == changes
    assert parent.data == candidate.data
    assert parent.evaluation == candidate.evaluation
    assert parent.model == candidate.model
    assert candidate.training.max_epochs.value == 50
    assert candidate.training.early_stopping is False
    if "scheduler" in changes:
        assert isinstance(
            load_training_schedule(candidate.training.scheduler.value), EndpointLRWarmupCosine
        )
