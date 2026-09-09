import math
from pathlib import Path

import pytest

from gradpert.config.loader import load_experiment_config
from gradpert.config.native import NativeArchitectureOptions
from gradpert.config.step_schedule import load_training_schedule


@pytest.mark.parametrize(
    ("row", "base"),
    [
        ("b0", "b0_historical_b2_e3_schedule_batch128"),
        ("b1", "b1_historical_b2_schedule_batch128"),
        ("a1", "a1_e3_l1_m1"),
    ],
)
def test_new_defaults_preserve_original_model(row, base):
    root = Path(__file__).resolve().parents[2] / "configs/combinations"
    old = load_experiment_config(root / base / "gradpert_b2/nadig_jurkat.yaml")
    new = load_experiment_config(
        root / f"{row}_original_step_batch512_loss111_epoch100/gradpert_b2/nadig_jurkat.yaml"
    )
    a = {k: v.value for k, v in old.model.parameters.items()}
    b = {k: v.value for k, v in new.model.parameters.items()}
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} <= {
        "teacher_ema_start",
        "condition_consistency_loss_weight",
        "masked_node_loss_weight",
    }
    assert [
        b[k + "_loss_weight"]
        for k in ["prediction", "condition_consistency", "masked_node", "spread"]
    ] == [1, 1, 1, 0.1]
    NativeArchitectureOptions.from_parameters(b)
    assert b.get("capacity_profile", "historical") == "historical"
    assert old.data == new.data and old.evaluation == new.evaluation
    assert new.training.max_epochs.value == 100
    assert new.training.train_batch_size.value == 512
    assert new.training.eval_batch_size == old.training.eval_batch_size
    assert new.training.optimizer == old.training.optimizer
    assert new.training.weight_decay == old.training.weight_decay
    assert new.training.early_stopping and new.training.early_stopping_patience.value == 10
    schedule = load_training_schedule(new.training.scheduler.value)
    assert schedule.max_lr == new.training.learning_rate.value == 2e-4 * math.sqrt(0.5)
    assert schedule.warmup_fraction == 0.16 and schedule.min_lr == 1e-6
    assert schedule.teacher_start == b["teacher_ema_start"] == 0.994
    assert schedule.teacher_end == 1
