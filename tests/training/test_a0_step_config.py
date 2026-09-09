from pathlib import Path

from gradpert.config.loader import load_experiment_config
from gradpert.config.native import NativeArchitectureOptions
from gradpert.config.step_schedule import load_training_schedule


def test_a0_only_training_recipe_changes():
    root = Path(__file__).resolve().parents[2]
    original = load_experiment_config(
        root / "configs/ablations/nadig_jurkat/a0_ratio_ring_half/gradpert_b2/nadig_jurkat.yaml"
    )
    new = load_experiment_config(
        root / "configs/combinations/a0_step_batch1024_epoch100/gradpert_b2/nadig_jurkat.yaml"
    )
    before = {k: v.value for k, v in original.model.parameters.items()}
    after = {k: v.value for k, v in new.model.parameters.items()}
    assert {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)} == {
        "teacher_ema_start",
        "performance_pilot_variant",
    }
    NativeArchitectureOptions.from_parameters(after)
    assert after["gene_feature_mode"] == "learned_id"
    assert after["local_view_count"] == 4
    assert new.data == original.data and new.evaluation == original.evaluation
    assert new.training.max_epochs.value == 100
    assert new.training.train_batch_size.value == 1024
    assert new.training.eval_batch_size == original.training.eval_batch_size
    assert new.training.optimizer == original.training.optimizer
    assert new.training.weight_decay == original.training.weight_decay
    assert new.training.early_stopping and new.training.early_stopping_patience.value == 10
    schedule = load_training_schedule(new.training.scheduler.value)
    assert schedule.max_lr == 0.0002
    assert schedule.warmup_fraction == 0.16
    assert schedule.teacher_start == after["teacher_ema_start"] == 0.994
    assert schedule.teacher_end == 1.0
