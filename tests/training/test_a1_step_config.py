from pathlib import Path

from gradpert.config.loader import load_experiment_config
from gradpert.config.native import NativeArchitectureOptions
from gradpert.config.step_schedule import load_training_schedule


def test_a1_structure_preserved_with_new_training_recipe():
    root = Path(__file__).resolve().parents[2] / "configs/combinations"
    old = load_experiment_config(root / "a1_e3_l1_m1/gradpert_b2/nadig_jurkat.yaml")
    new = load_experiment_config(root / "a1_step_batch1024_epoch100/gradpert_b2/nadig_jurkat.yaml")
    a = {k: v.value for k, v in old.model.parameters.items()}
    b = {k: v.value for k, v in new.model.parameters.items()}
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {"teacher_ema_start"}
    NativeArchitectureOptions.from_parameters(b)
    assert b["gene_feature_mode"] == "genept_initialized"
    assert b["local_view_builder"] == "fanout"
    assert b["graph_sources"] == "string"
    assert b["graph_encoder_family"] == "single_source_gat"
    assert old.data == new.data and old.evaluation == new.evaluation
    assert new.training.max_epochs.value == 100
    assert new.training.train_batch_size.value == 1024
    assert new.training.eval_batch_size == old.training.eval_batch_size
    assert new.training.optimizer == old.training.optimizer
    assert new.training.weight_decay == old.training.weight_decay
    assert new.training.early_stopping and new.training.early_stopping_patience.value == 10
    schedule = load_training_schedule(new.training.scheduler.value)
    assert schedule.max_lr == 0.0002 and schedule.warmup_fraction == 0.16
    assert schedule.teacher_start == b["teacher_ema_start"] == 0.994
    assert schedule.teacher_end == 1.0
