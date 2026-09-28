from pathlib import Path

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options


def test_new_b0_changes_only_stream_count_and_selection_monitor() -> None:
    root = Path(__file__).resolve().parents[2] / "configs/v2"
    old = load_experiment_config(
        root / "source_key_gate_b0_jurkat/three_epoch_m74_a2/gradpert_v2/nadig_jurkat.yaml"
    )
    new = load_experiment_config(
        root / "no_mhc_joint_eval_jurkat/three_epoch_m74_a2/gradpert_v2/nadig_jurkat.yaml"
    )
    old_arch, old_options = V2Options.parse_parameters(old.model.parameters)
    new_arch, new_options = V2Options.parse_parameters(new.model.parameters)
    assert old_arch.streams == 4
    assert new_arch.streams == 1
    assert {**old_arch.__dict__, "streams": 1} == new_arch.__dict__
    assert new_options == old_options
    assert new.training.max_epochs.value == 3
    assert new.training.monitor == "val/joint_loss"


def test_lower_memory_b0_changes_only_microbatch() -> None:
    root = Path(__file__).resolve().parents[2] / "configs/v2/no_mhc_joint_eval_jurkat"
    full = load_experiment_config(root / "three_epoch_m74_a2/gradpert_v2/nadig_jurkat.yaml")
    fallback = load_experiment_config(root / "three_epoch_m64_a2/gradpert_v2/nadig_jurkat.yaml")
    full_arch, full_options = V2Options.parse_parameters(full.model.parameters)
    fallback_arch, fallback_options = V2Options.parse_parameters(fallback.model.parameters)
    assert fallback_arch == full_arch
    assert {**full_options.__dict__, "microbatch": 64} == fallback_options.__dict__
    assert fallback.training.train_batch_size.value == 256
    assert fallback.training.monitor == "val/joint_loss"
