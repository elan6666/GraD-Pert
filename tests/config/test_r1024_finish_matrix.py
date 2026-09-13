from pathlib import Path

import pytest

from gradpert.config import NativeArchitectureOptions, load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/r50/batch1024/gradpert_b2/nadig_jurkat.yaml"
ROWS = {
    "r1024_lr_low_retest": ("training", "learning_rate", 0.0001),
    "r1024_weight_condition_half": ("model", "condition_consistency_loss_weight", 0.4),
    "r1024_weight_masked_double": ("model", "masked_node_loss_weight", 0.8),
    "r1024_weight_spread_half": ("model", "spread_loss_weight", 0.05),
    "r1024_scale_projector4096": ("model", "projector_hidden_dim", 4096),
    "r1024_scale_graph6": ("model", "graph_tower_layers", 6),
}


def test_batch1024_repeat_is_identical_except_fresh_run_root() -> None:
    original = load_experiment_config(BASE)
    repeat = load_experiment_config(
        ROOT / "configs/r50/r1024_ref_repeat/gradpert_b2/nadig_jurkat.yaml"
    )
    assert repeat.model == original.model
    assert repeat.training == original.training
    assert repeat.data == original.data
    assert repeat.evaluation == original.evaluation
    assert repeat.artifacts.root != original.artifacts.root
    assert repeat.training.train_batch_size.value == 1024
    assert repeat.training.max_epochs.value == 50


def test_projector_width_changes_architecture_identity_but_not_legacy_default() -> None:
    original = load_experiment_config(BASE)
    expanded = load_experiment_config(
        ROOT / "configs/r50/r1024_scale_projector4096/gradpert_b2/nadig_jurkat.yaml"
    )
    parent = {key: field.value for key, field in original.model.parameters.items()}
    wider = {key: field.value for key, field in expanded.model.parameters.items()}
    default_arch = NativeArchitectureOptions.from_parameters(parent)
    wider_arch = NativeArchitectureOptions.from_parameters(wider)
    assert default_arch.projector_hidden_dim == 2048
    assert "projector_hidden_dim" not in default_arch.payload()
    assert default_arch.payload_sha256 == (
        "5249b555ea048684093530adc8f55b6cad8c915468a5daf3a0bb7daf09c917ab"
    )
    assert wider_arch.projector_hidden_dim == 4096
    assert wider_arch.payload()["projector_hidden_dim"] == 4096
    assert wider_arch.payload_sha256 != default_arch.payload_sha256


@pytest.mark.parametrize("row", ROWS)
def test_new_gra_d_pert_rows_change_one_scientific_value(row: str) -> None:
    base = load_experiment_config(BASE)
    config = load_experiment_config(ROOT / f"configs/r50/{row}/gradpert_b2/nadig_jurkat.yaml")
    section, key, expected = ROWS[row]
    model_changes = {
        name
        for name, parameter in config.model.parameters.items()
        if name not in base.model.parameters or parameter.value != base.model.parameters[name].value
    }
    assert model_changes == ({key} if section == "model" else set())
    if section == "model":
        assert config.model.parameters[key].value == expected
    else:
        assert config.training.learning_rate.value == expected
    assert config.training.train_batch_size.value == 1024
    assert config.training.max_epochs.value == 50
    assert config.training.eval_batch_size == base.training.eval_batch_size
    assert config.training.early_stopping is False
    assert (
        config.model.parameters["gene_feature_mode"] == base.model.parameters["gene_feature_mode"]
    )
    assert config.data == base.data
    if section == "model":
        assert config.training == base.training
    else:
        assert config.training.optimizer == base.training.optimizer
        assert config.training.scheduler == base.training.scheduler


def test_txpert_user_batch_override_is_explicit_and_other_science_unchanged() -> None:
    original = load_experiment_config(ROOT / "configs/r50-rerun/txpert_public/nadig_jurkat.yaml")
    override = load_experiment_config(
        ROOT / "configs/r50-r1024-finish/txpert_public/nadig_jurkat.yaml"
    )
    assert original.training.train_batch_size.value == 64
    assert override.training.train_batch_size.value == 1024
    assert override.model.parameters["official_train_batch_override"].value == 1024
    assert override.training.max_epochs.value == original.training.max_epochs.value == 50
    assert override.training.learning_rate == original.training.learning_rate
    assert override.training.optimizer == original.training.optimizer
    assert override.training.weight_decay == original.training.weight_decay
    assert override.training.eval_batch_size == original.training.eval_batch_size
    assert override.data == original.data
    assert {
        name
        for name, parameter in override.model.parameters.items()
        if name not in original.model.parameters
        or parameter.value != original.model.parameters[name].value
    } == {"official_train_batch_override"}
