from pathlib import Path

from gradpert.config.loader import load_experiment_config
from gradpert.config.native import NativeArchitectureOptions


def test_b0_preserves_historical_b2_geometry_and_requested_overrides():
    root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        root
        / "configs/combinations/b0_historical_b2_e3_schedule_batch128"
        / "gradpert_b2/nadig_jurkat.yaml"
    )
    architecture = NativeArchitectureOptions.from_parameters(config.model.parameters)
    assert architecture.graph_axis_policy == "canonical_full"
    assert architecture.graph_encoder_family == "adaptive_relation_gat"
    assert architecture.graph_sources == ("string", "go")
    assert architecture.local_view_count == 8
    assert architecture.legacy_local_view_node_budget == 512
    assert architecture.legacy_local_anchor_mask_count == 4
    assert architecture.gene_feature_mode == "genept_initialized"
    assert architecture.decoder_mode == "additive"
    parameters = config.model.parameters
    assert [
        parameters[name].value
        for name in (
            "prediction_loss_weight",
            "condition_consistency_loss_weight",
            "masked_node_loss_weight",
            "spread_loss_weight",
        )
    ] == [1.0, 1.0, 0.1, 0.1]
    assert config.training.train_batch_size.value == 128
    assert config.training.eval_batch_size.value == 256
    assert config.training.max_epochs.value == 100
    assert config.training.early_stopping
    assert config.training.early_stopping_patience.value == 10
    assert config.training.run_seeds == [1]
    assert config.training.scheduler.value == {
        "name": "warmup_cosine_restarts",
        "interval": "epoch",
        "first_cycle_steps": 15,
        "cycle_mult": 2,
        "max_lr": 1e-4,
        "min_lr": 1e-6,
        "warmup_steps": 5,
        "gamma": 0.9,
    }
    assert config.artifacts.result_mode == "metrics_only"


def test_b1_changes_only_gene_prior_and_artifact_root():
    root = Path(__file__).resolve().parents[2] / "configs/combinations"
    b0 = load_experiment_config(
        root / "b0_historical_b2_e3_schedule_batch128/gradpert_b2/nadig_jurkat.yaml"
    ).model_dump()
    b1 = load_experiment_config(
        root / "b1_historical_b2_schedule_batch128/gradpert_b2/nadig_jurkat.yaml"
    ).model_dump()
    assert b1["model"]["parameters"]["gene_feature_mode"]["value"] == "learned_id"
    b0["model"]["parameters"]["gene_feature_mode"]["value"] = "learned_id"
    for name in ("genept_expected_sha256", "genept_artifact_path"):
        del b0["model"]["parameters"][name]
    b0["artifacts"]["root"] = b1["artifacts"]["root"]
    assert b0 == b1
