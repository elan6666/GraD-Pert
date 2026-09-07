from pathlib import Path

from gradpert.config import NativeArchitectureOptions, load_experiment_config


def test_a3_retains_e3_architecture_with_explicit_training_overrides():
    root = Path(__file__).resolve().parents[2] / "configs"
    e3 = load_experiment_config(
        root / "ablations/nadig_jurkat/e3_genept_initialized/gradpert_b2/nadig_jurkat.yaml"
    )
    a3 = load_experiment_config(
        root / "combinations/a3_a0_e3_lr1e7_batch128/gradpert_b2/nadig_jurkat.yaml"
    )
    expected = dict(e3.model.parameters)
    expected.pop("performance_pilot_variant")
    assert a3.model.parameters == expected
    assert a3.data == e3.data and a3.evaluation == e3.evaluation
    arch = NativeArchitectureOptions.from_parameters(a3.model.parameters)
    assert arch.graph_encoder_family == "multi_source_sparse_transformer"
    assert arch.graph_sources == ("string", "go")
    assert arch.local_view_builder == "ring_induced"
    assert arch.local_view_count == 4
    assert arch.graph_output_dim == 64 and arch.decoder_mode == "additive"
    assert arch.gene_feature_mode == "genept_initialized"
    assert a3.training.learning_rate.value == 1e-7
    assert a3.training.train_batch_size.value == 128
    assert a3.training.eval_batch_size == e3.training.eval_batch_size
    assert a3.training.max_epochs.value == 100
    assert a3.training.early_stopping
    assert a3.training.early_stopping_patience.value == 10
    assert a3.training.run_seeds == [1]
    assert a3.artifacts.result_mode == "metrics_only"
