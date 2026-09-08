from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gradpert.config.loader import load_experiment_config  # noqa: E402
from gradpert.config.native import NativeArchitectureOptions  # noqa: E402
from gradpert.modeling import GraDPertJointModel  # noqa: E402


@pytest.mark.parametrize("row", ["b0", "b1"])
def test_compact_complete_model_count_and_config(row):
    root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        root / f"configs/combinations/{row}_compact128_step_batch1024/gradpert_b2/nadig_jurkat.yaml"
    )
    parameters = {k: v.value for k, v in config.model.parameters.items()}
    architecture = NativeArchitectureOptions.from_parameters(parameters)
    # Synthetic initialization for structural testing, not scientific GenePT evidence.
    model = GraDPertJointModel(
        graph_gene_count=6506,
        expression_gene_count=5000,
        prototype_count=16384,
        architecture=architecture,
        genept_matrix=torch.ones(6506, 8) if row == "b0" else None,
    )
    assert sum(p.numel() for p in model.parameters()) == 6338568
    assert model.student_encoder.gene_embeddings.shape == (6506, 128)
    assert len(model.student_encoder.towers["go"].layers) == 2
    assert model.student_projector.mlp[0].out_features == 256
    assert all(not p.requires_grad for p in model.teacher_encoder.parameters())
    assert config.training.train_batch_size.value == 1024
    assert architecture.payload()["projector_bottleneck_dim"] == 32
    parameters["gene_embedding_dim"] = 64
    with pytest.raises(ValueError):
        NativeArchitectureOptions.from_parameters(parameters)


def test_historical_capacity_payload_unchanged():
    architecture = NativeArchitectureOptions.from_parameters({})
    assert "capacity_profile" not in architecture.payload()
    model = GraDPertJointModel(
        graph_gene_count=6506, expression_gene_count=5000, prototype_count=16384
    )
    assert sum(p.numel() for p in model.parameters()) == 30252744


def test_compact_pair_has_only_prior_difference():
    root = Path(__file__).resolve().parents[2] / "configs/combinations"
    rows = [
        load_experiment_config(
            root / f"{row}_compact128_step_batch1024/gradpert_b2/nadig_jurkat.yaml"
        )
        for row in ("b0", "b1")
    ]
    a, b = [{k: v.value for k, v in row.model.parameters.items()} for row in rows]
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {
        "gene_feature_mode",
        "genept_expected_sha256",
        "genept_artifact_path",
    }
    assert rows[0].training == rows[1].training
    assert rows[0].data == rows[1].data


@pytest.mark.parametrize("row", ["b0", "b1"])
def test_200_epoch_explicit_policy(row):
    root = Path(__file__).resolve().parents[2] / "configs/combinations"
    config = load_experiment_config(
        root / f"{row}_compact128_step_batch1024_epoch200/gradpert_b2/nadig_jurkat.yaml"
    )
    assert config.training.max_epochs.value == 200
    assert config.training.early_stopping_patience.value == 10
    assert config.training.formal_run_policy == "vnext_combination_200"
    invalid = config.training.model_dump()
    invalid["max_epochs"]["value"] = 100
    with pytest.raises(ValueError):
        config.training.__class__.model_validate(invalid)
