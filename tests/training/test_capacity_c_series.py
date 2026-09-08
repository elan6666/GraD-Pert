from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gradpert.config.loader import load_experiment_config  # noqa: E402
from gradpert.config.native import NativeArchitectureOptions  # noqa: E402
from gradpert.modeling import GraDPertJointModel  # noqa: E402


def scientific_values(value):
    if isinstance(value, dict):
        return {k: scientific_values(v) for k, v in value.items() if k != "reference"}
    if isinstance(value, list):
        return [scientific_values(v) for v in value]
    return value


@pytest.mark.parametrize(
    ("row", "depth", "hidden", "bottle", "total"),
    [
        ("c1", 3, 512, 32, 8359944),
        ("c2", 3, 512, 96, 10522760),
        ("c3", 4, 768, 96, 12839048),
    ],
)
def test_exact_capacity_and_frozen_scientific_factors(row, depth, hidden, bottle, total):
    root = Path(__file__).resolve().parents[2] / "configs/combinations"
    base = load_experiment_config(
        root / "b0_compact128_step_batch1024_epoch200/gradpert_b2/nadig_jurkat.yaml"
    )
    config = load_experiment_config(
        root / f"{row}_b0_capacity_step_batch1024_epoch200/gradpert_b2/nadig_jurkat.yaml"
    )
    parameters = {k: v.value for k, v in config.model.parameters.items()}
    original = {k: v.value for k, v in base.model.parameters.items()}
    changed = {
        k for k in parameters.keys() | original.keys() if parameters.get(k) != original.get(k)
    }
    assert changed <= {
        "capacity_profile",
        "graph_tower_layers",
        "projector_hidden_dim",
        "projector_bottleneck_dim",
    }
    assert scientific_values(config.training.model_dump()) == scientific_values(
        base.training.model_dump()
    )
    assert config.training.max_epochs.value == 200
    assert config.training.learning_rate.value == base.training.learning_rate.value
    assert config.training.scheduler.value == base.training.scheduler.value
    assert config.training.train_batch_size.value == 1024
    assert config.data == base.data and config.evaluation == base.evaluation
    architecture = NativeArchitectureOptions.from_parameters(parameters)
    model = GraDPertJointModel(
        graph_gene_count=6506,
        expression_gene_count=5000,
        prototype_count=16384,
        architecture=architecture,
        genept_matrix=torch.ones(6506, 8),
    )
    assert sum(p.numel() for p in model.parameters()) == total
    assert model.student_encoder.gene_embeddings.shape == (6506, 128)
    assert len(model.student_encoder.towers["go"].layers) == depth
    assert model.student_projector.mlp[0].out_features == hidden
    assert model.student_projector.mlp[-1].out_features == bottle
    assert model.basal_encoder.network[0].out_features == 128
    assert model.expression_decoder.network[0].out_features == 128
    parameters["basal_hidden_dim"] = 256
    with pytest.raises(ValueError):
        NativeArchitectureOptions.from_parameters(parameters)
