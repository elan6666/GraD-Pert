from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from gradpert.config import NativeArchitectureOptions  # noqa: E402
from gradpert.modeling import GraDPertJointModel  # noqa: E402
from gradpert.modeling.modules import AdaptiveGeneGraphEncoder  # noqa: E402


def test_historical_gat_e3_initializes_student_and_teacher():
    options = replace(
        NativeArchitectureOptions.from_parameters({}),
        gene_feature_mode="genept_initialized",
        genept_expected_sha256="a" * 64,
    )
    matrix = torch.arange(21, dtype=torch.float32).reshape(7, 3)
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
        architecture=options,
        genept_matrix=matrix,
    )
    generator = torch.Generator().manual_seed(20260828)
    expected = matrix @ (torch.randn(3, 128, generator=generator) / (3**0.5))
    assert isinstance(model.student_encoder, AdaptiveGeneGraphEncoder)
    assert torch.equal(model.student_encoder.gene_embeddings, expected)
    assert torch.equal(model.teacher_encoder.gene_embeddings, expected)
    assert model.student_encoder.gene_embeddings.requires_grad
    assert not model.teacher_encoder.gene_embeddings.requires_grad
    model.student_encoder.gene_embeddings.square().sum().backward()
    assert model.student_encoder.gene_embeddings.grad is not None
    with pytest.raises(ValueError, match="finite"):
        GraDPertJointModel(
            graph_gene_count=7,
            expression_gene_count=5,
            prototype_count=8192,
            architecture=options,
        )
