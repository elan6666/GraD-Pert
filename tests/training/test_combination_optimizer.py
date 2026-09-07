import pytest
import torch

from gradpert.modeling.modules import GraDPertJointModel
from gradpert.training.step import build_native_optimizer


def test_combination_rate_is_explicit_and_preserves_v1_guard():
    model = GraDPertJointModel(graph_gene_count=7, expression_gene_count=5, prototype_count=8192)
    with pytest.raises(ValueError, match="frozen"):
        build_native_optimizer(model, learning_rate=1e-7)
    optimizer = build_native_optimizer(
        model, learning_rate=1e-7, allow_combination_learning_rate=True
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-7
    assert optimizer.param_groups[0]["weight_decay"] == 0
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            build_native_optimizer(
                model, learning_rate=invalid, allow_combination_learning_rate=True
            )
    with pytest.raises(ValueError, match="frozen"):
        build_native_optimizer(model, weight_decay=0.01, allow_combination_learning_rate=True)
