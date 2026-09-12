from pathlib import Path

import pytest
import torch

from gradpert.config.loader import load_experiment_config
from gradpert.training.prediction_loss import expression_loss


def test_condition_mean_is_error_mean_not_mean_expression():
    p = torch.tensor([[1.0], [-1.0], [3.0]], requires_grad=True)
    y = torch.zeros_like(p)
    loss = expression_loss(p, y, ("A+ctrl", "A+ctrl", "B+C"), reduction="condition_mean")
    assert loss.item() == 5
    loss.backward()
    assert torch.equal(p.grad, torch.tensor([[0.5], [-0.5], [3.0]]))
    assert expression_loss(p, y, ("A", "A", "B"), reduction="cell_mean").item() == pytest.approx(
        11 / 3
    )
    # MSE of condition expression means would give (0 + 9)/2, not 5.
    assert loss.item() != 4.5


@pytest.mark.parametrize("ids", [("ctrl", "ctrl", "A+B", "A+B"), ("A",) * 4])
def test_balanced_groups_and_permutation(ids):
    p = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    y = p * 0.3
    expected = expression_loss(p, y, ids, reduction="cell_mean")
    assert torch.allclose(expression_loss(p, y, ids, reduction="condition_mean"), expected)
    order = [3, 1, 0, 2]
    assert torch.allclose(
        expression_loss(p[order], y[order], [ids[i] for i in order], reduction="condition_mean"),
        expected,
    )


def test_bad_reduction_or_alignment_fails():
    p = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        expression_loss(p, p, ("A",), reduction="condition_mean")
    with pytest.raises(ValueError):
        expression_loss(p, p, ("A", "B"), reduction="other")


def test_c1_config_only_adds_prediction_reduction():
    root = Path(__file__).resolve().parents[2] / "configs/r50"
    parent = load_experiment_config(root / "batch512/gradpert_b2/nadig_jurkat.yaml")
    c1 = load_experiment_config(root / "c1_condition_mse/gradpert_b2/nadig_jurkat.yaml")
    params = dict(c1.model.parameters)
    assert params.pop("prediction_reduction").value == "condition_mean"
    assert params == parent.model.parameters
    assert c1.training == parent.training
    assert c1.data == parent.data and c1.evaluation == parent.evaluation
