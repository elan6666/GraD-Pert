import copy

import numpy as np
import pytest
import torch
from test_components import fixture

from gradpert.training.v2.joint_validation import evaluate_joint_loss
from gradpert.training.v2.objective import JointObjective


def test_complete_joint_validation_is_reproducible_and_read_only():
    model, batch = fixture()
    objective = JointObjective(
        model,
        lambda1=1,
        lambda2=1,
        ssl1_weights=(0.8, 0.4, 0.1),
        ssl2_weights=(0.8, 0.4, 0.1),
        loss_reduction="row_mean",
        koleo_exclude_same_condition=True,
    ).train()
    state = copy.deepcopy(objective.state_dict())
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    batches = [(batch, "batch-identity")]
    first = evaluate_joint_loss(objective, batches, bf16=False)
    second = evaluate_joint_loss(objective, batches, bf16=False)
    assert first == second
    assert first["joint_loss"] == pytest.approx(
        first["components"]["prediction"]
        + sum(
            weight * first["components"][f"ssl{stage}_{name}"]
            for stage, names, weights in (
                (1, ("condition", "node", "spread"), (0.8, 0.4, 0.1)),
                (2, ("dino", "ibot", "koleo"), (0.8, 0.4, 0.1)),
            )
            for name, weight in zip(names, weights, strict=True)
        )
    )
    assert first["cell_count"] == len(batch.control)
    assert first["batch_count"] == 1
    assert objective.training and not objective.pending
    for name, value in state.items():
        torch.testing.assert_close(objective.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    observed_numpy_rng = np.random.get_state()
    assert observed_numpy_rng[0] == numpy_rng[0]
    np.testing.assert_array_equal(observed_numpy_rng[1], numpy_rng[1])
    assert observed_numpy_rng[2:] == numpy_rng[2:]


def test_joint_validation_rejects_pending_center_updates():
    model, batch = fixture()
    objective = JointObjective(model)
    objective.pending["ssl1_cls"] = [torch.ones(2, model.options.prototypes)]
    with pytest.raises(ValueError, match="pending"):
        evaluate_joint_loss(objective, [(batch, "batch-identity")], bf16=False)
