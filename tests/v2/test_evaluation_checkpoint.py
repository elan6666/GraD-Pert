import copy

import numpy as np
import pytest
import torch
from test_components import fixture

from gradpert.hashing import sha256_file
from gradpert.training.v2.checkpoint import load_evaluation_checkpoint, save_checkpoint
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer


def test_evaluation_loads_weights_without_restoring_rng_or_optimizer(tmp_path):
    model, _ = fixture()
    objective = JointObjective(model, 0, 0)
    optimizer = V2Optimizer(model, 0.001, 0)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        objective,
        optimizer,
        identity={"source": "training"},
        progress={"epoch": 3},
        generator=np.random.default_rng(5),
    )
    expected = copy.deepcopy(objective.state_dict())
    with torch.no_grad():
        next(model.parameters()).add_(2)
    torch.manual_seed(123)
    rng = torch.get_rng_state().clone()
    progress = load_evaluation_checkpoint(
        path,
        objective,
        training_identity={"source": "training"},
        checkpoint_sha256=sha256_file(path),
    )
    assert progress == {"epoch": 3}
    assert optimizer.steps == 0
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for name, tensor in objective.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="checksum"):
        load_evaluation_checkpoint(
            path, objective, training_identity={"source": "training"}, checkpoint_sha256="wrong"
        )
    with pytest.raises(ValueError, match="identity"):
        load_evaluation_checkpoint(
            path,
            objective,
            training_identity={"source": "evaluation"},
            checkpoint_sha256=sha256_file(path),
        )
