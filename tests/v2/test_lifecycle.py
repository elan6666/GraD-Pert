import copy

import numpy as np
import pytest
import torch
from test_components import fixture

from gradpert.config.step_schedule import EndpointLRWarmupCosine
from gradpert.training.v2.lifecycle import fit
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer


def test_interrupted_epoch_resume_matches_uninterrupted_and_keeps_best_last(tmp_path):
    model, batch = fixture()
    original = copy.deepcopy(model.state_dict())

    def run(root, *, resume=False, interrupt=False):
        model.load_state_dict(original)
        objective = JointObjective(model, lambda1=0, lambda2=0)
        optimizer = V2Optimizer(model, lr=0.001, weight_decay=0)
        rng = np.random.default_rng(6)

        def batches(epoch):
            for step in range(3):
                # Consume view RNG as the real assembler does.
                rng.random()
                if interrupt and epoch == 1 and step == 1:
                    raise RuntimeError("simulated worker interruption")
                yield batch

        def validate():
            epoch = optimizer.steps // 3
            return {"split": "val", "prediction_loss": 0.2 if epoch == 1 else 0.3}

        journal = fit(
            objective,
            optimizer,
            root=root,
            identity={"source": "test"},
            generator=rng,
            epochs=3,
            steps_per_epoch=3,
            batches=batches,
            validate=validate,
            schedule=EndpointLRWarmupCosine(0.001, 0.0002, 0.16),
            teacher_start=0.99,
            teacher_end=1,
            microbatch=2,
            bf16=False,
            resume=resume,
        )
        state = copy.deepcopy(objective.state_dict())
        from gradpert.training.v2.lifecycle import test_selected as evaluate_selected

        calls = []

        def test():
            calls.append(1)
            return {"split": "test", "metric": 0.5}

        receipts = evaluate_selected(
            objective, root=root, evaluation_identity={"source": "test"}, test=test
        )
        assert set(receipts) == {"best", "last"}
        assert len(calls) == 2
        evaluate_selected(objective, root=root, evaluation_identity={"source": "test"}, test=test)
        assert len(calls) == 2
        return state, rng.random(), journal

    full, full_rng, _ = run(tmp_path / "full")
    with pytest.raises(RuntimeError, match="simulated"):
        run(tmp_path / "resumed", interrupt=True)
    resumed, resumed_rng, journal = run(tmp_path / "resumed", resume=True)
    for name in full:
        torch.testing.assert_close(full[name], resumed[name], rtol=0, atol=0)
    assert full_rng == resumed_rng
    assert journal["best"]["epoch"] == 1
    assert journal["last"]["epoch"] == 3
    assert (tmp_path / "resumed" / "best.pt").resolve().name == "epoch-0001.pt"
    assert (tmp_path / "resumed" / "last.pt").resolve().name == "epoch-0003.pt"
    assert len(list((tmp_path / "resumed").glob("epoch-*.pt"))) == 2


def test_first_epoch_failure_has_resume_checkpoint_and_no_test_access(tmp_path):
    from gradpert.training.v2.lifecycle import test_selected as evaluate_selected

    model, batch = fixture()
    objective = JointObjective(model, lambda1=0, lambda2=0)
    optimizer = V2Optimizer(model, lr=0.001, weight_decay=0)

    def fail():
        raise RuntimeError("validation interrupted")

    with pytest.raises(RuntimeError, match="validation interrupted"):
        fit(
            objective,
            optimizer,
            root=tmp_path,
            identity={"source": "test"},
            generator=np.random.default_rng(1),
            epochs=3,
            steps_per_epoch=3,
            batches=lambda _: [batch] * 3,
            validate=fail,
            schedule=EndpointLRWarmupCosine(0.001, 0.0002, 0.16),
            teacher_start=0.99,
            teacher_end=1,
            microbatch=2,
            bf16=False,
        )
    assert (tmp_path / "epoch-0000.pt").is_file()
    with pytest.raises(ValueError, match="complete training budget"):
        evaluate_selected(objective, root=tmp_path, evaluation_identity={}, test=fail)
