"""Fixed-update budgets retain real data epochs and native replay semantics."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_components import fixture

from gradpert.config import load_experiment_config
from gradpert.config.step_schedule import EndpointLRWarmupCosine
from gradpert.data._io import read_json
from gradpert.training.v2.lifecycle import fit
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer


def test_partial_final_epoch_is_exact_and_resumes_without_extra_batch(tmp_path):
    model, batch = fixture()
    initial = copy.deepcopy(model.state_dict())
    initial_rng = torch.get_rng_state().clone()

    def run(root, *, interrupt=False, resume=False):
        model.load_state_dict(initial)
        torch.set_rng_state(initial_rng)
        objective = JointObjective(model, lambda1=0, lambda2=0)
        optimizer = V2Optimizer(model, lr=0.001, weight_decay=0)
        rng = np.random.default_rng(19)
        consumed = []

        def batches(epoch):
            for i in range(3):
                if interrupt and epoch == 1 and i == 1:
                    raise RuntimeError("interrupted partial epoch")
                consumed.append((epoch, i))
                rng.random()
                yield batch

        journal = fit(
            objective,
            optimizer,
            root=root,
            identity={"source": "test"},
            generator=rng,
            epochs=2,
            steps_per_epoch=3,
            batches=batches,
            validate=None,
            schedule=EndpointLRWarmupCosine(0.001, 0.0002, 0.25),
            teacher_start=0.99,
            teacher_end=1,
            microbatch=2,
            bf16=False,
            max_updates=5,
            resume=resume,
        )
        assert optimizer.steps == 5
        progress = read_json(root / "live_progress.json")
        assert progress["optimizer_steps_completed"] == progress["optimizer_steps_total"] == 5
        assert progress["epoch_steps_total"] == 2
        assert progress["fraction_complete"] == 1
        history = read_json(root / "history.json")
        assert [x["optimizer_steps"] for x in history] == [3, 5]
        assert [x["updates_this_data_epoch"] for x in history] == [3, 2]
        assert [x["data_epoch_complete"] for x in history] == [True, False]
        assert journal["contract"]["optimizer_update_budget"] == 5
        assert consumed == (
            [(1, 0), (1, 1)] if resume else [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        )
        return copy.deepcopy(objective.state_dict()), rng.random()

    full, full_rng = run(tmp_path / "full")
    with pytest.raises(RuntimeError, match="interrupted partial"):
        run(tmp_path / "resumed", interrupt=True)
    actual, actual_rng = run(tmp_path / "resumed", resume=True)
    for k in full:
        torch.testing.assert_close(actual[k], full[k], rtol=0, atol=0)
    assert actual_rng == full_rng
    # A completed partial-epoch checkpoint is itself resumable/idempotent.
    # No science or test evaluator is invoked here.


@pytest.mark.parametrize("budget", [True, 0, 3, 7])
def test_update_budget_rejects_invalid_or_wrong_epoch_ceiling(tmp_path, budget):
    model, batch = fixture()
    with pytest.raises(ValueError, match="update budget"):
        fit(
            JointObjective(model, lambda1=0, lambda2=0),
            V2Optimizer(model, lr=0.001, weight_decay=0),
            root=tmp_path,
            identity={},
            generator=np.random.default_rng(1),
            epochs=2,
            steps_per_epoch=3,
            batches=lambda _: [batch] * 3,
            validate=None,
            schedule=EndpointLRWarmupCosine(0.001, 0.0002, 0.25),
            teacher_start=0.99,
            teacher_end=1,
            microbatch=2,
            bf16=False,
            max_updates=budget,
        )
    assert not list(tmp_path.iterdir())


def test_four_dataset_configs_are_explicit_and_do_not_retrain_jurkat():
    root = Path(__file__).resolve().parents[2]
    path = root / "configs/v2/dataset_transfer_300_20261102/manifest.json"
    manifest = json.loads(path.read_text())
    assert len(manifest["rows"]) == 4
    assert not manifest["jurkat_retrained"]
    for row in manifest["rows"]:
        c = load_experiment_config(root / row["config"])
        assert c.dataset_id == row["dataset"] != "nadig_jurkat"
        assert c.training.formal_run_policy == "v2_updates_300"
        assert (
            c.training.max_epochs.value
            == (300 + row["steps_per_data_epoch"] - 1) // row["steps_per_data_epoch"]
        )
        assert c.model.excludes_test_target_expression
        assert c.model.parameters["train_selection_path"].value == ""
        assert c.model.parameters["response_injection_residual"].value is True
