"""Read-only probes distinguish target collapse and leave subsequent updates exact."""

import copy
import json
import math

import pytest
import torch
from test_components import fixture

from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer
from gradpert.training.v2.process_diagnostics import (
    DistributionSummary,
    FeatureSummary,
    evaluate_graph_gradient_diagnostics,
    evaluate_process_diagnostics,
    prediction_summary,
)


@pytest.mark.parametrize("kind", ["uniform", "sharp_constant", "sample_specific"])
def test_distribution_entropy_and_sample_dependence(kind):
    stats = DistributionSummary(8, max_samples=8)
    x = torch.zeros(8, 8)
    if kind == "sharp_constant":
        x[:, 0] = 200
    elif kind == "sample_specific":
        x = torch.eye(8) * 200
    stats.add(x[:3])
    stats.add(x[3:])
    result = stats.result()
    assert result["samples"] == 8
    if kind == "uniform":
        assert result["mean_sample_entropy"] == pytest.approx(math.log(8))
        assert result["sample_dependence_js"] == pytest.approx(0, abs=1e-12)
    elif kind == "sharp_constant":
        assert result["mean_sample_entropy"] < 1e-6
        assert result["sample_dependence_js"] < 1e-6
        assert result["largest_argmax_fraction"] == 1
    else:
        assert result["sample_dependence_js"] == pytest.approx(math.log(8))
        assert result["argmax_prototypes_used"] == 8


def test_feature_constant_rank_streaming_and_bounds():
    constant = FeatureSummary(10, 4)
    constant.add(torch.ones(100, 100))
    result = constant.result()
    assert result["samples"] == 10
    assert result["original_dimensions"] == 100
    assert result["analyzed_dimensions"] == 4
    assert result["covariance_effective_rank"] == 0
    varied = FeatureSummary(8, 4)
    x = torch.cat((torch.eye(4), -torch.eye(4)))
    varied.add(x[:3])
    varied.add(x[3:])
    assert varied.result()["covariance_effective_rank"] == pytest.approx(4)


def test_prediction_negative_output_and_actual_condition_weighted_tail():
    control = torch.ones(4, 4)
    prediction = control.clone()
    prediction[0, 0] = -1
    truth = prediction + torch.tensor([1, 1, 1, 3])[:, None]
    result = prediction_summary(
        prediction, control, truth, torch.tensor([0, 0, 0, 1]), strategy="condition_mean"
    )
    assert result["negative_prediction_fraction"] == 1 / 16
    assert result["negative_prediction_mean_absolute"] == 1
    assert result["prediction_mse"] == pytest.approx(5)
    assert result["top_one_percent_loss_fraction"] == pytest.approx(0.225)


def _exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _exact(a, b)
    else:
        assert left == right


def test_complete_objective_probe_preserves_multistep_optimizer_teacher_center_and_rng():
    model, batch = fixture()
    objective = JointObjective(model, loss_reduction="row_mean").train()
    versions = [copy.deepcopy(objective), copy.deepcopy(objective)]
    snapshots = []
    for index, version in enumerate(versions):
        torch.manual_seed(431)
        optimizer = V2Optimizer(version.student, 0.001, 0)
        for _ in range(2):
            optimizer_step(
                version, optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False
            )
            if index:
                state = copy.deepcopy(version.state_dict())
                grads = [None if p.grad is None else p.grad.clone() for p in version.parameters()]
                rng = torch.get_rng_state().clone()
                result = evaluate_process_diagnostics(version, batch, bf16=False, max_samples=8)
                json.dumps(result, allow_nan=False)
                assert result["split"] == "train" and result["selection_use"] == "none"
                assert result["features"]["student.cell.output.genes"]["samples"] == 8
                assert result["distributions"]["teacher.ssl1_cls"]["samples"] > 0
                pair = result["pairs"]["ssl2_cls"]
                assert pair["cross_entropy"] == pytest.approx(
                    pair["teacher_entropy"] + pair["teacher_to_student_kl"]
                )
                graph = evaluate_graph_gradient_diagnostics(version, batch, bf16=False)
                json.dumps(graph, allow_nan=False)
                assert graph["performed"] and graph["split"] == "train"
                assert graph["groups"]["graph"]["weighted_gradient_l2"]["ssl1"] > 0
                _exact(version.state_dict(), state)
                _exact(torch.get_rng_state(), rng)
                _exact([p.grad for p in version.parameters()], grads)
                assert version.training and not version.teacher.training and not version.pending
        snapshots.append(
            (copy.deepcopy(version.state_dict()), optimizer.state_dict(), torch.get_rng_state())
        )
    _exact(*snapshots)


def test_relay_interventions_preserve_ids_modes_order_settings_and_reproducibility():
    from test_relay_method import relay_training_fixture

    objective, batch = relay_training_fixture(checkpointed=False)
    objective.train()
    modes = [
        (m.training, getattr(m, "randomize_order", None), getattr(m, "eval_seed", None))
        for m in objective.modules()
    ]
    state = copy.deepcopy(objective.state_dict())
    rng = torch.get_rng_state().clone()
    positions = batch.query_positions.clone()
    one = evaluate_process_diagnostics(
        objective, batch, bf16=False, max_samples=8, intervention_budget=3
    )
    two = evaluate_process_diagnostics(
        objective, batch, bf16=False, max_samples=8, intervention_budget=3
    )
    one.pop("elapsed_seconds")
    two.pop("elapsed_seconds")
    _exact(one, two)
    _exact(state, objective.state_dict())
    _exact(rng, torch.get_rng_state())
    _exact(positions, batch.query_positions)
    assert modes == [
        (m.training, getattr(m, "randomize_order", None), getattr(m, "eval_seed", None))
        for m in objective.modules()
    ]
    assert one["interventions"]["changed_order_seed"]["delta_rms_change"] > 0


def test_pending_updates_rejected_and_observer_cleaned_after_failure(monkeypatch):
    from gradpert.training.v2 import objective as module

    model, batch = fixture()
    objective = JointObjective(model).train()
    original = module.cross_entropy_terms
    objective.pending["ssl1_cls"] = [torch.zeros(2, 11)]
    with pytest.raises(ValueError, match="pending"):
        evaluate_process_diagnostics(objective, batch, bf16=False)
    assert objective.pending  # Reject without discarding an uncommitted real update.
    objective.pending.clear()

    def failed_forward(*args, **kwargs):
        raise RuntimeError("probe failure")

    monkeypatch.setattr(objective, "forward", failed_forward)
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="probe failure"):
        evaluate_process_diagnostics(objective, batch, bf16=False)
    assert module.cross_entropy_terms is original
    assert objective.training and not objective.pending
    assert not model.cell._forward_hooks
    _exact(rng, torch.get_rng_state())
