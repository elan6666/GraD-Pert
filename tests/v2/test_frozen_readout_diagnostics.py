"""Frozen readout diagnostics observe the native head without changing training."""

import copy
import json
from dataclasses import replace

import pytest
import torch
from test_frozen_readout import fixture, update

from gradpert.training.v2.loss_diagnostics import evaluate_loss_diagnostics
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.process_diagnostics import (
    _read_only,
    evaluate_graph_gradient_diagnostics,
    evaluate_process_diagnostics,
)
from gradpert.training.v2.readout import HEAD_SHAPES, frozen_state_proof
from gradpert.training.v2.reductions import population_weights


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("updated", [False, True])
def test_native_prediction_head_gradients_and_complete_state_are_preserved(training, bf16, updated):
    obj, batch, opt = fixture()
    if updated:
        update(obj, batch, opt)
    obj.train(training)
    # Preserve individual module flags, including deliberately heterogeneous head modes.
    obj.student.prediction[1].training = not training
    parameters = tuple(obj.student.prediction.parameters())
    for i, p in enumerate(parameters):
        p.grad = torch.full_like(p, i + 0.5) if i % 2 else None
    before = copy.deepcopy(obj.state_dict())
    optimizer_before = copy.deepcopy(opt.state_dict())
    grads = [(p.grad, p.grad.clone() if p.grad is not None else None) for p in obj.parameters()]
    modes = [m.training for m in obj.modules()]
    rng = torch.get_rng_state().clone()
    with _read_only(obj, batch, 0x71A1):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            _, terms = obj(batch, include_ssl1=False)
        expected = torch.autograd.grad(terms["prediction"], parameters)
        expected_components = {k: float(v.detach()) for k, v in terms.items()}
    result = evaluate_loss_diagnostics(obj, batch, bf16=bf16)
    assert result["training_stage"] == "frozen_readout"
    assert result["shared_basal_weighted_gradient_l2"] is None
    assert result["shared_basal_gradient_status"] == "not_applicable:frozen_readout"
    assert result["prediction_ssl_gradient_cosine"] is None
    assert result["prediction_ssl_gradient_cosine_status"] == "not_applicable:ssl_disabled"
    assert result["prediction_head_weighted_gradient_l2"]["ssl"] is None
    assert result["diagnostic_objective_scope"]["unavailable_terms"] == {
        "ssl1": "disabled_in_frozen_readout",
        "ssl2": "disabled_in_frozen_readout",
    }
    assert result["components"] == expected_components
    actual = result["prediction_head_gradient_tensors"]
    assert [v["name"] for v in actual] == list(HEAD_SHAPES)
    for row, expected_gradient in zip(actual, expected, strict=True):
        assert row["finite"] is True
        assert row["l2"] == pytest.approx(float(expected_gradient.float().norm()))
    assert result["prediction_head_weighted_gradient_l2"]["prediction"] == pytest.approx(
        float(torch.cat([g.float().flatten() for g in expected]).norm())
    )
    assert result["selection_use"] == "none" and result["split"] == "train"
    assert not any(k.startswith("ssl") for k in result["components"])
    json.dumps(result, allow_nan=False)
    assert modes == [m.training for m in obj.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert not obj.pending
    for p, (old_grad, value) in zip(obj.parameters(), grads, strict=True):
        assert p.grad is old_grad
        if value is not None:
            assert torch.equal(p.grad, value)
    for name, value in before.items():
        assert torch.equal(obj.state_dict()[name], value), name
    from test_staged_joint_training import assert_nested_equal

    assert_nested_equal(opt.state_dict(), optimizer_before)
    # Restore the deliberately heterogeneous head mode before the scientific proof.
    obj.train(training)
    frozen_state_proof(obj, opt)


@pytest.mark.parametrize("strategy", ["row_mean", "condition_mean"])
def test_native_metrics_and_residual_tail_use_inference_outputs_and_configured_weights(strategy):
    obj, batch, _ = fixture()
    obj.prediction_reduction_override = strategy
    batch = replace(batch, condition_index=torch.tensor([0, 0, 0, 1]))
    with _read_only(obj, batch, 0x71A1), torch.no_grad():
        graph, conditions = obj._batch_graph(obj.student, batch)
        prediction = obj.student.encode_response(
            obj._select_graph(graph, batch.query_positions),
            batch.control,
            conditions,
            **obj.response_metadata(batch),
        )["prediction"]
    errors = torch.tensor([1.0, 2.0, 3.0, 9.0])[:, None].expand_as(prediction)
    batch = replace(batch, truth=prediction + errors)
    forwards = []
    handle = obj.student.prediction.register_forward_hook(
        lambda module, args, output: forwards.append(
            (torch.is_grad_enabled(), output.requires_grad)
        )
    )
    try:
        result = evaluate_loss_diagnostics(obj, batch, bf16=False)
    finally:
        handle.remove()
    assert forwards == [(True, True), (False, False)]
    assert result["prediction_mse"] == pytest.approx(float(errors.square().mean()))
    copy = float((batch.control - batch.truth).square().mean())
    assert result["control_copy_mse"] == pytest.approx(copy)
    assert result["mse_gain_over_copy"] == pytest.approx(
        1 - float(errors.square().mean()) / copy, abs=1e-7
    )
    weights = population_weights(batch.condition_index, torch.ones(4, dtype=torch.bool), strategy)
    contributions = (errors.square() * weights[:, None] / errors.shape[1]).flatten()
    assert result["components"]["prediction"] == pytest.approx(float(contributions.sum()))
    tail = max(1, (len(contributions) + 99) // 100)
    assert result["top_one_percent_loss_fraction"] == pytest.approx(
        float(contributions.topk(tail).values.sum() / contributions.sum())
    )
    assert list(result["absolute_residual_quantiles"].values()) == pytest.approx(
        errors.flatten().quantile(torch.tensor([0.5, 0.9, 0.99])).tolist()
    )


def test_pending_boundary_and_failed_probe_preserve_state(monkeypatch):
    obj, batch, _ = fixture()
    obj.pending["existing"] = [torch.tensor([2.0])]
    pending = obj.pending.copy()
    with pytest.raises(ValueError, match=r"pending|boundary"):
        evaluate_loss_diagnostics(obj, batch, bf16=False)
    assert obj.pending == pending
    obj.pending.clear()
    state = copy.deepcopy(obj.state_dict())
    modes = [m.training for m in obj.modules()]
    rng = torch.get_rng_state().clone()
    native = obj.student.encode_response
    calls = 0

    def fail_residual(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            torch.rand(3)
            raise RuntimeError("residual probe failure")
        return native(*args, **kwargs)

    monkeypatch.setattr(obj.student, "encode_response", fail_residual)
    with pytest.raises(RuntimeError, match="residual probe failure"):
        evaluate_loss_diagnostics(obj, batch, bf16=False)
    assert modes == [m.training for m in obj.modules()]
    assert torch.equal(rng, torch.get_rng_state()) and not obj.pending
    for name, value in state.items():
        assert torch.equal(obj.state_dict()[name], value), name
    assert all(p.grad is None for p in obj.parameters())


def test_runtime_process_and_graph_diagnostics_remain_valid_for_frozen_readout():
    obj, batch, opt = fixture()
    result = evaluate_process_diagnostics(obj, batch, bf16=False)
    assert result["outputs"]["prediction_mse"] >= 0
    assert evaluate_graph_gradient_diagnostics(obj, batch, bf16=False) == {
        "performed": False,
        "reason": "no_trainable_graph_parameters",
    }
    frozen_state_proof(obj, opt)


def test_bad_head_ownership_and_nonfinite_gradient_fail_closed():
    obj, batch, _ = fixture()
    obj.student.prediction[0].weight.requires_grad_(False)
    with pytest.raises(ValueError, match="exactly the four prediction tensors"):
        evaluate_loss_diagnostics(obj, batch, bf16=False)
    obj.student.prediction[0].weight.requires_grad_(True)
    state = copy.deepcopy(obj.state_dict())
    modes = [m.training for m in obj.modules()]
    rng = torch.get_rng_state().clone()
    bad_batch = replace(batch, truth=torch.full_like(batch.truth, float("nan")))
    with pytest.raises(ValueError, match="gradients must be finite"):
        evaluate_loss_diagnostics(obj, bad_batch, bf16=False)
    assert modes == [m.training for m in obj.modules()]
    assert torch.equal(rng, torch.get_rng_state()) and not obj.pending
    assert all(p.grad is None for p in obj.parameters())
    for name, value in state.items():
        assert torch.equal(obj.state_dict()[name], value), name


def test_unrelated_joint_objective_still_requires_shared_basal_parameters():
    obj, batch, _ = fixture()
    ordinary = JointObjective(obj.student, lambda1=0, lambda2=0, loss_reduction="row_mean")
    with pytest.raises(ValueError, match="diagnostics need shared basal parameters"):
        evaluate_loss_diagnostics(ordinary, batch, bf16=False)
