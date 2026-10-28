"""Synthetic CPU calibration tests: exact gradients, fixed eight and state closure."""

import copy
import random
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from gradpert.training.v2.objective import GraphView, TrainingBatch
from scripts.v2.calibrate_response_anchor import (
    calibrate_first_eight,
    calibrate_gradients,
    prediction_losses,
    preserved_calibration,
)
from scripts.v2.capacity_probe import validation_state


def test_bootstrap_masks_cuda_before_native_imports():
    from scripts.v2.calibrate_response_anchor import bootstrap_worker_visibility

    environment = {"LOCAL_RANK": "1", "WORLD_SIZE": "2", "CUDA_VISIBLE_DEVICES": "0,1"}
    bootstrap_worker_visibility(["--gpu", "0,1", "--bf16"], environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    with pytest.raises(ValueError, match="before CUDA imports"):
        bootstrap_worker_visibility(["--gpu", "0"], environment)


def _metadata():
    return [dict(step=i, ordered_identity_sha256=f"{i:064x}") for i in range(8)]


def test_median_of_norms_rule_not_median_of_ratios_and_preserves_grads():
    p = nn.Parameter(torch.tensor([1.0, 2.0]))
    p.grad = torch.tensor([7.0, 9.0])
    result = calibrate_gradients(
        [p],
        lambda i: [p[0] * (i + 1)],
        lambda i: [p[0] * 2 * (8 - i)],
        _metadata(),
        P=4,
        steps_per_epoch=2,
    )
    assert result["median_prediction_gradient_norm"] == 4.5
    assert result["median_unit_anchor_gradient_norm"] == 4.5
    assert result["selected_weight"] == pytest.approx(0.1)
    np.testing.assert_array_equal(p.grad.numpy(), [7, 9])
    assert result["optimizer_updates"] == result["teacher_updates"] == result["center_updates"] == 0


def test_gradient_vectors_sum_before_norm_and_readout_is_excluded():
    response = nn.Parameter(torch.tensor([2.0, 3.0]))
    head = nn.Parameter(torch.tensor(4.0))
    result = calibrate_gradients(
        [response],
        lambda _: [(response * torch.tensor([3.0, 4.0])).sum()],
        lambda _: [2 * response[0] * head, -response[0] * head],
        _metadata(),
        P=2,
        steps_per_epoch=1,
    )
    assert result["median_prediction_gradient_norm"] == 5
    assert result["median_unit_anchor_gradient_norm"] == 2
    assert result["selected_weight"] == pytest.approx(0.25)
    assert head.grad is None and response.grad is None


def test_rank_mean_gradient_before_norm_not_mean_of_norms():
    p = nn.Parameter(torch.ones(2))
    calls = []

    # Local prediction vector(3,0), remote(0,4); local anchor(2,0), remote(-2,2).
    def average(value):
        remote = (
            value.new_tensor([0.0, 4.0]) if len(calls) % 2 == 0 else value.new_tensor([-2.0, 2.0])
        )
        calls.append(value.clone())
        return (value + remote) / 2

    result = calibrate_gradients(
        [p],
        lambda _: [3 * p[0]],
        lambda _: [p[0]],
        _metadata(),
        P=1,
        steps_per_epoch=1,
        world=2,
        reduce_mean=average,
    )
    assert result["median_prediction_gradient_norm"] == 2.5
    assert result["median_unit_anchor_gradient_norm"] == 1
    assert result["selected_weight"] == 0.25
    assert len(calls) == 16


def test_empty_rank_groups_can_reduce_to_nonzero_global_gradient_and_weight_caps():
    p = nn.Parameter(torch.tensor(1.0))
    result = calibrate_gradients(
        [p],
        lambda _: [p * 100],
        lambda _: [],
        _metadata(),
        P=2,
        steps_per_epoch=1,
        world=2,
        reduce_mean=lambda value: value / 2 + 0.1,
    )
    assert result["selected_weight"] == 1


@pytest.mark.parametrize("bad", ["zero_pred", "zero_anchor", "nan_loss", "short", "duplicate"])
def test_invalid_fixed_batch_fails_without_resampling(bad):
    p = nn.Parameter(torch.tensor(1.0))
    seen = []

    def pred(i):
        seen.append(i)
        if bad == "zero_pred" and i == 2:
            return [p * 0]
        if bad == "nan_loss" and i == 2:
            return [p * float("nan")]
        return [p]

    metadata = _metadata()
    if bad == "short":
        metadata.pop()
    if bad == "duplicate":
        metadata[-1]["ordered_identity_sha256"] = metadata[0]["ordered_identity_sha256"]
    with pytest.raises(ValueError):
        calibrate_gradients(
            [p],
            pred,
            lambda i: [p * 0 if bad == "zero_anchor" and i == 2 else p],
            metadata,
            P=2,
            steps_per_epoch=1,
        )
    assert seen == ([] if bad in {"short", "duplicate"} else [0, 1, 2])


def test_nonfinite_backward_fails_even_when_loss_finite():
    class BadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            return grad * float("inf")

    p = nn.Parameter(torch.tensor(1.0))
    with pytest.raises(ValueError, match="gradient"):
        calibrate_gradients(
            [p],
            lambda _: [BadBackward.apply(p)],
            lambda _: [p],
            _metadata(),
            P=2,
            steps_per_epoch=1,
        )


@dataclass
class _Spec:
    perturbed_row_ids: tuple
    control_row_ids: tuple
    condition_ids: tuple


class _Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.response = nn.Linear(2, 2)
        self.prediction = nn.Linear(2, 2)
        self.response_anchor_readout = nn.Linear(2, 2)
        self.options = SimpleNamespace(control_conditioned_graph=False)

    def encode_response(self, graph, control, condition, **kwargs):
        return {"prediction": self.prediction(self.response(control))}


class _Objective(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = _Student()
        self.teacher = copy.deepcopy(self.student)
        self.register_buffer("ssl_center", torch.ones(2))
        self.pending = {"original": torch.tensor(3.0)}
        self.loss_reduction = "row_mean"
        self.prediction_error_power = 2
        self.prediction_reduction_override = "inherit"
        self.prediction_enabled = True
        self.population_response = False

    def _batch_graph(self, model, batch, **kwargs):
        return batch.control.new_zeros(2, 2), None

    def _select_graph(self, graph, positions):
        return graph

    def response_metadata(self, batch):
        return {}


class _Optimizer:
    def __init__(self):
        self.steps = 0

    def state_dict(self):
        return dict(steps=self.steps)


def _runtime(count=9):
    obj = _Objective()
    graph = GraphView(torch.tensor([0, 1]), *[torch.zeros(0, dtype=torch.long) for _ in range(6)])
    batches = [
        TrainingBatch(
            graph,
            torch.tensor([0, 1]),
            torch.arange(10).reshape(5, 2).float() / 10 + i / 10,
            torch.ones(5, 2) * 0.1,
            torch.zeros(5, dtype=torch.long),
            (),
            (),
        )
        for i in range(count)
    ]
    specs = tuple(
        _Spec(
            tuple(f"t{i}_{j}" for j in range(5)), tuple(f"c{i}_{j}" for j in range(5)), ("p",) * 5
        )
        for i in range(count)
    )
    data = SimpleNamespace(training_batch_identity_specs=lambda **_: specs)
    runtime = SimpleNamespace(
        objective=obj,
        optimizer=_Optimizer(),
        generator=np.random.default_rng(8),
        device=torch.device("cpu"),
        data=data,
        purpose="training",
        options=SimpleNamespace(microbatch=2, max_conditions=0),
        batch_size=5,
        steps_per_epoch=150,
    )
    runtime.consumed = []

    def stream(epoch):
        assert epoch == 0
        for i, batch in enumerate(batches):
            runtime.consumed.append(i)
            runtime.generator.random()
            data.control_only_training = False
            yield batch

    runtime.batches = stream

    class Controller:
        bank = SimpleNamespace(condition_ids=("p0", "p1", "p2"))

        def groups(self, step):
            return (step % 3,)

        def group_loss(self, batch, condition, epoch, *, bf16):
            assert epoch == 0
            z = obj.student.response(batch.control).mean(0)
            return (obj.student.response_anchor_readout(z) - (condition + 1)).square().mean()

    return runtime, Controller(), batches


def test_native_prediction_micro_grad_matches_whole_row_mean():
    runtime, _, batches = _runtime()
    params = tuple(runtime.objective.student.response.parameters())
    grads = [torch.zeros_like(p) for p in params]
    for loss in prediction_losses(runtime, batches[0], bf16=False, world=1, global_rows=5):
        for target, value in zip(grads, torch.autograd.grad(loss, params), strict=True):
            target.add_(value)
    model = runtime.objective.student
    reference_loss = (
        (model.prediction(model.response(batches[0].control)) - batches[0].truth).square().mean()
    )
    for actual, reference in zip(grads, torch.autograd.grad(reference_loss, params), strict=True):
        torch.testing.assert_close(actual, reference)


def test_exact_first_eight_runtime_batches_restore_all_protected_state():
    runtime, controller, _ = _runtime()
    runtime.objective.eval()
    for p in runtime.objective.parameters():
        p.grad = torch.ones_like(p) * 3
    before = validation_state(runtime)
    result = calibrate_first_eight(runtime, controller)
    assert runtime.consumed == list(range(8))
    assert len(result["batches"]) == 8
    assert result["state_restored"]
    assert validation_state(runtime) == before
    assert runtime.optimizer.steps == 0
    assert not hasattr(runtime.data, "control_only_training")
    assert all(
        row["global_rows"] == 5 and row["local_group_count"] == 1 for row in result["batches"]
    )


@pytest.mark.parametrize("failure", [False, True])
def test_pending_buffers_modes_rng_and_existing_gradients_restored_on_error(failure):
    runtime, _, _ = _runtime()
    runtime.objective.eval()
    before = validation_state(runtime)
    try:
        with preserved_calibration(runtime):
            runtime.objective.pending.clear()
            runtime.objective.ssl_center.add_(1)
            runtime.objective.student.response.weight.grad = torch.ones(2, 2)
            random.random()
            np.random.rand()
            torch.rand(3)
            runtime.generator.random()
            if failure:
                raise LookupError("synthetic failure")
    except LookupError:
        assert failure
    assert validation_state(runtime) == before


def test_short_epoch_and_existing_optimizer_step_rejected():
    runtime, controller, _ = _runtime(count=7)
    with pytest.raises(ValueError):
        calibrate_first_eight(runtime, controller)
    assert not runtime.consumed
    runtime.optimizer.steps = 1
    with pytest.raises(ValueError, match="step zero"):
        calibrate_first_eight(runtime, controller)


def test_metadata_construction_also_inside_restored_rng_scope():
    runtime, controller, _ = _runtime()
    original = runtime.data.training_batch_identity_specs

    def specs(**kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        runtime.generator.random()
        return original(**kwargs)

    runtime.data.training_batch_identity_specs = specs
    before = validation_state(runtime)
    calibrate_first_eight(runtime, controller)
    assert validation_state(runtime) == before


def test_midstream_gradient_failure_restores_runtime_without_ninth_batch():
    runtime, controller, _ = _runtime()
    original = controller.group_loss
    calls = []

    def fail(batch, condition, epoch, *, bf16):
        calls.append(condition)
        loss = original(batch, condition, epoch, bf16=bf16)
        return loss * 0 if len(calls) == 3 else loss

    controller.group_loss = fail
    before = validation_state(runtime)
    with pytest.raises(ValueError, match="no resampling"):
        calibrate_first_eight(runtime, controller)
    assert runtime.consumed == [0, 1, 2]
    assert validation_state(runtime) == before


def test_formal_cli_rejects_local_paths_before_materialization(tmp_path):
    from scripts.v2.calibrate_response_anchor import execute

    args = SimpleNamespace(
        config=tmp_path / "c.yaml",
        data_root=tmp_path / "data",
        output=tmp_path / "out",
        publication=tmp_path / "pub.json",
        shared_policy=tmp_path / "policy.json",
    )
    with pytest.raises(ValueError, match="/data/yilangliu"):
        execute(args)
    assert not args.output.exists()
