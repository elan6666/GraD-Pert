"""Opt-in metric copies preserve Python scalars and complete CPU updates."""

import copy
import math
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_relay_method import relay_training_fixture

from gradpert.training.v2 import engine
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer


def assert_float_bits(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        if math.isnan(b):
            assert math.isnan(a)
        else:
            assert a.hex() == b.hex()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_scalar_copies_preserve_extremes_signed_zero_and_nonfinites(dtype):
    finfo = torch.finfo(dtype)
    values = [0.0, -0.0, 1.0, -1.0, finfo.max, -finfo.max, finfo.tiny, -finfo.tiny]
    values += [float("inf"), float("-inf"), float("nan")]
    tensors = [torch.tensor(value, dtype=dtype) for value in values]
    tiny = torch.nextafter(torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype))
    tensors += [tiny, -tiny]
    assert_float_bits(engine._metric_scalar_values(tensors), [float(v.detach()) for v in tensors])


def test_mixed_scalar_dtypes_promote_before_stack_and_detach():
    leaf = torch.tensor(2.0, requires_grad=True)
    tensors = [
        torch.tensor(1.0000000000000002, dtype=torch.float64),
        torch.tensor(1e-200, dtype=torch.float64),
        torch.tensor(1.125, dtype=torch.bfloat16),
        torch.tensor(-0.0, dtype=torch.float32),
        torch.tensor(2**63 - 1, dtype=torch.int64),
        torch.tensor(True),
        (leaf * leaf).reshape(1, 1),
    ]
    assert_float_bits(engine._metric_scalar_values(tensors), [float(v.detach()) for v in tensors])
    assert leaf.grad is None
    assert engine._metric_scalar_values([]) == []


def test_python_fraction_multiply_and_accumulation_order_are_unchanged():
    events = [
        ({"prediction": torch.tensor(1.0000000000000002, dtype=torch.float64)}, 1 / 3),
        ({"prediction": torch.tensor(-0.125, dtype=torch.float16)}, 2 / 3),
        ({"prediction": torch.tensor(1e-200, dtype=torch.float64)}, 1 / 3),
    ]
    expected, actual = {}, {}
    for terms, fraction in events:
        for name, value in terms.items():
            expected[name] = expected.get(name, 0.0) + float(value.detach()) * fraction
        scalars = engine._metric_scalar_values(list(terms.values()))
        for (name, _), scalar in zip(terms.items(), scalars, strict=True):
            actual[name] = actual.get(name, 0.0) + scalar * fraction
        assert list(actual) == list(expected)
        assert_float_bits(list(actual.values()), list(expected.values()))


@pytest.mark.parametrize(
    "values",
    [
        [None],
        [torch.ones(2)],
        [torch.tensor(1.0), None],
        [torch.tensor(1 + 2j)],
        [torch.tensor(1.0), torch.empty((), device="meta")],
    ],
)
def test_unusual_inputs_retain_original_scalar_errors_without_stacking(values, monkeypatch):
    def unexpected_stack(*args, **kwargs):
        raise AssertionError("fallback must use original scalar conversions")

    monkeypatch.setattr(torch, "stack", unexpected_stack)
    with pytest.raises(Exception) as expected:
        [float(value.detach()) for value in values]
    with pytest.raises(type(expected.value)) as actual:
        engine._metric_scalar_values(values)
    assert str(actual.value) == str(expected.value)


def assert_tree_bits(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        a = actual.detach().contiguous().reshape(-1).view(torch.uint8)
        b = expected.detach().contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(a, b)
        assert not actual.is_floating_point() or torch.isfinite(actual).all()
    elif isinstance(actual, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(actual, dict):
        assert list(actual) == list(expected)
        for key in actual:
            assert_tree_bits(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_tree_bits(a, b)
    elif isinstance(actual, float):
        assert actual.hex() == expected.hex()
    else:
        assert actual == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_nonfinite_loss_still_fails_before_metric_copy_and_optimizer_update(enabled, monkeypatch):
    objective, batch = relay_training_fixture(checkpointed=False)
    optimizer = V2Optimizer(objective.student, 0.001, 0.0)

    def nonfinite_forward(*args, **kwargs):
        return torch.tensor(float("nan"), requires_grad=True), {}

    def unexpected_copy(values):
        raise AssertionError("finite-loss guard must run before metric copying")

    monkeypatch.setattr(objective, "forward", nonfinite_forward)
    monkeypatch.setattr(engine, "_metric_scalar_values", unexpected_copy)
    with pytest.raises(FloatingPointError, match="nonfinite v2 cell loss"):
        engine.optimizer_step(
            objective,
            optimizer,
            batch,
            microbatch=2,
            lr=0.001,
            momentum=0.99,
            bf16=False,
            batch_metric_scalars=enabled,
        )
    assert optimizer.steps == 0
    assert not objective.pending


@pytest.mark.parametrize("checkpointed", [False, True])
def test_complete_cpu_updates_preserve_gradients_teacher_centers_optimizer_and_rng(
    checkpointed, monkeypatch
):
    threads = torch.get_num_threads()
    torch_rng, numpy_rng, python_rng = (
        torch.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )
    torch.set_num_threads(1)
    try:
        torch.manual_seed(12)
        base, batch = relay_training_fixture(checkpointed=checkpointed)
        objective = JointObjective(
            base.student,
            ssl1_weights=(0.8, 0, 0),
            ssl2_weights=(0.8, 0.4, 0),
            loss_reduction="row_mean",
            au_alignment_weight=0.05,
            au_uniformity_weight=0.0,
            au_warmup_steps=30,
            cr_weight=0.00004,
            cr_warmup_steps=30,
            population_aux_space="ssl2_cls_bottleneck",
        )
        batch = replace(batch, context_index=torch.zeros(len(batch.control), dtype=torch.long))
        helper = engine._metric_scalar_values
        observed_sizes = []

        def record_values(values):
            observed_sizes.append(len(values))
            return helper(values)

        monkeypatch.setattr(engine, "_metric_scalar_values", record_values)
        histories = []
        for flag in (None, False, True):
            obj = copy.deepcopy(objective)
            optimizer = V2Optimizer(obj.student, 0.001, 0.0)
            torch.manual_seed(91)
            np.random.seed(91)
            random.seed(91)
            history = []
            for step in range(3):
                metrics = engine.optimizer_step(
                    obj,
                    optimizer,
                    batch,
                    microbatch=2,
                    lr=0.001,
                    momentum=0.99,
                    bf16=False,
                    **({} if flag is None else {"batch_metric_scalars": flag}),
                )
                assert optimizer.steps == step + 1 and not obj.pending
                assert all(p.grad is None for p in obj.teacher.parameters())
                history.append(
                    copy.deepcopy(
                        {
                            "metrics": metrics,
                            "objective": obj.state_dict(),
                            "gradients": {n: p.grad for n, p in obj.named_parameters()},
                            "optimizer": optimizer.state_dict(),
                            "pending": obj.pending,
                            "torch_rng": torch.get_rng_state(),
                            "numpy_rng": np.random.get_state(),
                            "python_rng": random.getstate(),
                        }
                    )
                )
            if flag is not True:
                assert not observed_sizes
            histories.append(history)
        assert observed_sizes and set(observed_sizes) == {6}
        assert_tree_bits(histories[1], histories[0])
        assert_tree_bits(histories[2], histories[0])
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(torch_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
