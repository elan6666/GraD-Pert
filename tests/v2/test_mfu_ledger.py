"""Real native updates and independent hand arithmetic for the partial ledger."""

import copy

import numpy as np
import pytest
import torch
from mfu_ledger import ArithmeticLedger, arithmetic, counting_regions
from mfu_probe import capture_state, digest
from test_relay_method import relay_training_fixture

from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.optimizer import V2Optimizer


def test_matmul_forward_backward_hand_count_and_unknown_preserved():
    torch.manual_seed(6)
    x, w = torch.randn(3, 5, requires_grad=True), torch.randn(5, 7, requires_grad=True)
    counter = ArithmeticLedger()
    with counter:
        y = x @ w
        y.square().sum().backward()
    rows = counter.payload()["rows"]
    assert (
        sum(r["calls"] * r["flops_per_call"] for r in rows if r["operator"] == "aten.mm.default")
        == 3 * 2 * 3 * 5 * 7
    )
    assert any(
        r["operator"] == "aten.pow.Tensor_Scalar" and r["flops_per_call"] is None for r in rows
    )
    assert counter.payload()["mfu"] is counter.payload()["hfu"] is None
    assert counter.payload()["unsupported_calls"] > 0


def test_batched_and_fp8_padded_shape_formula_without_fake_native_fp8():
    a, b = torch.empty(2, 3, 5), torch.empty(2, 5, 7)
    flops, _ = arithmetic(torch.ops.aten.bmm.default, (a, b), torch.empty(2, 3, 7))
    assert flops == 2 * 2 * 3 * 5 * 7
    # Shape-only formula contract. No FP8 GEMM or backend success is claimed.
    flops, status = arithmetic(
        torch.ops.aten._scaled_mm.default,
        (torch.empty(16, 32), torch.empty(32, 48)),
        torch.empty(16, 48),
    )
    assert flops == 2 * 16 * 32 * 48 and status == "padded_gemm_only"


def test_dispatch_does_not_call_a_registered_decomposition():
    calls = []

    @torch.library.custom_op("mfu_test::once", mutates_args=())
    def once(x: torch.Tensor) -> torch.Tensor:
        calls.append("native")
        return x.clone()

    from torch._decomp import register_decomposition

    @register_decomposition(torch.ops.mfu_test.once.default)
    def decomposed(x):
        calls.append("decomposed")
        return x.clone()

    try:
        with ArithmeticLedger() as ledger:
            result = once(torch.ones(4))
        assert torch.equal(result, torch.ones(4))
        assert calls == ["native"]
        assert any(
            r["operator"] == "mfu_test.once.default" and r["flops_per_call"] is None
            for r in ledger.payload()["rows"]
        )
    finally:
        from torch._decomp import decomposition_table

        decomposition_table.pop(torch.ops.mfu_test.once.default)


def test_nonfloating_shape_ops_are_zero_but_solve_and_cast_are_unknown():
    with ArithmeticLedger() as ledger:
        integer = torch.arange(4)
        integer + 1
        floating = torch.ones(2, 2)
        floating.to(torch.bfloat16)
        torch.linalg.solve_triangular(floating.tril() + torch.eye(2), floating, upper=False)
    rows = ledger.payload()["rows"]
    assert any(r["coverage"] == "non_floating" and r["flops_per_call"] == 0 for r in rows)
    assert any(
        "linalg_solve_triangular" in r["operator"] and r["flops_per_call"] is None for r in rows
    )
    assert any("_to_copy" in r["operator"] and r["flops_per_call"] is None for r in rows)


def test_scatter_reduction_is_unknown_instead_of_assignment_zero():
    with ArithmeticLedger() as ledger:
        index, values = torch.tensor([0, 0]), torch.tensor([2.0, 3.0])
        result = torch.scatter(torch.zeros(2), 0, index, values, reduce="add")
        torch.scatter(torch.zeros(2), 0, index, values)
    assert result.tolist() == [5.0, 0.0]
    rows = ledger.payload()["rows"]
    assert any(r["operator"] == "aten.scatter.reduce" and r["flops_per_call"] is None for r in rows)
    assert any(r["operator"] == "aten.scatter.src" and r["flops_per_call"] == 0 for r in rows)


def native_runtime(objective):
    from types import SimpleNamespace

    return SimpleNamespace(
        objective=objective,
        optimizer=V2Optimizer(objective.student, 0.001, 0),
        data=SimpleNamespace(),
        index=SimpleNamespace(),
        generator=np.random.default_rng(8),
    )


def test_observer_exact_three_positive_lr_checkpointed_native_updates_and_cleanup():
    import torch.utils.checkpoint as checkpoint_module

    objective, batch = relay_training_fixture(checkpointed=True)
    original_checkpoint = checkpoint_module.checkpoint
    original_backward = torch.autograd.backward
    states, outputs = [], []
    for observing in (False, True):
        runtime = native_runtime(copy.deepcopy(objective))
        torch.manual_seed(114)
        ledger = ArithmeticLedger()
        from contextlib import nullcontext
        from types import SimpleNamespace

        with (
            counting_regions(runtime, ledger) if observing else nullcontext(),
            ledger if observing else nullcontext(),
        ):
            metrics = []
            for i in range(3):
                ledger.update = i
                metrics.append(
                    optimizer_step(
                        runtime.objective,
                        runtime.optimizer,
                        batch,
                        microbatch=2,
                        lr=0.001,
                        momentum=0.99,
                        bf16=False,
                    )
                )
        source = SimpleNamespace(state_dict=lambda: dict(offset=3))
        # Independent native-state values, not the implementation's aggregate.
        states.append(
            (
                copy.deepcopy(runtime.objective.state_dict()),
                copy.deepcopy(runtime.optimizer.state_dict()),
                {
                    n: None if p.grad is None else p.grad.clone()
                    for n, p in runtime.objective.named_parameters()
                },
                torch.get_rng_state(),
            )
        )
        outputs.append(metrics)
        assert "forward" not in vars(runtime.objective.student.graph)
        assert "step" not in vars(runtime.optimizer)
        if observing:
            rows = ledger.payload()["rows"]
            assert any(r["recompute"] and (r["flops_per_call"] or 0) > 0 for r in rows)
            assert any(r["phase"] == "optimizer" for r in rows)
            assert any(r["owner"].startswith("teacher.") for r in rows)
            assert not any(
                r["phase"] == "backward" and r["owner"].startswith("teacher.") for r in rows
            )
            assert capture_state(runtime, source)["pending"] == {}
    assert digest(states[0]) == digest(states[1])
    assert outputs[0] == outputs[1]
    assert checkpoint_module.checkpoint is original_checkpoint
    assert torch.autograd.backward is original_backward


def test_nested_repeated_checkpoint_replay_and_exception_cleanup():
    import torch.utils.checkpoint as checkpoints

    objective, _ = relay_training_fixture(checkpointed=True)
    runtime = native_runtime(objective)
    checkpoint = checkpoints.checkpoint
    ledger = ArithmeticLedger()
    with counting_regions(runtime, ledger), ledger:
        x = torch.ones(3, 3, requires_grad=True)

        def inner(value):
            return checkpoints.checkpoint(lambda v: (v @ v).square(), value, use_reentrant=False)

        y = checkpoints.checkpoint(inner, x, use_reentrant=False)
        y.sum().backward(retain_graph=True)
        y.sum().backward()
    replay = [
        r for r in ledger.payload()["rows"] if r["recompute"] and r["operator"] == "aten.mm.default"
    ]
    assert replay and all(type(r["recompute"]) is bool for r in replay)
    assert sum(r["calls"] for r in replay) >= 2
    with pytest.raises(RuntimeError, match="injected"), counting_regions(runtime, ledger), ledger:
        raise RuntimeError("injected")
    assert checkpoints.checkpoint is checkpoint
    assert "step" not in vars(runtime.optimizer)
