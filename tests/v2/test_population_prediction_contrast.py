"""R18 synthetic signed-loss, atomic schedule, native RNG and receipt contracts."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_response_anchor_integration import integrated

from gradpert.training.v2.response_anchor import (
    PAIR_MODE,
    group_loss_coefficient,
    pair_schedule,
    pair_schedule_identity,
    prediction_contrast_loss,
)


def contrast_controller():
    objective, batch, optimizer, controller = integrated()
    del objective.student.response_anchor_readout
    del objective.teacher.response_anchor_readout
    controller.runtime.options.response_anchor_loss = PAIR_MODE
    controller.runtime.options.response_anchor_prediction_head_grad = True
    controller.population_targets = np.arange(1322 * 4, dtype=np.float32).reshape(1322, 4) / 100
    controller.groups = lambda step: ((0, 1),)
    return objective, batch, optimizer, controller


def test_signed_complete_means_swap_offset_cell_gradients_and_precision():
    p = torch.arange(12.0, requires_grad=True).reshape(4, 3)
    q = torch.arange(21.0, requires_grad=True).reshape(7, 3) / 3
    tp, tq = torch.tensor([2.0, 3.0, 1.0]), torch.tensor([4.0, 0.0, 2.0])
    loss = prediction_contrast_loss(p, q, tp, tq)
    torch.testing.assert_close(loss, prediction_contrast_loss(q, p, tq, tp))
    offset = torch.tensor([17.0, -9.0, 4.0])
    torch.testing.assert_close(loss, prediction_contrast_loss(p + offset, q + offset, tp, tq))
    error = (p.mean(0) - q.mean(0)) - (tp - tq)
    gp, gq = torch.autograd.grad(loss, (p, q))
    torch.testing.assert_close(gp, (2 * error / 3 / 4).expand_as(p))
    torch.testing.assert_close(gq, (-2 * error / 3 / 7).expand_as(q))
    assert prediction_contrast_loss(tp.expand(4, -1), tq.expand(7, -1), tp, tq) == 0
    assert prediction_contrast_loss(p.bfloat16(), q.bfloat16(), tp, tq).dtype == torch.float32
    # A negative signed target contrast remains negative; no abs/clamp.
    assert not torch.equal(loss, prediction_contrast_loss(p, q, tq, tp))
    for a, b, c, d in ((p[:3], q, tp, tq), (p, q, tp.requires_grad_(), tq)):
        with pytest.raises(ValueError):
            prediction_contrast_loss(a, b, c, d)


def test_two_match_coverage_atomic_rank_assignment_resume_and_hash():
    identity = pair_schedule_identity(1322, 150, 20261028)
    assert identity["connected_components"] == 1 and identity["distinct_edges"] == 1322
    assert (
        identity["schedule_sha256"]
        == "dfff6ba7002a7cb0d93e87f8afce4c8e8dbc81dd22329e99967da47b7a01be77"
    )
    for epoch in range(2):
        all_pairs = []
        for rank in range(2):
            plan = pair_schedule(1322, 150, epoch, 20261028, 2, rank)
            assert plan == pair_schedule(1322, 150, epoch, 20261028, 2, rank)
            all_pairs.extend(pair for step in plan for pair in step)
        assert len(all_pairs) == 661
        assert sorted(p for pair in all_pairs for p in pair) == list(range(1322))
    # Empty local steps are legal and still return a complete same-key metric map.
    assert any(not step for step in pair_schedule(4, 150, 0, 20261028, 2, 1))
    for args in ((5, 150, 0, 1), (4, 150, 2, 1), (4, 150, 0, 1, 2, 2)):
        with pytest.raises(ValueError):
            pair_schedule(*args)


def test_epoch_rank_mean_coefficient_is_pair_macro_mean():
    # Sum T updates then average explicit rank means equals pair population mean.
    coefficient = group_loss_coefficient(661, 150, 2, 0.3)
    values = torch.linspace(-1, 2, 1322, requires_grad=True)
    total = values.sum() * 0
    for rank in range(2):
        for units in pair_schedule(1322, 150, 0, 20261028, 2, rank):
            total = total + sum(
                ((values[p] - values[q]) ** 2 * coefficient for p, q in units), values.sum() * 0
            ) / (2 * 150)
    pairs = [p for step in pair_schedule(1322, 150, 0, 20261028) for p in step]
    expected = 0.3 * torch.stack([(values[p] - values[q]) ** 2 for p, q in pairs]).mean()
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(
        torch.autograd.grad(total, values)[0], torch.autograd.grad(expected, values)[0]
    )


@pytest.mark.parametrize("checkpoint,bf16", [(False, False), (True, False), (True, True)])
def test_native_two_graphs_head_gradients_rng_and_replay(checkpoint, bf16):
    objective, batch, _, controller = contrast_controller()
    # Force stochastic native operations and actual torch checkpoint recomputation.
    model = objective.student
    original = model.encode_response
    seen = []

    def encode(*args, **kwargs):
        seen.append(len(args[1]))

        def stochastic(control):
            return original(args[0], control * torch.rand_like(control), args[2], **kwargs)[
                "prediction"
            ]

        control = args[1]
        if checkpoint:
            from torch.utils.checkpoint import checkpoint as run_checkpoint

            pred = run_checkpoint(stochastic, control, use_reentrant=False)
        else:
            pred = stochastic(control)
        return {"prediction": pred}

    model.encode_response = encode
    teacher = copy.deepcopy(objective.teacher.state_dict())
    state = torch.get_rng_state().clone()
    results = []
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        metrics = controller.backward(batch, 0, bf16=bf16)
        results.append(
            (metrics, [None if p.grad is None else p.grad.clone() for p in model.parameters()])
        )
        assert torch.equal(state, torch.get_rng_state())
    assert results[0][0] == results[1][0]
    for a, b in zip(results[0][1], results[1][1], strict=True):
        if a is not None:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert seen == [
        4,
        4,
        4,
        4,
    ]  # exactly two group forwards per pair, checkpoint internal replay only
    assert metrics["response_anchor_groups"] == 2 and metrics["response_anchor_rows"] == 8
    assert (
        metrics["response_anchor_pairs"] == 1
        and metrics["response_anchor_pair_live_cells_max"] == 8
    )
    assert any(p.grad is not None and p.grad.abs().sum() for p in model.prediction.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() for p in model.response.parameters())
    assert not objective.pending
    assert all(p.grad is None for p in objective.teacher.parameters())
    for key, value in teacher.items():
        assert torch.equal(value, objective.teacher.state_dict()[key])
    controller.groups = lambda step: ()
    empty = controller.backward(batch, 0, bf16=bf16)
    assert set(empty) == set(metrics) and all(v == 0 for v in empty.values())


def test_same_ordered_gene_axis_complete_rows_and_forbidden_truth():
    _, batch, _, controller = contrast_controller()
    original = controller.bank.control_for_condition
    controller.bank.control_for_condition = lambda c: (
        np.repeat(original(c), 2, axis=0) if c else original(c)
    )
    controller.bank.control_slot_offsets = np.cumsum([0, 4, 8] + [4] * 1320)
    changed = replace(batch, truth=torch.full_like(batch.truth, float("nan")))
    details = {}
    with controller.unit_rng((0, 1), 0):
        loss = controller.unit_loss(changed, (0, 1), 0, diagnostics=details)
    assert details["group_rows"] == [4, 8]
    expected = ((controller.population_targets[0] - controller.population_targets[1]) ** 2).mean()
    assert np.isfinite(expected) and torch.isfinite(loss)
    # Excluded columns and a graph that silently permutes the output axis must fail.
    forbidden = replace(batch, query_positions=torch.tensor([0, 1, 2, 4]))
    with pytest.raises(ValueError, match="excluded"):
        controller.unit_loss(forbidden, (0, 1), 0)
    view = controller.runtime.index.view
    controller.runtime.index.view = lambda *a, **k: replace(
        view(*a, **k), ids=batch.graph.ids.flip(0)
    )
    with pytest.raises(ValueError, match="ordered query"):
        controller.unit_loss(batch, (0, 1), 0)
    controller.runtime.options.response_anchor_prediction_head_grad = False
    with pytest.raises(ValueError, match="normal native"):
        controller.unit_loss(batch, (0, 1), 0)


def test_opt_in_self_contained_config_and_postfit_descriptor_identity():
    from types import SimpleNamespace

    from gradpert.config import load_experiment_config
    from gradpert.config.v2 import V2Options
    from gradpert.training.v2.runtime import _configure_response_anchor
    from scripts.v2.response_anchor_postfit import population_descriptor, population_options

    root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        root
        / "configs/v2/response_population_contrast_20261030"
        / "calibration/gradpert_v2/nadig_jurkat.yaml"
    )
    _, options = V2Options.parse_parameters(config.model.parameters)
    assert population_options(config) == options and options.response_anchor_loss == PAIR_MODE
    with pytest.raises(ValueError):
        replace(options, response_anchor_prediction_head_grad=False)
    runtime = SimpleNamespace(
        options=options,
        objective=SimpleNamespace(),
        identity={},
        data=None,
        purpose="evaluation",
        allowed_expression_ids=None,
    )
    _configure_response_anchor(runtime)
    assert runtime.identity["response_anchor"] == population_descriptor(options)
    assert not hasattr(runtime.objective, "response_anchor_backward")
    assert "response_anchor_readout" not in json.dumps(runtime.identity)


@pytest.mark.parametrize("checkpoint,bf16", [(False, False), (True, True)])
def test_first8_native_steps_calibration_restores_state_and_uses_pair_denominator(checkpoint, bf16):
    from test_population_head_calibration import native

    from scripts.v2.calibrate_response_anchor import calibrate_first_eight
    from scripts.v2.capacity_probe import validation_state

    runtime, controller, _ = native(checkpoint)
    runtime.options.response_anchor_loss = PAIR_MODE
    runtime.options.response_anchor_prediction_head_grad = True
    controller.population_targets = np.arange(1322 * 4, dtype=np.float32).reshape(1322, 4) / 100
    controller.groups = lambda step: ((0, 1),)
    runtime.objective.eval()
    for p in runtime.objective.parameters():
        p.grad = torch.full_like(p, 0.125)
    before = validation_state(runtime)
    result = calibrate_first_eight(runtime, controller, bf16=bf16)
    assert validation_state(runtime) == before
    assert len(result["batches"]) == 8 and len(result["pair_diagnostics"]) == 8
    assert [r["step"] for r in result["pair_diagnostics"]] == list(range(8))
    assert result["anchor_coefficient"] == 150 / 661
    assert result["selected_weight"] == min(
        1.0,
        0.1
        * result["median_prediction_gradient_norm"]
        / result["median_unit_anchor_gradient_norm"],
    )
    assert result["optimizer_updates"] == result["teacher_updates"] == result["center_updates"] == 0
    assert result["pair_replay_audit"]["complete"]
    assert result["pair_replay_audit"]["extra_group_forwards"] == 4
    json.dumps(result, allow_nan=False)


def _distributed_pair_empty(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    from gradpert.training.v2.distributed import average_gradients

    objective, batch, _, controller = contrast_controller()
    params = tuple(objective.student.parameters())
    with controller.unit_rng((0, 1), 0):
        expected = torch.autograd.grad(
            0.2 * 150 / 661 * controller.unit_loss(batch, (0, 1), 0), params, allow_unused=True
        )
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        controller.world, controller.rank = 2, rank
        controller.groups = lambda step: ((0, 1),) if rank == 0 else ()
        for p in params:
            p.grad = torch.ones_like(p)
        terms = controller.backward(batch, 0)
        assert terms["response_anchor_pairs"] == (1 if rank == 0 else 0)
        average_gradients(objective.student)
        for p, g in zip(params, expected, strict=True):
            torch.testing.assert_close(
                p.grad, torch.ones_like(p) + (g if g is not None else 0), rtol=2e-5, atol=2e-6
            )
    finally:
        dist.destroy_process_group()


def test_world2_empty_pair_rank_preserves_main_gradients(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_distributed_pair_empty, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def test_pair_native_checkpoint_resume_keeps_update_rng_and_head(tmp_path):
    from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
    from gradpert.training.v2.engine import optimizer_step

    objective, batch, optimizer, _ = contrast_controller()
    generator = np.random.default_rng(31)

    def step(obj, opt):
        return optimizer_step(obj, opt, batch, microbatch=1, lr=1e-3, momentum=0.99, bf16=False)

    step(objective, optimizer)
    path = tmp_path / "synthetic-pair.pt"
    save_checkpoint(
        path,
        objective,
        optimizer,
        identity={"scope": "synthetic-pair"},
        progress={"step": 1},
        generator=generator,
    )
    expected = step(objective, optimizer)
    state = copy.deepcopy(objective.state_dict())
    other, _, other_optimizer, _ = contrast_controller()
    progress = load_checkpoint(
        path, other, other_optimizer, identity={"scope": "synthetic-pair"}, generator=generator
    )
    assert progress["step"] == 1
    assert step(other, other_optimizer) == expected
    assert other_optimizer.steps == 2
    for name, value in state.items():
        torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)
