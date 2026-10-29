"""Native GPM head protection: no mutation, preserved upstream Jacobians."""

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_relay_method import small_architecture, small_index
from test_response_anchor_integration import integrated

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.runtime import _configure_response_anchor

ROOT = Path(__file__).resolve().parents[2]
RECIPE = (
    ROOT
    / "configs/v2/response_population_head_protection_20261030/R17-GPM-HG0"
    / "gradpert_v2/nadig_jurkat.yaml"
)


def model(checkpoint=False):
    torch.manual_seed(21)
    arch = replace(
        small_architecture(),
        prediction_parameterization="softplus",
        checkpoint_layers=checkpoint,
        mlp_profile="restored",
        learned_genept_projection=True,
        genept_projection_activation="gelu",
        genept_hidden_width=16,
        genept_middle_width=12,
        ffn_variant="gelu",
        response_injection_residual=True,
    )
    return GraDPertV2(torch.randn(6, 12), arch).train()


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_native_forward_jacobian_existing_grad_and_main_update(checkpoint, bf16):
    base = model(checkpoint)
    protected = copy.deepcopy(base)
    torch.manual_seed(50)
    gene = torch.randn(4, 8, requires_grad=True)
    ctrl = torch.rand(3, 4, requires_grad=True)
    cond = torch.randn(3, 8, requires_grad=True)
    inputs = (gene, ctrl, cond)
    other = tuple(v.detach().clone().requires_grad_() for v in inputs)
    before = copy.deepcopy(protected.state_dict())
    flags = {name: p.requires_grad for name, p in protected.named_parameters()}
    for p in protected.prediction.parameters():
        p.grad = torch.full_like(p, 0.125)
    oldgrads = [p.grad.clone() for p in protected.prediction.parameters()]
    rng = torch.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        a = base.encode_response(*inputs)["prediction"]
    torch.set_rng_state(rng)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        b = protected.encode_response(*other, prediction_head_grad=False)["prediction"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def loss(x):
        return (x.float().mean(0) - 0.2).square().mean()

    torch.testing.assert_close(loss(a), loss(b), rtol=0, atol=0)
    loss(a).backward()
    loss(b).backward()
    for x, y in zip(inputs, other, strict=True):
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for (n, p), (q, v) in zip(
        base.response.named_parameters(), protected.response.named_parameters(), strict=True
    ):
        assert n == q
        torch.testing.assert_close(p.grad, v.grad, rtol=0, atol=0)
    assert any(p.grad is not None and p.grad.abs().sum() for p in protected.response.parameters())
    for p, old in zip(protected.prediction.parameters(), oldgrads, strict=True):
        torch.testing.assert_close(p.grad, old, rtol=0, atol=0)
    assert flags == {name: p.requires_grad for name, p in protected.named_parameters()}
    for key, value in before.items():
        torch.testing.assert_close(value, protected.state_dict()[key], rtol=0, atol=0)
    # Primary loss remains a normal head-training path after auxiliary protection.
    for p in protected.parameters():
        p.grad = None
    out = protected.encode_response(*(v.detach() for v in other))["prediction"]
    loss(out).backward()
    assert all(p.grad is not None for p in protected.prediction.parameters())
    opt = torch.optim.SGD(protected.parameters(), lr=0.01)
    old = [p.detach().clone() for p in protected.prediction.parameters()]
    opt.step()
    assert any(
        not torch.equal(p, v) for p, v in zip(protected.prediction.parameters(), old, strict=True)
    )


def test_native_functional_call_has_no_flag_mutation_even_on_error(monkeypatch):
    m = model()
    joint = torch.randn(2, 3, 16, requires_grad=True)
    flags = [p.requires_grad for p in m.prediction.parameters()]
    real = torch.func.functional_call

    def fail(module, state, args, **kwargs):
        assert flags == [p.requires_grad for p in m.prediction.parameters()]
        assert set(state) == set(module.state_dict())
        assert all(not v.requires_grad for v in state.values())
        real(module, state, args, **kwargs)
        raise RuntimeError("synthetic")

    monkeypatch.setattr(torch.func, "functional_call", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        m.decode_score(joint, prediction_head_grad=False)
    assert flags == [p.requires_grad for p in m.prediction.parameters()]
    assert all(p.grad is None for p in m.prediction.parameters())


def test_group_loss_routing_preserves_teacher_rng_and_main_accumulation():
    _, batch, _, controller = integrated()
    objective = JointObjective(model(True), 0, 0, loss_reduction="row_mean")
    controller.runtime.objective = objective
    controller.runtime.index = small_index(6)
    controller.runtime.options.response_anchor_loss = "population_prediction"
    controller.runtime.options.response_anchor_prediction_head_grad = False
    controller.population_targets = np.zeros((1322, 4), np.float32)
    before = copy.deepcopy(objective.teacher.state_dict())
    rng = torch.get_rng_state().clone()
    for p in objective.student.prediction.parameters():
        p.grad = torch.ones_like(p)
    grads = [p.grad.clone() for p in objective.student.prediction.parameters()]
    with controller.group_rng(0, 0):
        loss = controller.group_loss(batch, 0, 0)
        loss.backward()
    assert torch.equal(rng, torch.get_rng_state())
    assert not objective.pending
    assert all(p.grad is None for p in objective.teacher.parameters())
    for p, grad in zip(objective.student.prediction.parameters(), grads, strict=True):
        torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
    for k, v in before.items():
        torch.testing.assert_close(v, objective.teacher.state_dict()[k], rtol=0, atol=0)


def test_config_is_unit_calibration_and_strict_opt_in():
    config = load_experiment_config(RECIPE)
    _, options = V2Options.parse_parameters(config.model.parameters)
    assert options.response_anchor_weight == 1.0
    assert options.response_anchor_prediction_head_grad is False
    for value in (0, 1, "false", None):
        with pytest.raises(ValueError):
            replace(options, response_anchor_prediction_head_grad=value)
    for mode in ("mse", "cosine_direction"):
        with pytest.raises(ValueError):
            replace(options, response_anchor_loss=mode)
    with pytest.raises(ValueError):
        replace(options, response_anchor_weight=0)


def test_value_free_evaluation_identity_and_historical_default(monkeypatch):
    from gradpert.training.v2 import response_anchor_runtime

    _, options = V2Options.parse_parameters(load_experiment_config(RECIPE).model.parameters)

    def fail(*args, **kwargs):
        raise AssertionError("TRAIN read in evaluation")

    monkeypatch.setattr(response_anchor_runtime, "load_response_bank", fail)

    def identity(opts):
        rt = SimpleNamespace(
            options=opts,
            objective=SimpleNamespace(),
            identity={},
            data=None,
            purpose="evaluation",
            allowed_expression_ids=None,
        )
        _configure_response_anchor(rt)
        assert not hasattr(rt.objective, "response_anchor_backward")
        return rt.identity["response_anchor"]

    r17 = identity(options)
    old = identity(replace(options, response_anchor_prediction_head_grad=True))
    assert r17.pop("prediction_head_grad") is False
    assert r17.pop("gradient_routing") == "gpm_head_parameters_detached_preserve_input_jacobian"
    assert r17 == old


def _rank_protected_gradients(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist

    from gradpert.training.v2.distributed import average_gradients

    _, batch, _, controller = integrated()
    objective = JointObjective(model(True), 0, 0, loss_reduction="row_mean")
    controller.runtime.objective = objective
    controller.runtime.index = small_index(6)
    controller.runtime.options.response_anchor_loss = "population_prediction"
    controller.runtime.options.response_anchor_prediction_head_grad = False
    controller.population_targets = np.zeros((1322, 4), np.float32)
    params = tuple(p for p in objective.student.parameters() if p.requires_grad)
    with controller.group_rng(0, 0):
        expected = torch.autograd.grad(
            0.2 * 150 / 1322 * controller.group_loss(batch, 0, 0), params, allow_unused=True
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
        controller.groups = lambda step: (0,) if rank == 0 else ()
        for p in params:
            p.grad = torch.ones_like(p)
        controller.backward(batch, 0)
        average_gradients(objective.student)
        for p, g in zip(params, expected, strict=True):
            torch.testing.assert_close(
                p.grad, torch.ones_like(p) + (g if g is not None else 0), rtol=2e-5, atol=2e-6
            )
        for p in objective.student.prediction.parameters():
            torch.testing.assert_close(p.grad, torch.ones_like(p), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_explicit_rank_average_preserves_main_head_with_empty_group_rank(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_rank_protected_gradients, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def test_protected_head_rejects_unsupported_native_decoder_metadata():
    config = load_experiment_config(RECIPE)
    for name, value in [("attention", "dense"), ("gene_conditioned_readout", True)]:
        params = dict(config.model.parameters)
        params[name] = replace_value = (
            params[name].model_copy(update={"value": value})
            if name in params
            else next(iter(params.values())).model_copy(update={"value": value})
        )
        assert replace_value.value == value
        with pytest.raises(ValueError):
            V2Options.parse_parameters(params)


def test_functional_reparameterization_restores_live_head_on_internal_forward_error(monkeypatch):
    m = model(True)
    parameters = dict(m.prediction.named_parameters())
    existing = {}
    for name, p in parameters.items():
        p.grad = torch.full_like(p, 0.25)
        existing[name] = p.grad.clone()

    def broken_forward(*args, **kwargs):
        # The original Parameter objects have not had their global flags changed.
        assert all(p.requires_grad for p in parameters.values())
        raise RuntimeError("inside original native prediction module")

    monkeypatch.setattr(m.prediction, "forward", broken_forward)
    with pytest.raises(RuntimeError, match="inside original native"):
        m.decode_score(torch.randn(2, 3, 16, requires_grad=True), prediction_head_grad=False)
    for name, p in m.prediction.named_parameters():
        assert p is parameters[name]
        assert p.requires_grad
        torch.testing.assert_close(p.grad, existing[name], rtol=0, atol=0)
