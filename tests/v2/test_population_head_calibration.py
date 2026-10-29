"""Fixed native synthetic head-route audit; no server data or fitted weights."""

import copy
import sys
from dataclasses import dataclass, replace
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
from scripts.v2.calibrate_response_anchor import (
    audit_population_head_route,
    calibrate_first_eight,
    compare_head_audit_gradients,
)
from scripts.v2.capacity_probe import validation_state

# The native script launcher imports sibling CLI modules by their script names.
# Match its existing lifecycle tests without depending on their collection order.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v2"))
from scripts.v2.run_population_calibration import require_head_route_audit


@dataclass
class Spec:
    control_row_ids: tuple[str, ...]


def native(checkpoint=False):
    _, batch, _, controller = integrated()
    torch.manual_seed(51)
    architecture = replace(
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
    runtime = controller.runtime
    runtime.index = small_index(6)
    batch = replace(
        batch,
        graph=runtime.index.view(
            np.arange(6),
            [[0], [1]],
            rng=np.random.default_rng(3),
            device=torch.device("cpu"),
            induced=False,
        ),
    )
    runtime.objective = JointObjective(
        GraDPertV2(torch.randn(6, 12), architecture), 0, 0, loss_reduction="row_mean"
    ).train()
    runtime.options.response_anchor_loss = "population_prediction"
    runtime.options.response_anchor_prediction_head_grad = False
    runtime.options.microbatch = 2
    runtime.options.max_conditions = 0
    runtime.options.response_anchor_population_sha256 = "a" * 64
    runtime.options.response_anchor_population_arrays_sha256 = "b" * 64
    runtime.generator = np.random.default_rng(52)
    runtime.purpose = "training"
    runtime.batch_size = len(batch.control)
    runtime.data.training_batch_identity_specs = lambda **kwargs: tuple(
        Spec(tuple(f"control-{step}-{i}" for i in range(len(batch.control)))) for step in range(8)
    )
    runtime.batches = lambda epoch: iter([batch] * 8)
    controller.population_targets = np.zeros((1322, 4), np.float32)
    return runtime, controller, batch


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_native_audit_restores_all_state_and_exports_only_scalars(checkpoint, bf16):
    runtime, controller, batch = native(checkpoint)
    for p in runtime.objective.parameters():
        p.grad = torch.full_like(p, 0.125)
    original_options = runtime.options
    before = validation_state(runtime)
    result = audit_population_head_route(
        runtime, controller, batch, 0, bf16=bf16, world=1, global_rows=len(batch.control)
    )
    assert runtime.options is original_options
    assert validation_state(runtime) == before
    assert result["complete"] and result["primary_head_gradient_preserved"]
    assert result["group_replays"] == 3
    assert all(isinstance(v, (str, int, float, bool, dict)) for v in result.values())
    for comparison in result["gradient_comparisons"].values():
        assert comparison["max_abs"] == 0
        assert comparison["atol"] == comparison["rtol"] == 0
    assert result["input_gradient_norm"] > 0
    assert result["response_gradient_norm"] > 0


@pytest.mark.parametrize("difference,passes", [(0.0, True), (1.9e-6, True), (2.1e-6, False)])
def test_preregistered_cuda_gradient_absolute_tolerance(difference, passes):
    actual, expected = [torch.tensor([difference])], [torch.tensor([0.0])]
    if passes:
        report = compare_head_audit_gradients(actual, expected, atol=2e-6)
        assert report["max_abs"] == pytest.approx(difference)
        assert report["atol"] == 2e-6 and report["rtol"] == 0
    else:
        with pytest.raises(ValueError, match="gradient audit differs"):
            compare_head_audit_gradients(actual, expected, atol=2e-6)


@pytest.mark.parametrize(
    "actual,expected,field",
    [
        ([None], [torch.ones(1)], "missing_mismatch"),
        ([torch.ones(1)], [torch.ones(2)], "shape_mismatch"),
        ([torch.tensor([float("nan")])], [torch.zeros(1)], "nonfinite_tensors"),
    ],
)
def test_gradient_comparison_rejects_missing_shape_and_nonfinite(actual, expected, field):
    with pytest.raises(ValueError, match=field):
        compare_head_audit_gradients(actual, expected, atol=2e-6)


def test_first_eight_calibration_keeps_same_primary_norms_and_weight():
    runtime, controller, _ = native()
    before = validation_state(runtime)
    protected = calibrate_first_eight(runtime, controller)
    assert validation_state(runtime) == before
    runtime.options.response_anchor_prediction_head_grad = True
    reference = calibrate_first_eight(runtime, controller)
    assert protected["batches"] == reference["batches"]
    assert protected["selected_weight"] == reference["selected_weight"]
    assert protected["population_head_route_audit"]["calibration_step"] == 0
    assert "population_head_route_audit" not in reference
    assert "response_anchor_prediction_head_grad" not in reference


def test_audit_replays_frozen_native_options_without_mutation():
    runtime, controller, batch = native()
    path = Path(__file__).resolve().parents[2] / (
        "configs/v2/response_population_head_protection_20261030/"
        "R17-GPM-HG0/gradpert_v2/nadig_jurkat.yaml"
    )
    _, options = V2Options.parse_parameters(load_experiment_config(path).model.parameters)
    runtime.options = replace(options, microbatch=2)
    original = runtime.options
    result = audit_population_head_route(
        runtime, controller, batch, 0, bf16=False, world=1, global_rows=len(batch.control)
    )
    assert result["complete"]
    assert runtime.options is original
    assert original.response_anchor_prediction_head_grad is False


@pytest.mark.parametrize("damage", ["ignore_flag", "detach_input", "clear_main_grad"])
def test_bad_native_gradient_routes_fail_closed_and_restore_state(monkeypatch, damage):
    runtime, controller, batch = native()
    head = runtime.objective.student.prediction
    if damage == "ignore_flag":
        original = runtime.objective.student.encode_response

        def incorrect(*args, **kwargs):
            kwargs.pop("prediction_head_grad", None)
            return original(*args, **kwargs)

        monkeypatch.setattr(runtime.objective.student, "encode_response", incorrect)
    elif damage == "detach_input":
        head.register_forward_pre_hook(lambda module, args: (args[0].detach(),))
    else:
        original = controller.group_loss
        calls = []

        def incorrect(*args, **kwargs):
            calls.append(1)
            loss = original(*args, **kwargs)
            if len(calls) == 3:
                for p in head.parameters():
                    p.grad = None
            return loss

        monkeypatch.setattr(controller, "group_loss", incorrect)
    before = validation_state(runtime)
    with pytest.raises((ValueError, RuntimeError)):
        audit_population_head_route(
            runtime,
            controller,
            batch,
            0,
            bf16=False,
            world=1,
            global_rows=len(batch.control),
        )
    assert validation_state(runtime) == before
    assert runtime.options.response_anchor_prediction_head_grad is False


def test_launcher_requires_protected_audit_from_both_ranks_only():
    runtime, controller, batch = native()
    audit = audit_population_head_route(
        runtime, controller, batch, 0, bf16=False, world=1, global_rows=len(batch.control)
    )
    # Simulate the server CUDA receipt's preregistered gradient tolerance;
    # actual CPU audit above continues to require exact equality.
    audit["gradient_comparisons"]["response"]["atol"] = 2e-6
    audit["gradient_comparisons"]["input"]["atol"] = 2e-6
    result = dict(
        response_anchor_prediction_head_grad=False,
        gradient_routing=audit["gradient_routing"],
        population_head_route_audit_ranks=[audit, copy.deepcopy(audit)],
    )
    require_head_route_audit({}, SimpleNamespace(response_anchor_prediction_head_grad=True))
    require_head_route_audit(result, runtime.options)
    for field in ("complete", "input_gradient_equal", "primary_head_gradient_preserved"):
        bad = copy.deepcopy(result)
        bad["population_head_route_audit_ranks"][1][field] = False
        with pytest.raises(ValueError, match="audit differs"):
            require_head_route_audit(bad, runtime.options)
    for scope in ("response", "input"):
        bad = copy.deepcopy(result)
        bad["population_head_route_audit_ranks"][1]["gradient_comparisons"][scope]["max_abs"] = (
            2.1e-6
        )
        with pytest.raises(ValueError, match="audit differs"):
            require_head_route_audit(bad, runtime.options)
    result["population_head_route_audit_ranks"].pop()
    with pytest.raises(ValueError, match="audit differs"):
        require_head_route_audit(result, runtime.options)
