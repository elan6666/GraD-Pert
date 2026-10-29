"""TRAIN epoch0 first-eight gradient calibration; no optimizer/EMA/center update.

Formal CUDA execution requires a clean published source, allocator and sealed
resource admission. Local tests use synthetic CPU graphs only. No checkpoints,
validation/test responses, expression arrays or gradient arrays are persisted.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import statistics
import sys
import traceback
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any


def bootstrap_worker_visibility(argv: Sequence[str], environ: dict[str, str]) -> None:
    """Select the physical device before imports can initialize/cache CUDA."""
    if "LOCAL_RANK" not in environ:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default="0")
    early, _ = parser.parse_known_args(argv)
    devices = early.gpu.split(",")
    rank, world = int(environ["LOCAL_RANK"]), int(environ.get("WORLD_SIZE", "1"))
    if (
        len(devices) != world
        or len(set(devices)) != world
        or not set(devices) <= {"0", "1"}
        or not 0 <= rank < world
    ):
        raise ValueError("GPU list must exactly match torchrun world before CUDA imports")
    environ["CUDA_VISIBLE_DEVICES"] = devices[rank]


if __name__ == "__main__":
    bootstrap_worker_visibility(sys.argv[1:], os.environ)

# CUDA visibility must be frozen before importing the numerical stack.
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

_BATCHES = 8
_PROTECTED_ROUTE = "gpm_head_parameters_detached_preserve_input_jacobian"
_CUDA_RESPONSE_GRADIENT_ATOL = 2e-6


def _audit_tensor_summary(
    actual: Sequence[Tensor | None], expected: Sequence[Tensor | None], names: Sequence[str]
) -> dict[str, Any]:
    """Diagnostic scalars only; never an acceptance check or exported gradient."""
    if len(actual) != len(expected) or len(actual) != len(names):
        raise ValueError("diagnostic tensor/name counts differ")
    report: dict[str, Any] = dict(
        tensor_count=0,
        missing_both=0,
        missing_mismatch=0,
        shape_mismatch=0,
        dtype_mismatch=0,
        nonfinite_tensors=0,
        max_abs=0.0,
    )
    rows, delta_squared, reference_squared = [], 0.0, 0.0
    for name, left, right in zip(names, actual, expected, strict=True):
        row = dict(
            name=name,
            actual_shape=None if left is None else list(left.shape),
            expected_shape=None if right is None else list(right.shape),
            actual_dtype=None if left is None else str(left.dtype),
            expected_dtype=None if right is None else str(right.dtype),
            missing=left is None or right is None,
            nonfinite=False,
            max_abs=None,
            relative_l2=None,
        )
        if row["missing"]:
            report["missing_both" if left is None and right is None else "missing_mismatch"] += 1
        else:
            report["tensor_count"] += 1
            report["dtype_mismatch"] += int(left.dtype != right.dtype)
            if left.shape != right.shape:
                report["shape_mismatch"] += 1
            elif not torch.isfinite(left).all() or not torch.isfinite(right).all():
                row["nonfinite"] = True
                report["nonfinite_tensors"] += 1
            else:
                delta = left.detach().float() - right.detach().float()
                if not torch.isfinite(delta).all():
                    row["nonfinite"] = True
                    report["nonfinite_tensors"] += 1
                    rows.append(row)
                    continue
                d2 = float(torch.linalg.vector_norm(delta, dtype=torch.float64)) ** 2
                r2 = float(torch.linalg.vector_norm(right.detach(), dtype=torch.float64)) ** 2
                row["max_abs"] = float(delta.abs().max())
                row["relative_l2"] = math.sqrt(d2 / r2) if r2 > 0 else (0.0 if d2 == 0 else None)
                delta_squared += d2
                reference_squared += r2
                report["max_abs"] = max(report["max_abs"], row["max_abs"])
        rows.append(row)
    report["relative_l2"] = (
        math.sqrt(delta_squared / reference_squared)
        if reference_squared > 0
        else (0.0 if delta_squared == 0 else None)
    )
    report["relative_l2_reference"] = "expected; null when zero denominator and nonzero difference"
    report["worst_tensors"] = sorted(
        rows,
        key=lambda row: (
            row["nonfinite"] or row["actual_shape"] != row["expected_shape"],
            row["max_abs"] if row["max_abs"] is not None else -1.0,
        ),
        reverse=True,
    )[:5]
    return report


def _audit_replay_summary(
    actual: dict[str, Any], expected: dict[str, Any], response_names: Sequence[str]
) -> dict[str, Any]:
    return {
        **{
            name: _audit_tensor_summary((actual[name],), (expected[name],), (name,))
            for name in ("score", "loss", "input")
        },
        "response": _audit_tensor_summary(actual["response"], expected["response"], response_names),
    }


def compare_head_audit_gradients(
    actual: Sequence[Tensor | None], expected: Sequence[Tensor | None], *, atol: float
) -> dict[str, Any]:
    """Fixed absolute comparison with explicit missing/shape/nonfinite accounting."""
    report = dict(
        tensor_count=0,
        missing_both=0,
        missing_mismatch=0,
        shape_mismatch=0,
        nonfinite_tensors=0,
        max_abs=0.0,
        atol=atol,
        rtol=0.0,
    )
    if len(actual) != len(expected):
        raise ValueError("gradient audit parameter counts differ")
    for left, right in zip(actual, expected, strict=True):
        if left is None or right is None:
            report["missing_both" if left is None and right is None else "missing_mismatch"] += 1
            continue
        report["tensor_count"] += 1
        if left.shape != right.shape:
            report["shape_mismatch"] += 1
            continue
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            report["nonfinite_tensors"] += 1
            continue
        report["max_abs"] = max(
            report["max_abs"], float((left.detach().float() - right.detach().float()).abs().max())
        )
    if (
        report["missing_mismatch"]
        or report["shape_mismatch"]
        or report["nonfinite_tensors"]
        or report["max_abs"] > atol
    ):
        raise ValueError(f"population gradient audit differs: {json.dumps(report, sort_keys=True)}")
    return report


def _calibration_failure_payload(error: Exception, run_id: str) -> dict[str, Any]:
    result = dict(
        run_id=run_id,
        calibration_complete=False,
        scientific_complete=False,
        error_type=type(error).__name__,
        error=str(error),
        optimizer_updates=0,
    )
    if hasattr(error, "population_head_route_diagnostic"):
        result["population_head_route_diagnostic"] = error.population_head_route_diagnostic
    if hasattr(error, "population_head_route_deterministic_audit"):
        result["population_head_route_deterministic_audit"] = (
            error.population_head_route_deterministic_audit
        )
    return result


@contextmanager
def head_route_deterministic_context(device: Any) -> Iterator[dict[str, Any]]:
    """Strict CUDA audit only; restore both process settings even on kernel failure."""

    def state() -> dict[str, bool]:
        return dict(
            enabled=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )

    before = state()
    cuda = device.type == "cuda"
    metadata: dict[str, Any] = dict(
        scope="route_audit_only_not_norms_or_fit",
        cuda=cuda,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        before=before,
    )
    try:
        if cuda:
            if metadata["cublas_workspace_config"] not in (":4096:8", ":16:8"):
                raise ValueError("CUDA route audit requires legal CUBLAS_WORKSPACE_CONFIG")
            torch.use_deterministic_algorithms(True, warn_only=False)
        metadata["inside"] = state()
        yield metadata
    except BaseException as error:
        metadata["failure"] = dict(
            error_type=type(error).__name__,
            error=str(error),
            original_traceback=traceback.format_exc(),
            first_reported_operation=str(error),
        )
        error.__dict__["population_head_route_deterministic_audit"] = metadata
        raise
    finally:
        if cuda:
            torch.use_deterministic_algorithms(before["enabled"], warn_only=before["warn_only"])
        metadata["restored"] = state()
        metadata["settings_restored"] = metadata["restored"] == before


def audit_population_head_route(runtime: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    with head_route_deterministic_context(runtime.device) as metadata:
        result = _audit_population_head_route(runtime, *args, **kwargs)
    result["deterministic_audit"] = metadata
    return result


def _audit_population_head_route(
    runtime: Any,
    controller: Any,
    batch: Any,
    condition: int,
    *,
    bf16: bool,
    world: int,
    global_rows: int,
) -> dict[str, Any]:
    """Replay one already selected group; never contribute to calibration norms.

    Each replay starts at identical model/RNG state. Only scalar observations
    leave this function; transient scores and gradients are compared in memory.
    """
    if (
        runtime.options.response_anchor_loss != "population_prediction"
        or runtime.options.response_anchor_prediction_head_grad is not False
    ):
        raise ValueError("head-route audit requires protected population prediction")
    model = runtime.objective.student
    head = tuple(model.prediction.parameters())
    response = tuple(model.response.parameters())
    if not head or not response or set(map(id, head)) & set(map(id, response)):
        raise ValueError("disjoint native prediction head and Response parameters required")
    flags = tuple(p.requires_grad for p in runtime.objective.parameters())

    def nonzero_norm(grads: Sequence[Tensor | None], scope: str) -> float:
        values = [g.detach() for g in grads if g is not None]
        if any(not torch.isfinite(g).all() for g in values):
            raise ValueError(f"nonfinite {scope} audit gradient")
        norm = math.sqrt(sum(float(g.double().square().sum()) for g in values))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"zero or nonfinite {scope} audit gradient")
        return norm

    def replay(protected: bool) -> dict[str, Any]:
        original = runtime.options
        options = (
            replace(original, response_anchor_prediction_head_grad=not protected)
            if is_dataclass(original)
            else copy.copy(original)
        )
        if not is_dataclass(original):
            options.response_anchor_prediction_head_grad = not protected
        captured: dict[str, Tensor] = {}

        def pre_hook(module: Any, args: tuple[Tensor, ...]) -> None:
            captured["input"] = args[0]

        def post_hook(module: Any, args: Any, output: Tensor) -> None:
            captured["score"] = output

        try:
            runtime.options = options
            with preserved_calibration(runtime), controller.group_rng(condition, 0):
                before = model.prediction.register_forward_pre_hook(pre_hook)
                after = model.prediction.register_forward_hook(post_hook)
                try:
                    loss = controller.group_loss(batch, condition, 0, bf16=bf16)
                    if set(captured) != {"input", "score"}:
                        raise ValueError("native population head was not observed")
                    grads = torch.autograd.grad(
                        loss, (captured["input"], *response, *head), allow_unused=True
                    )
                finally:
                    before.remove()
                    after.remove()
                return dict(
                    loss=loss.detach().clone(),
                    score=captured["score"].detach().clone(),
                    input=grads[0],
                    response=grads[1 : 1 + len(response)],
                    head=grads[1 + len(response) :],
                )
        finally:
            runtime.options = original

    protected = replay(True)
    reference = replay(False)
    try:
        for name in ("score", "loss"):
            if not torch.equal(protected[name], reference[name]):
                raise ValueError(f"protected native head {name} differs from default route")
        comparisons = {}
        for scope in ("input", "response"):
            left = (protected[scope],) if scope == "input" else protected[scope]
            right = (reference[scope],) if scope == "input" else reference[scope]
            nonzero_norm(left, scope)
            atol = _CUDA_RESPONSE_GRADIENT_ATOL if runtime.device.type == "cuda" else 0.0
            comparisons[scope] = compare_head_audit_gradients(left, right, atol=atol)
    except ValueError as original_error:
        diagnostic: dict[str, Any] = dict(
            schema="population-head-route-failure-diagnostic-v1",
            diagnostic_only=True,
            original_failure=str(original_error),
            condition_index=condition,
            epoch=0,
            gradient_routing=_PROTECTED_ROUTE,
            acceptance_gate_unchanged=True,
            additional_replay_limit_per_route=1,
            additional_replays_started=0,
            route_labels="True/False denote prediction_head_grad",
            comparisons={},
        )
        try:
            names = tuple(
                "student.response." + name for name, _ in model.response.named_parameters()
            )
            diagnostic["comparisons"]["true_false"] = _audit_replay_summary(
                reference, protected, names
            )
            diagnostic["additional_replays_started"] += 1
            repeated = replay(True)
            diagnostic["comparisons"]["false_false"] = _audit_replay_summary(
                protected, repeated, names
            )
            del repeated
            diagnostic["additional_replays_started"] += 1
            repeated = replay(False)
            diagnostic["comparisons"]["true_true"] = _audit_replay_summary(
                reference, repeated, names
            )
            del repeated
            diagnostic["repeatability_diagnostic_complete"] = True
        except BaseException as diagnostic_error:
            diagnostic["repeatability_diagnostic_complete"] = False
            diagnostic["diagnostic_error_type"] = type(diagnostic_error).__name__
            diagnostic["diagnostic_error"] = str(diagnostic_error)
        original_error.population_head_route_diagnostic = diagnostic
        original_error.add_note(
            "population_head_route_diagnostic=" + json.dumps(diagnostic, sort_keys=True)
        )
        raise
    repeatability: dict[str, Any] = {}
    replay_cases = (
        (("false_false", protected, True), ("true_true", reference, False))
        if runtime.device.type == "cuda"
        else ()
    )
    for label, baseline, route in replay_cases:
        repeated = replay(route)
        if any(not torch.equal(baseline[name], repeated[name]) for name in ("score", "loss")):
            raise ValueError(f"{label} native head score/loss replay differs")
        repeatability[label] = {}
        for scope in ("input", "response"):
            left = (baseline[scope],) if scope == "input" else baseline[scope]
            right = (repeated[scope],) if scope == "input" else repeated[scope]
            repeatability[label][scope] = compare_head_audit_gradients(
                left,
                right,
                atol=_CUDA_RESPONSE_GRADIENT_ATOL if runtime.device.type == "cuda" else 0.0,
            )
        del repeated
    if any(
        g is not None and (not torch.isfinite(g).all() or torch.count_nonzero(g))
        for g in protected["head"]
    ):
        raise ValueError("protected population head received an auxiliary gradient")
    reference_head_norm = nonzero_norm(reference["head"], "reference head")

    with preserved_calibration(runtime):
        for parameter in head:
            parameter.grad = None
        for loss in prediction_losses(
            runtime, batch, bf16=bf16, world=world, global_rows=global_rows
        ):
            loss.backward(inputs=head)
        primary = tuple(None if p.grad is None else p.grad.clone() for p in head)
        primary_head_norm = nonzero_norm(primary, "primary head")
        with controller.group_rng(condition, 0):
            controller.group_loss(batch, condition, 0, bf16=bf16).backward()
        if any(
            (p.grad is None) != (g is None) or (g is not None and not torch.equal(p.grad, g))
            for p, g in zip(head, primary, strict=True)
        ):
            raise ValueError("population auxiliary changed accumulated primary head gradients")
    if flags != tuple(p.requires_grad for p in runtime.objective.parameters()):
        raise ValueError("population audit changed parameter requires_grad flags")
    return dict(
        schema="population-head-route-audit-v1",
        complete=True,
        gradient_routing=_PROTECTED_ROUTE,
        condition_index=condition,
        epoch=0,
        group_replays=3 + len(replay_cases),
        additional_audit_replays=len(replay_cases),
        replay_norms_included_in_first_eight=False,
        repeatability_comparisons=repeatability,
        head_score_and_loss_equal=True,
        input_gradient_equal=True,
        response_gradient_equal=True,
        gradient_comparisons=comparisons,
        score_atol=0.0,
        score_rtol=0.0,
        loss_atol=0.0,
        loss_rtol=0.0,
        cuda_response_gradient_atol=_CUDA_RESPONSE_GRADIENT_ATOL,
        cuda_response_gradient_rtol=0.0,
        cuda_input_gradient_atol=_CUDA_RESPONSE_GRADIENT_ATOL,
        cuda_input_gradient_rtol=0.0,
        auxiliary_head_gradient_zero=True,
        primary_head_gradient_preserved=True,
        state_restored=True,
        input_gradient_norm=nonzero_norm((protected["input"],), "input"),
        response_gradient_norm=nonzero_norm(protected["response"], "Response"),
        reference_head_gradient_norm=reference_head_norm,
        primary_head_gradient_norm=primary_head_norm,
        optimizer_updates=0,
        teacher_updates=0,
        center_updates=0,
    )


def calibrate_gradients(
    parameters: Sequence[Tensor],
    prediction_losses: Callable[[int], Iterable[Tensor]],
    anchor_losses: Callable[[int], Iterable[Tensor]],
    metadata: Sequence[dict[str, Any]],
    *,
    P: int,
    steps_per_epoch: int,
    world: int = 1,
    reduce_mean: Callable[[Tensor], Tensor] | None = None,
) -> dict[str, Any]:
    """Accumulate each complete group before norm, then take eight norm medians.

    Prediction callbacks supply already row/global-weighted microbatch losses.
    Anchor callbacks supply unweighted complete-condition losses; this function
    applies world*T/P once. reduce_mean averages parameter gradients across ranks,
    never rank norms. autograd.grad leaves existing .grad untouched.
    """
    from gradpert.training.v2.response_anchor import group_loss_coefficient

    params = tuple(parameters)
    if (
        not params
        or len({id(p) for p in params}) != len(params)
        or any(not p.requires_grad or not p.is_floating_point() for p in params)
        or len(metadata) != _BATCHES
        or [row.get("step") for row in metadata] != list(range(_BATCHES))
        or len({row.get("ordered_identity_sha256") for row in metadata}) != _BATCHES
        or any(
            not isinstance(row.get("ordered_identity_sha256"), str)
            or len(row["ordered_identity_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in row["ordered_identity_sha256"])
            for row in metadata
        )
    ):
        raise ValueError("eight ordered identities and unique trainable shared parameters required")
    if world > 1 and reduce_mean is None:
        raise ValueError("distributed calibration requires gradient rank-mean reduction")
    factor = group_loss_coefficient(P, steps_per_epoch, world)

    def norm(losses: Iterable[Tensor], scale: float, *, require_loss: bool) -> float:
        bank = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        seen = 0
        for loss in losses:
            if loss.ndim or not loss.requires_grad or not torch.isfinite(loss):
                raise ValueError("calibration requires finite differentiable scalar losses")
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            for total, grad in zip(bank, grads, strict=True):
                if grad is not None:
                    if not torch.isfinite(grad).all():
                        raise ValueError("nonfinite shared response gradient")
                    total.add_(grad.detach().float(), alpha=scale)
            seen += 1
        if require_loss and not seen:
            raise ValueError("prediction has no local rows")
        if reduce_mean is not None:
            bank = [reduce_mean(value) for value in bank]
        if any(
            value.shape != p.shape
            or value.dtype != torch.float32
            or value.device != p.device
            or not torch.isfinite(value).all()
            for value, p in zip(bank, params, strict=True)
        ):
            raise ValueError("invalid reduced gradient")
        value = math.sqrt(sum(float(g.double().square().sum()) for g in bank))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "zero or nonfinite shared gradient; fixed batch invalid, no resampling"
            )
        return value

    rows = []
    for step in range(_BATCHES):
        pred = norm(prediction_losses(step), 1.0, require_loss=True)
        anchor = norm(anchor_losses(step), factor, require_loss=False)
        rows.append(
            {
                **metadata[step],
                "prediction_gradient_norm": pred,
                "unit_anchor_gradient_norm": anchor,
            }
        )
    gp = statistics.median(row["prediction_gradient_norm"] for row in rows)
    ga = statistics.median(row["unit_anchor_gradient_norm"] for row in rows)
    weight = min(1.0, 0.1 * gp / ga)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("invalid calibration weight")
    return dict(
        schema="response-anchor-gradient-calibration-v1",
        batches=rows,
        median_prediction_gradient_norm=gp,
        median_unit_anchor_gradient_norm=ga,
        selected_weight=weight,
        anchor_coefficient=factor,
        parameter_scope="student.response.parameters; excludes response_anchor_readout",
        gradient_bank_dtype="float32",
        norm_accumulation_dtype="float64",
        rule="min(1,0.1*median(Gpred)/median(Ganchor))",
        epoch=0,
        optimizer_updates=0,
        teacher_updates=0,
        center_updates=0,
        calibration_complete=True,
        scientific_complete=False,
    )


@contextmanager
def preserved_calibration(runtime: Any) -> Iterator[None]:
    """Restore ephemeral state and verify parameter/optimizer byte invariance.

    No second model or optimizer is allocated. Parameter changes are forbidden
    and detected, rather than hidden by resetting updated weights afterwards.
    """
    from scripts.v2.capacity_probe import state_fingerprint, validation_state

    objective = runtime.objective
    before = validation_state(runtime)
    rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if runtime.device.type == "cuda" else None,
    )
    generator = copy.deepcopy(runtime.generator.bit_generator.state)
    pending = copy.deepcopy(objective.pending)
    modes = [
        (
            m,
            m.training,
            {
                key: getattr(m, key)
                for key in ("randomize_relay_order", "randomize_order")
                if hasattr(m, key)
            },
        )
        for m in objective.modules()
    ]
    buffers = [(buffer, buffer.detach().clone()) for buffer in objective.buffers()]
    grads = [
        (p, None if p.grad is None else p.grad.detach().clone()) for p in objective.parameters()
    ]
    flags = {
        name: getattr(runtime.data, name)
        for name in ("control_only_training",)
        if hasattr(runtime.data, name)
    }
    optimizer_hash = state_fingerprint(runtime.optimizer.state_dict())
    try:
        objective.train()
        yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for parameter, value in grads:
            parameter.grad = value
        objective.pending.clear()
        objective.pending.update(pending)
        runtime.generator.bit_generator.state = generator
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        for module, mode, orders in modes:
            module.training = mode
            for key, value in orders.items():
                setattr(module, key, value)
        for name in ("control_only_training",):
            if name in flags:
                setattr(runtime.data, name, flags[name])
            elif hasattr(runtime.data, name):
                delattr(runtime.data, name)
        if optimizer_hash != state_fingerprint(runtime.optimizer.state_dict()):
            raise RuntimeError("calibration changed optimizer state")
        if validation_state(runtime) != before:
            raise RuntimeError("calibration changed protected objective/RNG/gradient state")


def prediction_losses(
    runtime: Any, batch: Any, *, bf16: bool, world: int, global_rows: int
) -> Iterator[Tensor]:
    """Native Student main prediction only; exact row-mean/global-row weighting."""
    from gradpert.training.v2.engine import slice_cells

    objective, model = runtime.objective, runtime.objective.student
    if (
        objective.loss_reduction != "row_mean"
        or objective.prediction_error_power != 2
        or objective.prediction_reduction_override != "inherit"
        or not objective.prediction_enabled
        or objective.population_response
    ):
        raise ValueError("anchor calibration requires original RI row_mean prediction MSE")
    for start in range(0, len(batch.control), runtime.options.microbatch):
        micro = slice_cells(
            batch, start, min(start + runtime.options.microbatch, len(batch.control))
        )
        with torch.autocast(runtime.device.type, dtype=torch.bfloat16, enabled=bf16):
            prefix = (
                model.graph.relay_prefix(micro.graph.context)
                if model.options.control_conditioned_graph and micro.graph.context is not None
                else None
            )
            graph, condition = objective._batch_graph(model, micro, prefix=prefix)
            response = model.encode_response(
                objective._select_graph(graph, micro.query_positions),
                micro.control,
                condition,
                **objective.response_metadata(micro),
            )
            loss = (response["prediction"].float() - micro.truth.float()).square().mean()
        yield loss * (world * len(micro.control) / global_rows)


def calibrate_first_eight(runtime: Any, controller: Any, *, bf16: bool = False) -> dict[str, Any]:
    """Use the exact first eight native epoch0 batches at initialization, no filtering."""
    from gradpert.hashing import sha256_json
    from gradpert.training.v2.engine import slice_cells

    if runtime.purpose != "training" or runtime.optimizer.steps != 0:
        raise ValueError("calibration requires fresh training runtime at optimizer step zero")
    distributed = torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0
    protected_head = getattr(runtime.options, "response_anchor_prediction_head_grad", True) is False
    head_audit = None
    params = tuple(runtime.objective.student.response.parameters())
    readout = getattr(runtime.objective.student, "response_anchor_readout", None)
    if readout is not None and any(id(p) in {id(q) for q in readout.parameters()} for p in params):
        raise ValueError("readout overlaps shared response parameters")

    def reduce_mean(value: Tensor) -> Tensor:
        if distributed:
            torch.distributed.all_reduce(value)
            value.div_(world)
        return value

    with preserved_calibration(runtime):
        specs = runtime.data.training_batch_identity_specs(
            epoch=0,
            batch_size=runtime.batch_size,
            max_unique_conditions=(
                0
                if runtime.options.max_conditions == 0
                else min(runtime.options.max_conditions, runtime.batch_size)
            ),
        )[:_BATCHES]
        metadata = [
            dict(
                step=i,
                ordered_identity_sha256=sha256_json(asdict(spec)),
                global_rows=len(spec.control_row_ids),
            )
            for i, spec in enumerate(specs)
        ]
        current: dict[str, Any] = {}
        stream = iter(runtime.batches(0))
        try:

            def pred(step: int) -> Iterable[Tensor]:
                try:
                    batch = next(stream)
                except StopIteration as error:
                    raise ValueError("epoch0 has fewer than eight batches") from error
                rows = len(batch.control)
                if rows != metadata[step]["global_rows"] or rows < world:
                    raise ValueError("native batch differs from sealed ordered metadata")
                metadata[step]["query_ids_sha256"] = sha256_json(
                    batch.graph.ids[batch.query_positions].detach().cpu().tolist()
                )
                current["batch"] = slice_cells(
                    batch, rows * rank // world, rows * (rank + 1) // world
                )
                return prediction_losses(
                    runtime, current["batch"], bf16=bf16, world=world, global_rows=rows
                )

            def anchor(step: int) -> Iterable[Tensor]:
                nonlocal head_audit
                groups = controller.groups(step)
                metadata[step]["local_group_indices_sha256"] = sha256_json(list(groups))
                metadata[step]["local_group_count"] = len(groups)
                for condition in groups:
                    if protected_head:
                        if head_audit is None:
                            head_audit = audit_population_head_route(
                                runtime,
                                controller,
                                current["batch"],
                                condition,
                                bf16=bf16,
                                world=world,
                                global_rows=metadata[step]["global_rows"],
                            )
                            head_audit["calibration_step"] = step
                        # Match the native auxiliary backward's private RNG scope,
                        # including activation-checkpoint recomputation.
                        with controller.group_rng(condition, 0):
                            yield controller.group_loss(current["batch"], condition, 0, bf16=bf16)
                    else:
                        yield controller.group_loss(current["batch"], condition, 0, bf16=bf16)

            result = calibrate_gradients(
                params,
                pred,
                anchor,
                metadata,
                P=len(controller.bank.condition_ids),
                steps_per_epoch=runtime.steps_per_epoch,
                world=world,
                reduce_mean=reduce_mean,
            )
        finally:
            if hasattr(stream, "close"):
                stream.close()
    result["state_restored"] = True
    result["world_size"] = world
    result["rank"] = rank
    result["forward_precision"] = "bfloat16_autocast" if bf16 else "float32"
    result["ordered_batch_metadata_sha256"] = sha256_json(metadata)
    if protected_head:
        if head_audit is None:
            raise ValueError("protected head calibration did not audit a selected complete group")
        result["response_anchor_prediction_head_grad"] = False
        result["gradient_routing"] = _PROTECTED_ROUTE
        result["population_head_route_audit"] = head_audit
    if getattr(runtime.options, "response_anchor_loss", "mse") == "population_prediction":
        result["response_anchor_loss"] = "population_prediction"
        result["population_manifest_sha256"] = runtime.options.response_anchor_population_sha256
        result["population_arrays_sha256"] = (
            runtime.options.response_anchor_population_arrays_sha256
        )
    if getattr(runtime.options, "response_anchor_loss", "mse") == "cosine_direction":
        result["response_anchor_loss"] = "cosine_direction"
        result["direction_diagnostics"] = controller.direction_summary()
        result["direction_diagnostics_scope"] = "this_rank_controller_lifetime_not_rank_sum"
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Formal wrapper; never called by the local synthetic test suite."""
    from contextlib import ExitStack

    root = Path(__file__).resolve().parents[2]
    for path in (args.config, args.data_root, args.output, args.publication, args.shared_policy):
        if not path.resolve().is_relative_to("/data/yilangliu"):
            raise ValueError("formal calibration paths must be on /data/yilangliu")
    if args.output.resolve().is_relative_to(args.data_root.resolve()) or (
        args.output.resolve().is_relative_to(root)
    ):
        raise ValueError("calibration output must be separate from data and source trees")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator contract missing")
    if not args.bf16:
        raise ValueError("formal anchor calibration must match native BF16 training")

    from gradpert.config import load_experiment_config
    from gradpert.data._io import atomic_json
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.hashing import sha256_file, sha256_json
    from gradpert.training.v2.runtime import prepare_runtime
    from scripts.v2.run_group import lock
    from scripts.v2.shared_gpu import admit, descendant, parent_map, snapshot

    before_files = {
        str(path): sha256_file(path) for path in (args.config, args.publication, args.shared_policy)
    }
    config = load_experiment_config(args.config)
    if config.continuation is not None:
        raise ValueError("calibration must start from original initialization")
    source = inspect_source_identity(
        root,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    if (
        source.commit != args.source_commit
        or sha256_file(args.shared_policy) != args.shared_policy_sha256
    ):
        raise ValueError("source or resource policy identity mismatch")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    devices = args.gpu.split(",")
    if len(devices) != world or len(set(devices)) != world or not set(devices) <= {"0", "1"}:
        raise ValueError("GPU list must exactly match torchrun world")
    if not 0 <= rank < world:
        raise ValueError("invalid local rank")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
    if not torch.cuda.is_available():
        raise ValueError("formal calibration requires CUDA")
    owner = args.resource_owner_pid or os.getpid()
    if owner != os.getpid() and not descendant(os.getpid(), owner, parent_map()):
        raise ValueError("resource owner must be self or ancestor")
    policy_doc = json.loads(args.shared_policy.read_text())
    policy = policy_doc.get("shared_gpu_policy", policy_doc)
    owned_output = False
    created_group = False
    try:
        if world > 1:
            torch.distributed.init_process_group("nccl")
            created_group = True
        with ExitStack() as stack:
            stack.enter_context(
                lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{devices[rank]}.lock")
            )
            admission = snapshot(owner)
            admit(policy, admission)
            torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], 0)
            torch.set_num_threads(1)
            if rank == 0:
                args.output.mkdir(parents=True, exist_ok=False)
                owned_output = True
            if world > 1:
                torch.distributed.barrier()
            with prepare_runtime(
                config,
                data_root=args.data_root,
                run_seed=args.seed,
                device=torch.device("cuda:0"),
                purpose="training",
            ) as runtime:
                callback = getattr(runtime.objective, "response_anchor_backward", None)
                controller = getattr(callback, "__self__", None)
                if controller is None:
                    raise ValueError("enabled training response anchor controller required")
                bank = controller.bank
                before_files.update(
                    {
                        str(bank.manifest_path): bank.manifest_sha256,
                        str(bank.arrays_path): bank.arrays_sha256,
                    }
                )
                if runtime.options.response_anchor_loss == "population_prediction":
                    population_path = Path(runtime.options.response_anchor_population_path)
                    population_doc = json.loads(population_path.read_text())
                    before_files[str(population_path)] = (
                        runtime.options.response_anchor_population_sha256
                    )
                    before_files[str(population_path.parent / population_doc["arrays"]["path"])] = (
                        runtime.options.response_anchor_population_arrays_sha256
                    )
                result = calibrate_first_eight(runtime, controller, bf16=args.bf16)
                result.update(
                    run_id=args.run_id,
                    source=source.payload(),
                    identity=runtime.identity,
                    config_sha256=before_files[str(args.config)],
                    resolved_config_sha256=sha256_json(config.model_dump(mode="json")),
                    bank_manifest_sha256=bank.manifest_sha256,
                    bank_arrays_sha256=bank.arrays_sha256,
                    admission=admission,
                    resource_policy_sha256=args.shared_policy_sha256,
                    environment=inspect_environment(root, device_name="cuda:0").payload(),
                )
            # Native data handle is closed before terminal acceptance.
            if any(sha256_file(Path(path)) != digest for path, digest in before_files.items()):
                raise ValueError("calibration input bytes changed")
            after = inspect_source_identity(
                root,
                formal=True,
                expected_repository=config.source_code.repository,
                publication_receipt=args.publication,
                expected_publication_receipt_sha256=args.publication_sha256,
            )
            if after != source:
                raise ValueError("calibration source changed")
            atomic_json(args.output / f"RANK_{rank}.json", result)
            if world > 1:
                torch.distributed.barrier()
            if rank == 0:
                ranks = [
                    json.loads((args.output / f"RANK_{i}.json").read_text()) for i in range(world)
                ]
                if len({row["selected_weight"] for row in ranks}) != 1:
                    raise ValueError("rank calibration weights disagree")
                if getattr(runtime.options, "response_anchor_prediction_head_grad", True) is False:
                    if any(
                        row.get("response_anchor_prediction_head_grad") is not False
                        or row.get("gradient_routing") != _PROTECTED_ROUTE
                        or row.get("population_head_route_audit", {}).get("complete") is not True
                        for row in ranks
                    ):
                        raise ValueError("protected population head audit missing on a rank")
                    result["population_head_route_audit_ranks"] = [
                        row["population_head_route_audit"] for row in ranks
                    ]
                if list(args.output.rglob("*.pkl")):
                    raise ValueError("calibration must remain zero-PKL")
                result["input_files_sha256"] = before_files
                atomic_json(args.output / "COMPLETE.json", result)
            return result
    except Exception as error:
        if owned_output:
            atomic_json(
                args.output / "FAILURE.json",
                _calibration_failure_payload(error, args.run_id),
            )
        raise
    finally:
        if created_group:
            torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-root", "output", "publication", "shared-policy"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("publication-sha256", "shared-policy-sha256", "source-commit", "run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resource-owner-pid", type=int)
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
