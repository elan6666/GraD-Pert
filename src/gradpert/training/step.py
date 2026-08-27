"""Exact B2 forward, loss, optimizer, Teacher EMA, and center update order."""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from gradpert.config.native import NativeArchitectureOptions
from gradpert.graphs import (
    GraDPertTrainingViews,
    GraphTopology,
    build_incoming_neighbor_index,
    build_prediction_graph_view,
    build_training_graph_views,
    clean_graph_view,
)
from gradpert.hashing import sha256_json
from gradpert.modeling import (
    CenterState,
    EncodedGraphView,
    GraDPertJointModel,
    centered_teacher_probabilities,
    condition_consistency_loss,
    cosine_teacher_momentum,
    embedding_spread_loss,
    masked_node_consistency_loss,
    update_center,
    update_teacher_ema,
)
from gradpert.training.batch import GraDPertTrainingBatch


@dataclass(frozen=True)
class GraDPertStepMetrics:
    total_loss: float
    prediction_loss: float
    condition_consistency_loss: float
    masked_node_loss: float
    spread_loss: float
    spread_available: bool
    teacher_momentum: float
    prediction_graph_gradient_norm: float
    auxiliary_graph_gradient_norm: float
    prediction_to_auxiliary_gradient_ratio: float | None
    condition_target_entropy: float
    masked_node_target_entropy: float | None
    condition_prototypes_used: int
    masked_node_prototypes_used: int
    condition_center_norm: float
    masked_node_center_norm: float
    unique_condition_count: int
    masked_node_count: int
    batch_cell_count: int
    data_read_ms: float
    host_to_device_ms: float
    view_build_ms: float
    teacher_forward_ms: float
    student_global_ms: float
    student_local_ms: float
    prediction_ms: float
    backward_update_ms: float
    step_wall_ms: float


@dataclass(frozen=True)
class LossWeights:
    """Direct objective weights; defaults preserve historical pilot behavior."""

    prediction: float = 1.0
    condition_consistency: float = 0.1
    masked_node: float = 0.1
    spread: float = 0.01

    def __post_init__(self) -> None:
        values = (
            self.prediction,
            self.condition_consistency,
            self.masked_node,
            self.spread,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("loss weights must be finite and non-negative")
        if self.prediction == 0:
            raise ValueError("prediction loss weight must be positive")


def build_native_optimizer(
    model: GraDPertJointModel,
    *,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
) -> torch.optim.AdamW:
    if learning_rate != 0.001 or weight_decay != 0.0:
        raise ValueError("v1 AdamW learning rate/weight decay are frozen to 1e-3/0")
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _stack_condition_states(
    encoded: EncodedGraphView,
    views: GraDPertTrainingViews,
) -> Tensor:
    return torch.stack(
        [
            encoded.condition_state(views.anchors_by_condition[condition_id])
            for condition_id in views.anchors_by_condition
        ]
    )


def _gradient_norm(gradients: tuple[Tensor | None, ...], reference: Tensor) -> float:
    squared = reference.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return float(torch.sqrt(squared).item())


def _distribution_health(probabilities: Tensor) -> tuple[float, int]:
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()
    prototypes_used = int(torch.unique(probabilities.argmax(dim=-1)).numel())
    return float(entropy.item()), prototypes_used


def _model_state_sha256(model: GraDPertJointModel) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(sha256_json(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _rng_state_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(sha256_json(random.getstate()).encode("ascii"))
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    digest.update(
        sha256_json(
            [
                algorithm,
                cast(np.ndarray[Any, Any], keys).tolist(),
                position,
                has_gauss,
                cached_gaussian,
            ]
        ).encode("ascii")
    )
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def _view_structure_sha256(views: GraDPertTrainingViews) -> str:
    def view_payload(view: Any) -> dict[str, object]:
        return {
            "view_id": view.view_id,
            "node_ids_sha256": sha256_json(list(view.node_ids)),
            "edges_by_source_sha256": {
                source: sha256_json([[edge.source, edge.target, edge.weight] for edge in edges])
                for source, edges in sorted(view.edges_by_source.items())
            },
            "masked_node_ids": list(view.masked_node_ids),
            "masked_anchor_ids": list(view.masked_anchor_ids),
        }

    return sha256_json(
        {
            "prediction": view_payload(views.prediction),
            "globals": [view_payload(view) for view in views.globals],
            "locals_by_condition": {
                condition: [view_payload(view) for view in local_views]
                for condition, local_views in views.locals_by_condition.items()
            },
            "masked_global_index": views.masked_global_index,
            "masked_local_indices_by_condition": {
                condition: list(indices)
                for condition, indices in views.masked_local_indices_by_condition.items()
            },
        }
    )


class GraDPertStepEngine:
    """Own one full native B2 train step without hiding validation or test access."""

    def __init__(
        self,
        *,
        model: GraDPertJointModel,
        topology: GraphTopology,
        optimizer: torch.optim.Optimizer,
        centers: CenterState,
        run_seed: int,
        total_schedule_steps: int,
        heldout_target_ids: tuple[int, ...],
        architecture: NativeArchitectureOptions | None = None,
        loss_weights: LossWeights | None = None,
        resident_graph_tensors: bool = False,
        capture_equivalence_health: bool = False,
    ) -> None:
        if topology.n_nodes != model.graph_gene_count:
            raise ValueError("topology and model graph-gene counts differ")
        if total_schedule_steps <= 0:
            raise ValueError("total_schedule_steps must be positive")
        if centers.condition.shape != centers.masked_node.shape:
            raise ValueError("condition and masked-node centers must share prototype width")
        if centers.condition.shape[1] != model.student_projector.prototype_count:
            raise ValueError("center and projector prototype widths differ")
        self.model = model
        self.topology = topology
        self.optimizer = optimizer
        self.centers = centers
        self.run_seed = run_seed
        self.total_schedule_steps = total_schedule_steps
        self.heldout_target_ids = heldout_target_ids
        self.architecture = architecture or NativeArchitectureOptions.from_parameters({})
        self.loss_weights = loss_weights or LossWeights()
        self.prediction_view = build_prediction_graph_view(topology)
        self.incoming_neighbors = (
            build_incoming_neighbor_index(topology) if resident_graph_tensors else None
        )
        self.resident_graph_tensors = resident_graph_tensors
        self.capture_equivalence_health = capture_equivalence_health
        self.first_step_health: dict[str, object] | None = None
        self.model.student_encoder.configure_string_weight_contract(self.prediction_view)
        self.model.teacher_encoder.configure_string_weight_contract(self.prediction_view)
        if resident_graph_tensors:
            self.model.student_encoder.configure_resident_graph_tensors(self.prediction_view)
            self.model.teacher_encoder.configure_resident_graph_tensors(self.prediction_view)

    def train_step(
        self,
        batch: GraDPertTrainingBatch,
        *,
        global_step: int,
    ) -> GraDPertStepMetrics:
        if not 0 <= global_step < self.total_schedule_steps:
            raise ValueError("global_step is outside the frozen maximum schedule")
        if batch.control_expression.shape[1] != self.model.expression_gene_count:
            raise ValueError("batch and model expression-gene counts differ")
        capture_health = self.capture_equivalence_health and global_step == 0
        parameter_state_before_sha256 = _model_state_sha256(self.model) if capture_health else None
        rng_state_before_sha256 = _rng_state_sha256() if capture_health else None
        update_order: list[str] = []
        step_started = time.perf_counter()
        self.model.train()
        view_started = time.perf_counter()
        views = build_training_graph_views(
            self.topology,
            anchors_by_condition=batch.anchors_by_condition,
            heldout_target_ids=self.heldout_target_ids,
            run_seed=self.run_seed,
            global_step=global_step,
            prediction_view=self.prediction_view if self.resident_graph_tensors else None,
            incoming_neighbors=self.incoming_neighbors,
            local_count=self.architecture.local_view_count,
            local_node_budget=self.architecture.local_view_node_budget,
            local_builder=self.architecture.local_view_builder,
            local_fanouts=self.architecture.local_view_fanout,
            local_anchor_mask_count=self.architecture.local_anchor_mask_count,
        )
        view_structure_sha256 = _view_structure_sha256(views) if capture_health else None
        view_build_ms = (time.perf_counter() - view_started) * 1000.0
        use_cuda_events = self.model.student_encoder.gene_embeddings.is_cuda
        events: dict[str, Any] = {}

        def mark(name: str) -> None:
            if use_cuda_events:
                event = cast(Any, torch.cuda.Event)(enable_timing=True)
                event.record()
                events[name] = event

        mark("teacher_start")
        clean_globals = tuple(clean_graph_view(view) for view in views.globals)
        with torch.no_grad():
            teacher_encoded = self.model.teacher_encoder.forward_many(clean_globals)
            teacher_condition_states = tuple(
                _stack_condition_states(encoded, views) for encoded in teacher_encoded
            )
            teacher_condition_logits = tuple(
                self.model.teacher_projector(states) for states in teacher_condition_states
            )
        mark("teacher_end")

        mark("student_global_start")
        student_encoded = self.model.student_encoder.forward_many(views.globals)
        student_global_states = tuple(
            _stack_condition_states(encoded, views) for encoded in student_encoded
        )
        student_view_logits: list[Tensor] = [
            self.model.student_projector(states) for states in student_global_states
        ]
        mark("student_global_end")
        mark("student_local_start")
        for local_index in range(self.architecture.local_view_count):
            condition_ids = tuple(views.anchors_by_condition)
            local_views = tuple(
                views.locals_by_condition[condition_id][local_index]
                for condition_id in condition_ids
            )
            local_encoded = self.model.student_encoder.forward_many(local_views)
            local_states = [
                encoded.condition_state(views.anchors_by_condition[condition_id])
                for condition_id, encoded in zip(condition_ids, local_encoded, strict=True)
            ]
            student_view_logits.append(self.model.student_projector(torch.stack(local_states)))
        mark("student_local_end")

        condition_loss = condition_consistency_loss(
            student_view_logits=tuple(student_view_logits),
            teacher_global_logits=teacher_condition_logits,
            center=self.centers.condition,
        )

        masked_global_index = views.masked_global_index
        masked_node_ids = views.globals[masked_global_index].masked_node_ids
        student_masked_states = student_encoded[masked_global_index].node_states_for(
            masked_node_ids
        )
        teacher_masked_states = teacher_encoded[masked_global_index].node_states_for(
            masked_node_ids
        )
        student_masked_logits = self.model.student_projector(student_masked_states)
        with torch.no_grad():
            teacher_masked_logits = self.model.teacher_projector(teacher_masked_states)
        masked_loss = masked_node_consistency_loss(
            student_logits=student_masked_logits,
            teacher_logits=teacher_masked_logits,
            center=self.centers.masked_node,
        )

        spread_terms = [embedding_spread_loss(states) for states in student_global_states]
        spread_available = all(available for _, available in spread_terms)
        spread_loss = torch.stack([value for value, _ in spread_terms]).mean()
        auxiliary_loss = (
            self.loss_weights.condition_consistency * condition_loss
            + self.loss_weights.masked_node * masked_loss
            + self.loss_weights.spread * spread_loss
        )

        mark("prediction_start")
        prediction = self.model.predict_expression_batch(
            batch.control_expression,
            views.prediction,
            batch.condition_ids,
            views.anchors_by_condition,
        )
        prediction_loss = F.mse_loss(prediction, batch.target_expression)
        weighted_prediction_loss = self.loss_weights.prediction * prediction_loss
        total_loss = weighted_prediction_loss + auxiliary_loss
        mark("prediction_end")

        condition_probabilities = torch.cat(
            [
                centered_teacher_probabilities(logits, self.centers.condition)
                for logits in teacher_condition_logits
            ]
        )
        condition_entropy, condition_used = _distribution_health(condition_probabilities)
        masked_entropy: float | None = None
        masked_used = 0
        if masked_node_ids:
            masked_probabilities = centered_teacher_probabilities(
                teacher_masked_logits,
                self.centers.masked_node,
            )
            masked_entropy, masked_used = _distribution_health(masked_probabilities)

        mark("backward_start")
        trainable_parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        auxiliary_gradients = torch.autograd.grad(
            auxiliary_loss,
            trainable_parameters,
            allow_unused=True,
        )
        self.optimizer.zero_grad(set_to_none=True)
        weighted_prediction_loss.backward()  # type: ignore[no-untyped-call]

        graph_parameter_ids = {
            id(parameter)
            for name, parameter in self.model.student_encoder.named_parameters()
            if name != "mask_token" and parameter.requires_grad
        }
        prediction_graph_gradients = tuple(
            parameter.grad
            for parameter in trainable_parameters
            if id(parameter) in graph_parameter_ids
        )
        auxiliary_graph_gradients = tuple(
            gradient
            for parameter, gradient in zip(trainable_parameters, auxiliary_gradients, strict=True)
            if id(parameter) in graph_parameter_ids
        )
        prediction_gradient_norm = _gradient_norm(
            prediction_graph_gradients,
            weighted_prediction_loss,
        )
        auxiliary_gradient_norm = _gradient_norm(
            auxiliary_graph_gradients,
            auxiliary_loss,
        )
        gradient_ratio = (
            prediction_gradient_norm / auxiliary_gradient_norm
            if auxiliary_gradient_norm > 0
            else None
        )

        for parameter, auxiliary_gradient in zip(
            trainable_parameters, auxiliary_gradients, strict=True
        ):
            if auxiliary_gradient is None:
                continue
            if parameter.grad is None:
                parameter.grad = auxiliary_gradient.detach()
            else:
                parameter.grad.add_(auxiliary_gradient.detach())
        self.optimizer.step()
        if capture_health:
            update_order.append("optimizer_step")
        schedule_last_step = max(1, self.total_schedule_steps - 1)
        momentum = cosine_teacher_momentum(
            global_step=global_step,
            total_steps=schedule_last_step,
        )
        update_teacher_ema(
            self.model.student_encoder,
            self.model.teacher_encoder,
            momentum=momentum,
        )
        update_teacher_ema(
            self.model.student_projector,
            self.model.teacher_projector,
            momentum=momentum,
        )
        if capture_health:
            update_order.append("teacher_ema")
        update_center(
            self.centers.condition,
            torch.cat(teacher_condition_logits),
        )
        if masked_node_ids:
            update_center(self.centers.masked_node, teacher_masked_logits)
        if capture_health:
            update_order.append("center_update")
        mark("backward_end")

        if use_cuda_events:
            torch.cuda.synchronize(self.model.student_encoder.gene_embeddings.device)

            def elapsed(start: str, end: str) -> float:
                return float(events[start].elapsed_time(events[end]))

            teacher_forward_ms = elapsed("teacher_start", "teacher_end")
            student_global_ms = elapsed("student_global_start", "student_global_end")
            student_local_ms = elapsed("student_local_start", "student_local_end")
            prediction_ms = elapsed("prediction_start", "prediction_end")
            backward_update_ms = elapsed("backward_start", "backward_end")
        else:
            teacher_forward_ms = 0.0
            student_global_ms = 0.0
            student_local_ms = 0.0
            prediction_ms = 0.0
            backward_update_ms = 0.0

        metrics = GraDPertStepMetrics(
            total_loss=float(total_loss.detach().item()),
            prediction_loss=float(prediction_loss.detach().item()),
            condition_consistency_loss=float(condition_loss.detach().item()),
            masked_node_loss=float(masked_loss.detach().item()),
            spread_loss=float(spread_loss.detach().item()),
            spread_available=spread_available,
            teacher_momentum=momentum,
            prediction_graph_gradient_norm=prediction_gradient_norm,
            auxiliary_graph_gradient_norm=auxiliary_gradient_norm,
            prediction_to_auxiliary_gradient_ratio=(
                gradient_ratio if gradient_ratio is None or math.isfinite(gradient_ratio) else None
            ),
            condition_target_entropy=condition_entropy,
            masked_node_target_entropy=masked_entropy,
            condition_prototypes_used=condition_used,
            masked_node_prototypes_used=masked_used,
            condition_center_norm=float(self.centers.condition.norm().item()),
            masked_node_center_norm=float(self.centers.masked_node.norm().item()),
            unique_condition_count=len(views.anchors_by_condition),
            masked_node_count=len(masked_node_ids),
            batch_cell_count=len(batch.condition_ids),
            data_read_ms=batch.data_read_ms,
            host_to_device_ms=batch.host_to_device_ms,
            view_build_ms=view_build_ms,
            teacher_forward_ms=teacher_forward_ms,
            student_global_ms=student_global_ms,
            student_local_ms=student_local_ms,
            prediction_ms=prediction_ms,
            backward_update_ms=backward_update_ms,
            step_wall_ms=(time.perf_counter() - step_started) * 1000.0,
        )
        if capture_health:
            self.first_step_health = {
                "schema_version": "native-first-step-equivalence-v1",
                "perturbed_row_ids_sha256": batch.perturbed_row_ids_sha256,
                "control_row_ids_sha256": batch.control_row_ids_sha256,
                "pretransfer_control_sha256": batch.pretransfer_control_sha256,
                "pretransfer_target_sha256": batch.pretransfer_target_sha256,
                "view_structure_sha256": view_structure_sha256,
                "rng_state_before_sha256": rng_state_before_sha256,
                "rng_state_after_sha256": _rng_state_sha256(),
                "parameter_state_before_sha256": parameter_state_before_sha256,
                "parameter_state_after_sha256": _model_state_sha256(self.model),
                "losses": {
                    "total_loss": metrics.total_loss,
                    "prediction_loss": metrics.prediction_loss,
                    "condition_consistency_loss": metrics.condition_consistency_loss,
                    "masked_node_loss": metrics.masked_node_loss,
                    "spread_loss": metrics.spread_loss,
                },
                "loss_weights": {
                    "prediction": self.loss_weights.prediction,
                    "condition_consistency": self.loss_weights.condition_consistency,
                    "masked_node": self.loss_weights.masked_node,
                    "spread": self.loss_weights.spread,
                },
                "native_architecture": self.architecture.payload(),
                "update_order": update_order,
            }
        return metrics
