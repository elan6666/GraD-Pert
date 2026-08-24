"""Exact B2 forward, loss, optimizer, Teacher EMA, and center update order."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from gradpert.graphs import (
    GraDPertTrainingViews,
    GraphTopology,
    build_training_graph_views,
    clean_graph_view,
)
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
    ssl_graph_gradient_norm: float
    weighted_ssl_graph_gradient_norm: float
    prediction_to_weighted_ssl_gradient_ratio: float | None
    condition_target_entropy: float
    masked_node_target_entropy: float | None
    condition_prototypes_used: int
    masked_node_prototypes_used: int
    condition_center_norm: float
    masked_node_center_norm: float
    unique_condition_count: int
    masked_node_count: int


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


def _gradient_norm(loss: Tensor, parameters: tuple[Tensor, ...]) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return float(torch.sqrt(squared).item())


def _distribution_health(probabilities: Tensor) -> tuple[float, int]:
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()
    prototypes_used = int(torch.unique(probabilities.argmax(dim=-1)).numel())
    return float(entropy.item()), prototypes_used


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
        self.model.train()
        views = build_training_graph_views(
            self.topology,
            anchors_by_condition=batch.anchors_by_condition,
            heldout_target_ids=self.heldout_target_ids,
            run_seed=self.run_seed,
            global_step=global_step,
        )

        clean_globals = tuple(clean_graph_view(view) for view in views.globals)
        with torch.no_grad():
            teacher_encoded = tuple(self.model.teacher_encoder(view) for view in clean_globals)
            teacher_condition_states = tuple(
                _stack_condition_states(encoded, views) for encoded in teacher_encoded
            )
            teacher_condition_logits = tuple(
                self.model.teacher_projector(states) for states in teacher_condition_states
            )

        student_encoded = tuple(self.model.student_encoder(view) for view in views.globals)
        student_global_states = tuple(
            _stack_condition_states(encoded, views) for encoded in student_encoded
        )
        student_view_logits: list[Tensor] = [
            self.model.student_projector(states) for states in student_global_states
        ]
        for local_index in range(8):
            local_states = []
            for condition_id in views.anchors_by_condition:
                encoded = self.model.student_encoder(
                    views.locals_by_condition[condition_id][local_index]
                )
                local_states.append(
                    encoded.condition_state(views.anchors_by_condition[condition_id])
                )
            student_view_logits.append(self.model.student_projector(torch.stack(local_states)))

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
        ssl_loss = condition_loss + masked_loss + 0.1 * spread_loss

        prediction = self.model.predict_expression_batch(
            batch.control_expression,
            views.prediction,
            batch.condition_ids,
            views.anchors_by_condition,
        )
        prediction_loss = F.mse_loss(prediction, batch.target_expression)
        total_loss = prediction_loss + 0.1 * ssl_loss

        graph_parameters = tuple(
            parameter
            for name, parameter in self.model.student_encoder.named_parameters()
            if name != "mask_token" and parameter.requires_grad
        )
        prediction_gradient_norm = _gradient_norm(prediction_loss, graph_parameters)
        ssl_gradient_norm = _gradient_norm(ssl_loss, graph_parameters)
        weighted_ssl_gradient_norm = 0.1 * ssl_gradient_norm
        gradient_ratio = (
            prediction_gradient_norm / weighted_ssl_gradient_norm
            if weighted_ssl_gradient_norm > 0
            else None
        )

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

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
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
        update_center(
            self.centers.condition,
            torch.cat(teacher_condition_logits),
        )
        if masked_node_ids:
            update_center(self.centers.masked_node, teacher_masked_logits)

        return GraDPertStepMetrics(
            total_loss=float(total_loss.detach().item()),
            prediction_loss=float(prediction_loss.detach().item()),
            condition_consistency_loss=float(condition_loss.detach().item()),
            masked_node_loss=float(masked_loss.detach().item()),
            spread_loss=float(spread_loss.detach().item()),
            spread_available=spread_available,
            teacher_momentum=momentum,
            prediction_graph_gradient_norm=prediction_gradient_norm,
            ssl_graph_gradient_norm=ssl_gradient_norm,
            weighted_ssl_graph_gradient_norm=weighted_ssl_gradient_norm,
            prediction_to_weighted_ssl_gradient_ratio=(
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
        )
