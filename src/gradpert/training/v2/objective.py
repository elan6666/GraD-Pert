"""Joint prediction and two independently weighted self-distillation paths."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.model import GraphContext, PredictionMetadata
from gradpert.modeling.v2.reconstruction import ControlReconstruction

from .reductions import nearest_neighbor_terms, population_weights


@dataclass
class GraphView:
    ids: Tensor
    neighbors: Tensor
    valid: Tensor
    sources: Tensor
    target_positions: Tensor
    target_valid: Tensor
    masked_positions: Tensor
    context: GraphContext | None = None


@dataclass
class CellView:
    positions: Tensor
    mask: Tensor


@dataclass
class TrainingBatch:
    graph: GraphView
    query_positions: Tensor
    control: Tensor
    truth: Tensor
    # Each row maps to one distinct condition in graph.target_positions.
    condition_index: Tensor
    graph_views: tuple[GraphView, ...]
    cell_views: tuple[CellView, ...]
    unique_cell_indices: Tensor | None = None


def cross_entropy(
    student: Tensor, teacher: Tensor, center: Tensor, weights: Tensor | None = None
) -> Tensor:
    terms = cross_entropy_terms(student, teacher, center)
    return terms.mean() if weights is None else (terms * weights).sum() / weights.sum()


def cross_entropy_terms(student: Tensor, teacher: Tensor, center: Tensor) -> Tensor:
    probabilities = ((teacher.detach().float() - center) / 0.04).softmax(-1)
    return -(probabilities * (student.float() / 0.1).log_softmax(-1)).sum(-1)


def nearest_spread(x: Tensor) -> Tensor:
    if len(x) < 2:
        return x.sum() * 0
    x = F.normalize(x.float(), dim=-1)
    # Detach nearest-neighbor selection only; distances retain both gradients.
    similarity = x.detach() @ x.detach().T
    similarity.fill_diagonal_(float("-inf"))
    nearest = similarity.argmax(-1)
    return cast(Tensor, -(torch.linalg.vector_norm(x - x[nearest], dim=-1) + 1e-8).log().mean())


class JointObjective(nn.Module):
    """Teacher/centers are part of the resumable module state.

    forward never updates teacher or centers. commit_statistics runs exactly
    once after a successful optimizer step. Failed/nonfinite steps discard the
    pending statistics. An external accumulator must retain them until commit.
    """

    auxiliary_rng_counter: Tensor
    ssl1_cls_center: Tensor
    ssl1_node_center: Tensor
    ssl2_cls_center: Tensor
    ssl2_node_center: Tensor

    def __init__(
        self,
        student: GraDPertV2,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        ssl1_weights: tuple[float, float, float] = (0.8, 0.4, 0.1),
        ssl2_weights: tuple[float, float, float] = (0.8, 0.4, 0.1),
        ssl1_reduction: str = "condition_mean",
        prediction_reduction: str = "cell_mean",
        loss_reduction: str | None = None,
        koleo_exclude_same_condition: bool = False,
        prediction_error_power: int = 2,
        prediction_reduction_override: str = "inherit",
        auxiliary_mask_ratio: float = 0.0,
        lambda_gene_mask: float = 0.0,
        lambda_cls_mask: float = 0.0,
        auxiliary_seed: int = 1,
    ) -> None:
        super().__init__()
        if min(lambda1, lambda2, *ssl1_weights, *ssl2_weights) < 0:
            raise ValueError("loss weights must be nonnegative")
        self.student = student
        self.teacher = copy.deepcopy(student).requires_grad_(False).eval()
        if prediction_error_power not in (2, 4):
            raise ValueError("prediction error power must be 2 or 4")
        if prediction_reduction_override not in ("inherit", "row_mean", "condition_mean"):
            raise ValueError("unknown prediction-only reduction")
        if not 0 <= auxiliary_mask_ratio < 1 or min(lambda_gene_mask, lambda_cls_mask) < 0:
            raise ValueError("invalid reconstruction settings")
        if bool(auxiliary_mask_ratio) != bool(lambda_gene_mask or lambda_cls_mask):
            raise ValueError("reconstruction mask and supervision must be enabled together")
        self.prediction_error_power = prediction_error_power
        self.prediction_reduction_override = prediction_reduction_override
        self.auxiliary_mask_ratio = auxiliary_mask_ratio
        self.lambda_gene_mask, self.lambda_cls_mask = lambda_gene_mask, lambda_cls_mask
        self.auxiliary_seed = auxiliary_seed
        if auxiliary_mask_ratio:
            # Initialize after the complete main model/Teacher, without consuming
            # their CPU/CUDA random streams. Auxiliary heads belong only to Student.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(auxiliary_seed)
                student.control_reconstruction = ControlReconstruction(student.options.width).to(
                    next(student.parameters()).device
                )
            self.register_buffer("auxiliary_rng_counter", torch.zeros((), dtype=torch.long))
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.weights = (ssl1_weights, ssl2_weights)
        if loss_reduction not in (None, "row_mean", "condition_mean"):
            raise ValueError("unknown unified loss reduction")
        self.loss_reduction = loss_reduction
        self.koleo_exclude_same_condition = koleo_exclude_same_condition
        if koleo_exclude_same_condition and loss_reduction is None:
            raise ValueError("condition-excluding KoLeo requires a unified global population")
        if loss_reduction is not None:
            ssl1_reduction = loss_reduction
            prediction_reduction = "cell_mean" if loss_reduction == "row_mean" else loss_reduction
        if ssl1_reduction not in ("condition_mean", "row_mean") or prediction_reduction not in (
            "cell_mean",
            "condition_mean",
        ):
            raise ValueError("unknown objective reduction")
        self.ssl1_reduction, self.prediction_reduction = ssl1_reduction, prediction_reduction
        for name in ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node"):
            self.register_buffer(name + "_center", torch.zeros(student.options.prototypes))
        self.pending: dict[str, list[Tensor]] = {}

    def train(self, mode: bool = True) -> JointObjective:
        super().train(mode)
        self.teacher.eval()
        if self.student.options.attention == "relay_full":
            self.student.set_relay_order_randomization(mode)
            self.teacher.set_relay_order_randomization(mode)
        return self

    def _targets(self, name: str, logits: Tensor) -> None:
        if logits.numel():
            self.pending.setdefault(name, []).append(
                torch.stack(
                    (
                        logits.detach().float().sum(0),
                        logits.new_full((logits.shape[-1],), len(logits), dtype=torch.float32),
                    )
                )
            )

    def _graph(self, model: GraDPertV2, view: GraphView, masked: bool) -> tuple[Tensor, Tensor]:
        ids = view.ids[view.masked_positions] if masked else None
        graph = model.graph(view.ids, view.neighbors, view.valid, view.sources, ids, view.context)
        condition = model.aggregate_targets(graph, view.target_positions, view.target_valid)
        return graph, condition

    def response_metadata(
        self, batch: TrainingBatch, positions: Tensor | None = None
    ) -> PredictionMetadata:
        if not (
            self.student.options.gene_conditioned_readout or self.student.options.direct_target_flag
        ):
            return {}
        query_positions = (
            batch.query_positions if positions is None else batch.query_positions[positions]
        )
        targets = batch.graph.ids[batch.graph.target_positions.clamp_min(0)]
        return self.student.prediction_metadata(
            batch.graph.ids[query_positions],
            targets[batch.condition_index],
            batch.graph.target_valid[batch.condition_index],
        )

    def forward(
        self,
        batch: TrainingBatch,
        *,
        include_ssl1: bool = True,
        ibot_population: Tensor | None = None,
        reduction_weights: tuple[Tensor, Tensor] | None = None,
        prediction_weights: Tensor | None = None,
        row_weights: Tensor | None = None,
        deferred_koleo: list[tuple[Tensor, Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        model = self.student
        graph, conditions = self._graph(model, batch.graph, False)
        condition = conditions[batch.condition_index]
        response = model.encode_response(
            graph[batch.query_positions], batch.control, condition, **self.response_metadata(batch)
        )
        squared = (response["prediction"].float() - batch.truth.float()).square()
        mse_rows = squared.mean(-1)
        errors = mse_rows if self.prediction_error_power == 2 else squared.square().mean(-1)
        if prediction_weights is not None:
            loss = (errors * prediction_weights).sum()
        elif self.prediction_reduction_override != "inherit":
            weights = population_weights(
                batch.condition_index,
                torch.ones_like(batch.condition_index, dtype=torch.bool),
                self.prediction_reduction_override,
            )
            loss = (errors * weights).sum()
        elif self.loss_reduction is not None:
            if reduction_weights is None:
                reduction_weights = self.local_reduction_weights(batch)
            loss = (errors * reduction_weights[0]).sum()
        elif self.prediction_reduction == "condition_mean":
            unique_conditions = torch.unique(batch.condition_index)
            loss = torch.stack(
                [errors[batch.condition_index == c].mean() for c in unique_conditions]
            ).mean()
        else:
            loss = errors.mean()
        metrics = {"prediction": loss}
        if row_weights is None:
            row_weights = torch.full_like(mse_rows, 1.0 / len(mse_rows))
        metrics["prediction_mse"] = (mse_rows * row_weights).sum()
        metrics["control_copy_mse"] = (
            (batch.control.float() - batch.truth.float()).square().mean(-1) * row_weights
        ).sum()
        if self.auxiliary_mask_ratio:
            auxiliary = self.control_reconstruction_loss(batch, graph, row_weights)
            metrics.update(auxiliary)
            if self.lambda_gene_mask:
                loss = loss + self.lambda_gene_mask * auxiliary["gene_mask"]
            if self.lambda_cls_mask:
                loss = loss + self.lambda_cls_mask * auxiliary["cls_mask"]
        if self.lambda1 and include_ssl1:
            ssl1 = self.graph_loss(batch.graph_views, batch.condition_index)
            metrics.update({f"ssl1_{k}": v for k, v in ssl1.items()})
            loss = loss + self.lambda1 * sum(
                w * v for w, v in zip(self.weights[0], ssl1.values(), strict=True)
            )
        if self.lambda2:
            # Independent teacher graph; shared teacher parameters are EMA-updated once.
            with torch.no_grad():
                tg, tc = self._graph(self.teacher, batch.graph, False)
            ssl2 = self.cell_loss(
                batch,
                graph,
                condition,
                tg,
                tc[batch.condition_index],
                ibot_population=ibot_population,
                reduction_weights=reduction_weights,
                deferred_koleo=deferred_koleo,
            )
            metrics.update({f"ssl2_{k}": v for k, v in ssl2.items()})
            loss = loss + self.lambda2 * sum(
                w * v for w, v in zip(self.weights[1], ssl2.values(), strict=True)
            )
        return loss, metrics

    def graph_loss(
        self, views: tuple[GraphView, ...], condition_index: Tensor | None = None
    ) -> dict[str, Tensor]:
        if len(views) < 2:
            raise ValueError("SSL1 requires two globals")
        active_views = views if self.weights[0][0] else views[:2]
        student_states = [self._graph(self.student, v, True) for v in active_views]
        with torch.no_grad():
            teacher_states = [self._graph(self.teacher, v, False) for v in views[:2]]
            teacher_logits = (
                [self.teacher.ssl1_cls(c) for _, c in teacher_states] if self.weights[0][0] else []
            )
        student_logits = (
            [self.student.ssl1_cls(c) for _, c in student_states] if self.weights[0][0] else []
        )
        counts = None
        if self.ssl1_reduction == "row_mean":
            if condition_index is None:
                raise ValueError("row reduction needs condition frequencies")
            counts = torch.bincount(condition_index, minlength=len(student_states[0][1])).float()
        terms = []
        for j, target in enumerate(teacher_logits):
            self._targets("ssl1_cls", target)
            for i, source in enumerate(student_logits):
                if i != j:
                    terms.append(cross_entropy(source, target, self.ssl1_cls_center, counts))
        node_terms = []
        for i, view in enumerate(views[:2]):
            selected = view.masked_positions
            if self.weights[0][1] and selected.numel():
                source = self.student.ssl1_node(student_states[i][0][selected])
                with torch.no_grad():
                    target = self.teacher.ssl1_node(teacher_states[i][0][selected])
                self._targets("ssl1_node", target)
                node_terms.append(cross_entropy(source, target, self.ssl1_node_center))
        zero = student_states[0][0].sum() * 0
        return {
            "condition": torch.stack(terms).mean() if terms else zero,
            "node": torch.stack(node_terms).mean() if node_terms else zero,
            "spread": (
                torch.stack([nearest_spread(c) for _, c in student_states[:2]]).mean()
                if self.weights[0][2]
                else zero
            ),
        }

    @property
    def prediction_strategy(self) -> str:
        if self.prediction_reduction_override != "inherit":
            return self.prediction_reduction_override
        return self.loss_reduction or (
            "row_mean" if self.prediction_reduction == "cell_mean" else "condition_mean"
        )

    def control_reconstruction_loss(
        self, batch: TrainingBatch, graph: Tensor, row_weights: Tensor
    ) -> dict[str, Tensor]:
        decoder = self.student.control_reconstruction
        if decoder is None:
            raise ValueError("control reconstruction decoder is absent")
        device = batch.control.device
        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        counter = int(self.auxiliary_rng_counter.item())
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        seed = self.auxiliary_seed + 104729 * counter + 1000003 * rank
        with torch.random.fork_rng(devices=cuda_devices):
            torch.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            count = max(1, int(batch.control.shape[1] * self.auxiliary_mask_ratio))
            positions = torch.rand(batch.control.shape, device=device).argsort(-1)[:, :count]
            mask = torch.zeros_like(batch.control, dtype=torch.bool).scatter_(1, positions, True)
            gene = graph[batch.query_positions]
            basal, cls, _ = self.student.encode_control(
                gene, batch.control, mask, mask_token=decoder.mask_token
            )
            token, cell = decoder(basal, cls, gene)
            targets = batch.control.float()
            losses = {
                "gene_mask": ((token.float() - targets).square() * mask).sum(-1) / count,
                "cls_mask": ((cell.float() - targets).square() * mask).sum(-1) / count,
            }
        if self.training:
            self.auxiliary_rng_counter.add_(1)
        return {name: (value * row_weights).sum() for name, value in losses.items()}

    def local_reduction_weights(self, batch: TrainingBatch) -> tuple[Tensor, Tensor]:
        strategy = self.loss_reduction or "row_mean"
        ids = batch.condition_index
        row = population_weights(ids, torch.ones_like(ids, dtype=torch.bool), strategy)
        valid = [v.mask.any(-1) for v in batch.cell_views[:2]]
        active = max(1, sum(bool(v.any()) for v in valid))
        nodes = torch.stack([population_weights(ids, v, strategy) / active for v in valid])
        return row, nodes

    def cell_loss(
        self,
        batch: TrainingBatch,
        graph: Tensor,
        condition: Tensor,
        teacher_graph: Tensor,
        teacher_condition: Tensor,
        *,
        ibot_population: Tensor | None = None,
        reduction_weights: tuple[Tensor, Tensor] | None = None,
        deferred_koleo: list[tuple[Tensor, Tensor, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        if len(batch.cell_views) < 2:
            raise ValueError("SSL2 requires two globals")
        student_outputs, teacher_outputs = [], []
        active_views = batch.cell_views if self.weights[1][0] else batch.cell_views[:2]
        for i, view in enumerate(active_views):
            p = view.positions
            args = (graph[batch.query_positions[p]], batch.control[:, p], condition, view.mask)
            metadata = self.response_metadata(batch, p)
            student_outputs.append(self.student.encode_response(*args, **metadata))
            if i < 2:
                with torch.no_grad():
                    teacher_outputs.append(
                        self.teacher.encode_response(
                            teacher_graph[batch.query_positions[p]],
                            batch.control[:, p],
                            teacher_condition,
                            **metadata,
                        )
                    )
        source_logits = (
            [self.student.ssl2_cls(o["response_cls"]) for o in student_outputs]
            if self.weights[1][0]
            else []
        )
        with torch.no_grad():
            target_logits = (
                [self.teacher.ssl2_cls(o["response_cls"]) for o in teacher_outputs]
                if self.weights[1][0]
                else []
            )
        terms, nodes = [], []
        for j, target in enumerate(target_logits):
            self._targets("ssl2_cls", target)
            for i, source in enumerate(source_logits):
                if i != j:
                    terms.append(
                        cross_entropy(source, target, self.ssl2_cls_center)
                        if reduction_weights is None
                        else (
                            cross_entropy_terms(source, target, self.ssl2_cls_center)
                            * reduction_weights[0]
                        ).sum()
                    )
        for j in range(2):
            mask = batch.cell_views[j].mask
            if self.weights[1][1] and mask.any():
                source = self.student.ssl2_node(student_outputs[j]["response_tokens"][mask])
                with torch.no_grad():
                    target_node = self.teacher.ssl2_node(
                        teacher_outputs[j]["response_tokens"][mask]
                    )
                self._targets("ssl2_node", target_node)
                node_loss = cross_entropy(source, target_node, self.ssl2_node_center)
                if reduction_weights is not None:
                    token_terms = cross_entropy_terms(source, target_node, self.ssl2_node_center)
                    rows = mask.nonzero(as_tuple=True)[0]
                    per_row = token_terms.new_zeros(len(mask)).scatter_add(0, rows, token_terms)
                    per_row = per_row / mask.sum(-1).clamp_min(1)
                    node_loss = (per_row * reduction_weights[1][j]).sum()
                elif ibot_population is not None:
                    # Return a cell-scaled contribution so the engine's row
                    # weighting cancels and yields the global masked-token mean.
                    node_loss = (
                        node_loss
                        * (mask.sum() / ibot_population[j])
                        * (ibot_population[2] / len(batch.control))
                    )
                nodes.append(node_loss)
        zero = student_outputs[0]["response_cls"].sum() * 0
        if self.weights[1][2] and self.loss_reduction is not None:
            selected = batch.unique_cell_indices
            values = [
                o["response_cls"] if selected is None else o["response_cls"][selected]
                for o in student_outputs[:2]
            ]
            ids = batch.condition_index if selected is None else batch.condition_index[selected]
            if deferred_koleo is not None:
                deferred_koleo.append((values[0], values[1], ids))
                koleo = zero
            else:
                weights = population_weights(
                    ids, torch.ones_like(ids, dtype=torch.bool), self.loss_reduction
                )
                koleo = torch.stack(
                    [
                        (
                            nearest_neighbor_terms(
                                v,
                                ids if self.koleo_exclude_same_condition else None,
                            )
                            * weights
                        ).sum()
                        for v in values
                    ]
                ).mean()
        else:
            koleo = None
        return {
            "dino": torch.stack(terms).mean() if terms else zero,
            "ibot": (
                (
                    torch.stack(nodes).mean()
                    if ibot_population is None and reduction_weights is None
                    else torch.stack(nodes).sum()
                    if reduction_weights is not None
                    else torch.stack(nodes).sum()
                    / (cast(Tensor, ibot_population)[:2] > 0).sum().clamp_min(1)
                )
                if nodes
                else zero
            ),
            "koleo": koleo
            if koleo is not None
            else torch.stack(
                [
                    nearest_spread(
                        o["response_cls"]
                        if batch.unique_cell_indices is None
                        else o["response_cls"][batch.unique_cell_indices]
                    )
                    for o in student_outputs[:2]
                ]
            ).mean()
            if self.weights[1][2]
            else zero,
        }

    @torch.no_grad()
    def commit_statistics(self, momentum: float) -> None:
        if not 0 <= momentum <= 1:
            raise ValueError("invalid teacher EMA momentum")
        parameters = dict(self.student.named_parameters())
        for name, teacher in self.teacher.named_parameters():
            teacher.lerp_(parameters[name], 1 - momentum)
        # Every rank enters the same collectives, even when its masked-token
        # population is empty. Aggregate sums/counts, not rank means.
        for name in ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node"):
            center = getattr(self, name + "_center")
            statistics = self.pending.get(name)
            sums = (
                torch.stack(statistics).sum(0)
                if statistics
                else center.new_zeros((2, center.numel()))
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(sums)
            if sums[1, 0] > 0:
                center.lerp_(sums[0] / sums[1], 0.1)
        self.pending.clear()
