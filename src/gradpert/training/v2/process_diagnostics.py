"""Bounded, training-only representation probes with no training-state updates.

The caller supplies a sealed training batch; this module cannot establish row
provenance itself. Aggregates describe a fixed eval-mode probe, not population
metrics or a collapse verdict. No full graph activations/logits are retained.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn

from gradpert.hashing import sha256_json
from gradpert.modeling.v2.operators import RelayDeltaAttention

from . import objective as objective_module
from .diagnostic_objective import diagnostic_objective
from .engine import slice_cells
from .objective import JointObjective, TrainingBatch
from .reductions import population_weights

HEADS = ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node")
_OBSERVER_LOCK = threading.Lock()


def _entropy(p: Tensor) -> Tensor:
    return -(p * p.clamp_min(1e-300).log()).sum(-1)


class DistributionSummary:
    """CPU float64 accumulation; retain one prototype vector, never row logits."""

    def __init__(self, width: int, max_samples: int = 128) -> None:
        self.width, self.budget = width, max_samples
        self.count = self.observed = 0
        self.sum = torch.zeros(width, dtype=torch.float64)
        self.hits = torch.zeros(width, dtype=torch.long)
        self.entropy_sum = self.max_sum = 0.0

    def add(self, logits: Tensor) -> None:
        x = logits.detach().reshape(-1, self.width)
        self.observed += len(x)
        x = x[: max(0, self.budget - self.count)].double().cpu()
        if not len(x):
            return
        p = x.softmax(-1)
        self.count += len(x)
        self.sum += p.sum(0)
        self.hits += torch.bincount(p.argmax(-1), minlength=self.width)
        self.entropy_sum += float(_entropy(p).sum())
        self.max_sum += float(p.max(-1).values.sum())

    def result(self) -> dict[str, Any]:
        if not self.count:
            return {"samples": 0, "observed_samples": self.observed}
        h = self.entropy_sum / self.count
        hm = float(_entropy(self.sum / self.count))
        return {
            "samples": self.count,
            "observed_samples": self.observed,
            "mean_sample_entropy": h,
            "uniform_entropy": math.log(self.width),
            "marginal_entropy": hm,
            "sample_dependence_js": max(0.0, hm - h),
            "effective_marginal_prototypes": math.exp(hm),
            "mean_top1_probability": self.max_sum / self.count,
            "argmax_prototypes_used": int((self.hits > 0).sum()),
            "largest_argmax_fraction": float(self.hits.max()) / self.count,
        }


class FeatureSummary:
    """Bound covariance dimension and sample count independently of graph size."""

    def __init__(self, max_samples: int = 128, max_dimensions: int = 256) -> None:
        self.budget, self.max_dimensions = max_samples, max_dimensions
        self.count = self.observed = 0
        self.sum: Tensor | None = None
        self.outer: Tensor | None = None
        self.norm_sum = 0.0
        self.original_dimensions = 0

    def add(self, features: Tensor) -> None:
        # Slice before materializing a potentially expanded mHC/graph tensor.
        self.original_dimensions = features.shape[-1]
        remaining = max(0, self.budget - self.count)
        self.observed += features.numel() // features.shape[-1]
        if not remaining:
            return
        x = features.detach()
        if x.ndim > 2:
            x = x[: min(x.shape[0], remaining)]
        x = x.reshape(-1, x.shape[-1])[:remaining, : self.max_dimensions].double().cpu()
        if self.sum is None:
            self.sum = torch.zeros(x.shape[-1], dtype=torch.float64)
            self.outer = torch.zeros(x.shape[-1], x.shape[-1], dtype=torch.float64)
        assert self.outer is not None
        self.count += len(x)
        self.sum += x.sum(0)
        self.outer += x.T @ x
        self.norm_sum += float(x.norm(dim=-1).sum())

    def result(self) -> dict[str, Any]:
        if not self.count:
            return {"samples": 0, "observed_samples": self.observed}
        assert self.sum is not None and self.outer is not None
        mean = self.sum / self.count
        cov = self.outer / self.count - mean[:, None] * mean[None, :]
        eig = torch.linalg.eigvalsh(cov).clamp_min(0)
        trace = float(eig.sum())
        # Roundoff in a constant representation must not imply a high rank.
        scale = max(float(self.outer.trace()) / self.count, 1e-30)
        rank = math.exp(float(_entropy(eig / trace))) if trace > scale * 1e-12 else 0.0
        return {
            "samples": self.count,
            "observed_samples": self.observed,
            "original_dimensions": self.original_dimensions,
            "analyzed_dimensions": len(mean),
            "dimension_selection": "first_dimensions",
            "mean_norm": self.norm_sum / self.count,
            "total_centered_variance": trace,
            "relative_centered_variance": trace / scale,
            "covariance_effective_rank": rank,
        }


def prediction_summary(
    prediction: Tensor,
    control: Tensor,
    truth: Tensor,
    condition_index: Tensor,
    *,
    strategy: str = "row_mean",
    error_power: int = 2,
) -> dict[str, Any]:
    """Output diagnostics, not paired-cell causal accuracy or test metrics."""
    y, x, t = (v.detach().float() for v in (prediction, control, truth))
    if y.shape != x.shape or y.shape != t.shape or y.ndim != 2 or not y.numel():
        raise ValueError("aligned nonempty cell-by-gene matrices required")
    if error_power not in (2, 4):
        raise ValueError("prediction error power must be 2 or 4")
    weights = population_weights(
        condition_index, torch.ones_like(condition_index, dtype=torch.bool), strategy
    )
    delta, effect = y - x, t - x
    squared = (y - t).square()
    mse = float((squared.mean(-1) * weights).sum())
    copy = float((effect.square().mean(-1) * weights).sum())
    losses = (squared if error_power == 2 else squared.square()) * weights[:, None] / y.shape[1]
    flat = losses.flatten()
    total = float(flat.sum())
    tail = max(1, math.ceil(len(flat) / 100))
    negative = y < 0
    return {
        "elements": y.numel(),
        "prediction_mse": mse,
        "control_copy_mse": copy,
        "mse_gain_over_copy": 1 - mse / copy if copy else None,
        "negative_prediction_fraction": float(negative.float().mean()),
        "negative_prediction_mean_absolute": float(y[negative].abs().mean())
        if negative.any()
        else 0.0,
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std(unbiased=False)),
        "delta_rms": float(delta.square().mean().sqrt()),
        "truth_minus_paired_control_rms": float(effect.square().mean().sqrt()),
        "paired_effect_rms_ratio": float(
            delta.square().mean().sqrt() / effect.square().mean().sqrt()
        )
        if effect.square().sum() > 0
        else None,
        "top_one_percent_loss_fraction": float(flat.topk(tail).values.sum()) / total
        if total
        else None,
        "reduction": strategy,
        "effect_scope": "random_paired_training_cells_not_counterfactual",
    }


@contextmanager
def _read_only(objective: JointObjective, batch: TrainingBatch, seed: int) -> Iterator[None]:
    if objective.pending:
        raise ValueError("process diagnostics require no pending teacher statistics")
    modules = list(objective.modules())
    modes = [(m, m.training) for m in modules]
    orders = [
        (m, name, getattr(m, name))
        for m in modules
        for name in ("randomize_order", "randomize_relay_order", "eval_seed")
        if hasattr(m, name)
    ]
    counter = getattr(objective, "auxiliary_rng_counter", None)
    previous = counter.clone() if counter is not None else None
    device = batch.control.device
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            objective.eval()
            yield
    finally:
        objective.pending.clear()
        if counter is not None and previous is not None:
            counter.copy_(previous)
        for module, mode in modes:
            module.training = mode
        for module, name, value in orders:
            setattr(module, name, value)


def evaluate_process_diagnostics(
    objective: JointObjective,
    batch: TrainingBatch,
    *,
    bf16: bool,
    seed: int = 0x71A1,
    max_cells: int = 4,
    max_samples: int = 128,
    max_feature_dimensions: int = 256,
    intervention_budget: int = 0,
) -> dict[str, Any]:
    """Observe complete objective and optionally three extra prediction forwards.

    Hook samples are deterministic traversal prefixes, not unbiased graph samples.
    CE-pair decomposition uses bounded unweighted rows; actual weighted objective
    components are reported separately. This calls the bare objective, without
    engine/DDP population gathering or center commits. Head statistics describe
    this rank's fixed probe; never label them global training KoLeo.
    """
    if min(max_cells, max_samples, max_feature_dimensions) < 1 or intervention_budget not in range(
        4
    ):
        raise ValueError("positive probe bounds and intervention budget 0..3 required")
    batch = slice_cells(batch, 0, min(max_cells, len(batch.control)))
    if not len(batch.control):
        raise ValueError("process diagnostics need training rows")
    started = perf_counter()
    distributions: dict[str, DistributionSummary] = {}
    features: dict[str, FeatureSummary] = {}
    pairs: dict[str, list[float]] = {}
    logits_names: dict[int, str] = {}
    center_names = {id(getattr(objective, name + "_center")): name for name in HEADS}
    handles = []
    original_ce = objective_module.cross_entropy_terms

    def feature(key: str, value: Tensor) -> None:
        features.setdefault(key, FeatureSummary(max_samples, max_feature_dimensions)).add(value)

    def token_hook(
        key: str, has_cls: bool
    ) -> Callable[[nn.Module, tuple[Tensor, ...], Tensor], None]:
        def observe(module: nn.Module, values: tuple[Tensor, ...], output: Tensor) -> None:
            if not isinstance(output, Tensor):
                return
            # Manifold layers return [cell,gene,stream,width]; aggregate streams
            # per token, rather than treating stream copies as independent genes.
            value = output.detach()
            if has_cls and value.ndim == 4:
                value = value[:max_cells].mean(-2)
            if has_cls and value.ndim == 3:
                feature(key + ".genes", value[:, :-1, :])
                feature(key + ".cls", value[:, -1, :])
            else:
                feature(key, value)

        return observe

    def observed_ce(
        student: Tensor, teacher: Tensor, center: Tensor, *, teacher_temperature: float = 0.04
    ) -> Tensor:
        value = original_ce(student, teacher, center, teacher_temperature=teacher_temperature)
        key = center_names.get(id(center), logits_names.get(id(student), "unattributed"))
        n = min(max_samples, len(student))
        p = ((teacher[:n].detach().double() - center.double()) / teacher_temperature).softmax(-1)
        h = _entropy(p)
        ce = -(p * (student[:n].detach().double() / 0.1).log_softmax(-1)).sum(-1)
        tally = pairs.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        tally[0] += n
        tally[1] += float(ce.sum())
        tally[2] += float(h.sum())
        tally[3] += float((ce - h).sum())
        return value

    # Temporary CE observer is serialized and always removed on error.
    with _OBSERVER_LOCK, _read_only(objective, batch, seed):
        try:
            for role in ("student", "teacher"):
                model = getattr(objective, role)
                for name in HEADS:
                    key = role + "." + name
                    center = getattr(objective, name + "_center")
                    distributions[key] = DistributionSummary(model.options.prototypes, max_samples)

                    def head_hook(
                        head: nn.Module,
                        values: tuple[Tensor, ...],
                        output: Tensor,
                        key: str = key,
                        role: str = role,
                        center: Tensor = center,
                    ) -> None:
                        logits_names[id(output)] = key
                        # Bound rows before temporary float64/softmax conversion.
                        x = output[:max_samples].detach()
                        scaled = (
                            x.float() / 0.1
                            if role == "student"
                            else (x.float() - center) / objective.teacher_temperature
                        )
                        distributions[key].add(scaled)
                        distributions[key].observed += len(output) - len(x)
                        feature(key + ".input", values[0])

                    def bottleneck_hook(
                        head: nn.Module,
                        values: tuple[Tensor, ...],
                        output: Tensor,
                        key: str = key,
                    ) -> None:
                        feature(key + ".bottleneck", output)

                    head = getattr(model, name)
                    handles.append(head.register_forward_hook(head_hook))
                    handles.append(head.features.register_forward_hook(bottleneck_hook))
                handles.append(
                    model.graph.adapter.register_forward_hook(
                        token_hook(role + ".identity_map", False)
                    )
                )
                handles.append(
                    model.graph.register_forward_hook(token_hook(role + ".graph.output", False))
                )
                for index, layer in enumerate(model.graph.layers):
                    handles.append(
                        layer.register_forward_hook(token_hook(f"{role}.graph.layer{index}", False))
                    )
                for part in ("cell", "response"):
                    encoder = getattr(model, part)
                    handles.append(
                        encoder.register_forward_hook(
                            token_hook(role + "." + part + ".output", True)
                        )
                    )
                    layers = getattr(encoder, "layers", getattr(encoder, "ffn_layers", ()))
                    for index, layer in enumerate(layers):
                        handles.append(
                            layer.register_forward_hook(
                                token_hook(f"{role}.{part}.layer{index}", True)
                            )
                        )
            objective_module.cross_entropy_terms = observed_ce
            with (
                torch.no_grad(),
                torch.autocast(batch.control.device.type, dtype=torch.bfloat16, enabled=bf16),
            ):
                joint, terms, diagnostic_scope = diagnostic_objective(objective, batch)
            # Avoid folding intervention activations into the fixed-view statistics.
        finally:
            objective_module.cross_entropy_terms = original_ce
            for handle in handles:
                handle.remove()
        objective.pending.clear()
        with (
            torch.no_grad(),
            torch.autocast(batch.control.device.type, dtype=torch.bfloat16, enabled=bf16),
        ):

            def predict(selected: TrainingBatch) -> dict[str, Tensor]:
                graph, condition = objective._batch_graph(objective.student, selected)
                return objective.student.encode_response(
                    objective._select_graph(graph, selected.query_positions),
                    selected.control,
                    condition,
                    **objective.response_metadata(selected),
                )

            if objective.prediction_enabled:
                output = predict(batch)
                interventions = {}
                if intervention_budget:
                    changed = predict(replace(batch, control=batch.control.roll(1, 0)))
                    interventions["rolled_control"] = _output_change(output, changed)
                    interventions["rolled_control"]["context_matched"] = False
                    interventions["rolled_control"]["changed_rows"] = int(
                        (batch.control != batch.control.roll(1, 0)).any(-1).sum()
                    )
                if intervention_budget >= 2:
                    changed = predict(
                        replace(batch, condition_index=batch.condition_index.roll(1, 0))
                    )
                    interventions["rolled_condition"] = _output_change(output, changed)
                    interventions["rolled_condition"]["changed_rows"] = int(
                        (batch.condition_index != batch.condition_index.roll(1, 0)).sum()
                    )
                if intervention_budget >= 3:
                    model = objective.student
                    original_options = model.options
                    previous_seeds = [
                        (m, m.eval_seed)
                        for m in model.modules()
                        if isinstance(m, RelayDeltaAttention)
                    ]
                    try:
                        model.options = replace(original_options, relay_eval_seed=seed + 1)
                        for module, _ in previous_seeds:
                            module.eval_seed = seed + 1
                        changed = predict(batch)
                        interventions["changed_order_seed"] = _output_change(output, changed)
                        interventions["changed_order_seed"]["applies_to"] = (
                            "relay_graph_cell_response"
                        )
                        interventions["changed_order_seed"]["applicable"] = (
                            model.options.attention == "relay_full"
                        )
                    finally:
                        model.options = original_options
                        for module, value in previous_seeds:
                            module.eval_seed = value
                outputs = prediction_summary(
                    output["prediction"],
                    batch.control,
                    batch.truth,
                    batch.condition_index,
                    strategy=objective.prediction_strategy,
                    error_power=objective.prediction_error_power,
                )
            else:
                outputs = {"status": "not_applicable:ssl_pretrain_no_prediction_labels"}
                interventions = {}

    return {
        "schema": "training-process-diagnostics-1",
        "split": "train",
        "mode": "fixed_training_batch_eval_no_updates",
        "selection_use": "none",
        "input_sha256": sha256_json(
            {
                "queries": batch.query_positions.cpu().tolist(),
                "gene_ids": batch.graph.ids[batch.query_positions].cpu().tolist(),
                "conditions": batch.condition_index.cpu().tolist(),
                "control": batch.control.float().cpu().tolist(),
                "truth": batch.truth.float().cpu().tolist()
                if objective.prediction_enabled
                else "not_loaded:ssl_pretrain",
                "graph_views": [v.ids.cpu().tolist() for v in batch.graph_views],
                "cell_views": [
                    {"positions": v.positions.cpu().tolist(), "mask": v.mask.cpu().tolist()}
                    for v in batch.cell_views
                ],
            }
        ),
        "cells": len(batch.control),
        "query_genes": batch.control.shape[1],
        "seed": seed,
        "budget": {
            "feature_samples": max_samples,
            "feature_dimensions": max_feature_dimensions,
            "intervention_forwards": intervention_budget,
        },
        **diagnostic_scope,
        "joint_loss": float(joint),
        "components": {k: float(v) for k, v in terms.items()},
        **(
            {"training_stage": objective.training_stage}
            if hasattr(objective, "training_stage")
            else {}
        ),
        "distributions": {k: v.result() for k, v in distributions.items()},
        "centers": {
            name: {
                "l2": float(getattr(objective, name + "_center").norm()),
                "std": float(getattr(objective, name + "_center").std(unbiased=False)),
            }
            for name in HEADS
        },
        "features": {k: v.result() for k, v in features.items()},
        "pairs": {
            k: {
                "sample_pair_count": int(v[0]),
                "cross_entropy": v[1] / v[0],
                "teacher_entropy": v[2] / v[0],
                "teacher_to_student_kl": v[3] / v[0],
                "reduction": "unweighted_bounded_rows_per_pair",
            }
            for k, v in pairs.items()
            if v[0]
        },
        "outputs": outputs,
        "interventions": interventions,
        "elapsed_seconds": perf_counter() - started,
        "scope": {
            "sampling": "deterministic_prefix_not_population_estimate",
            "distributed": "rank0_fixed_probe_bare_objective_not_global_population",
            "gene_grouping": "none_train_expression_queries_only",
            "gradients": "not_computed_use_separate_loss_diagnostics",
            "identity_alignment": "query_positions_and_gene_ids_preserved",
        },
    }


def _output_change(base: dict[str, Tensor], changed: dict[str, Tensor]) -> dict[str, Any]:
    return {
        name + "_rms_change": float(
            (changed[name].float() - base[name].float()).square().mean().sqrt()
        )
        for name in ("prediction", "delta", "response_cls")
    }


def evaluate_graph_gradient_diagnostics(
    objective: JointObjective, batch: TrainingBatch, *, bf16: bool
) -> dict[str, Any]:
    """Compare weighted prediction/SSL1/SSL2 on their shared graph parameters.

    Autograd.grad never accumulates into parameter.grad. Report separate scopes,
    because graph SSL has no derivative on the ordinary Cell/response modules.
    The caller's existing basal diagnostics continue to cover that latter scope.
    """
    batch = slice_cells(batch, 0, min(4, len(batch.control)))
    parameters = [
        (name, value)
        for name, value in objective.student.named_parameters()
        if name.startswith("graph.") and value.requires_grad
    ]
    if not parameters:
        return {"performed": False, "reason": "no_trainable_graph_parameters"}
    started = perf_counter()
    with _read_only(objective, batch, 0x71A1):
        with torch.autocast(batch.control.device.type, dtype=torch.bfloat16, enabled=bf16):
            _, terms, diagnostic_scope = diagnostic_objective(objective, batch)
            zero = next(iter(terms.values())) * 0
            losses = {"prediction": terms["prediction"]} if objective.prediction_enabled else {}
            for index, (prefix, names) in enumerate(
                (
                    ("ssl1", ("condition", "node", "spread")),
                    ("ssl2", ("dino", "ibot", "koleo")),
                )
            ):
                coefficient = objective.lambda1 if index == 0 else objective.lambda2
                losses[prefix] = coefficient * sum(
                    weight * terms.get(prefix + "_" + name, zero)
                    for weight, name in zip(objective.weights[index], names, strict=True)
                )
        gradients = {
            name: torch.autograd.grad(
                value,
                [p for _, p in parameters],
                retain_graph=i + 1 < len(losses),
                allow_unused=True,
            )
            for i, (name, value) in enumerate(losses.items())
        }
        groups = {}
        for prefix in ("graph.adapter", "graph.layers", "graph"):
            indices = [i for i, (name, _) in enumerate(parameters) if name.startswith(prefix)]
            if not indices:
                continue
            norm2 = {
                name: sum(
                    float(g[i].detach().float().square().sum()) for i in indices if g[i] is not None
                )
                for name, g in gradients.items()
            }
            cosines = {}
            for other in ("ssl1", "ssl2") if "prediction" in gradients else ():
                dot = sum(
                    float(
                        (
                            gradients["prediction"][i].detach().float()
                            * gradients[other][i].detach().float()
                        ).sum()
                    )
                    for i in indices
                    if gradients["prediction"][i] is not None and gradients[other][i] is not None
                )
                denominator = math.sqrt(norm2["prediction"] * norm2[other])
                cosines["prediction_" + other] = dot / denominator if denominator else None
            groups[prefix] = {
                "weighted_gradient_l2": {k: math.sqrt(v) for k, v in norm2.items()},
                "cosine": cosines,
                "parameter_tensors": len(indices),
            }
    return {
        "performed": True,
        **diagnostic_scope,
        "split": "train",
        "groups": groups,
        "elapsed_seconds": perf_counter() - started,
        "selection_use": "none",
        "scope": "rank0_fixed_training_probe_shared_graph_not_global_population",
    }
