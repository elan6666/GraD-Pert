"""Fixed training-input diagnostics without optimization or state mutation."""

from __future__ import annotations

from typing import Any

import torch

from gradpert.hashing import sha256_json

from .objective import JointObjective, TrainingBatch
from .reductions import population_weights


def evaluate_loss_diagnostics(
    objective: JointObjective, batch: TrainingBatch, *, bf16: bool
) -> dict[str, Any]:
    if objective.pending:
        raise ValueError("training diagnostics require a committed update boundary")
    was_training = objective.training
    device = batch.control.device
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    counter = (
        objective.auxiliary_rng_counter.detach().clone() if objective.auxiliary_mask_ratio else None
    )
    parameters = tuple(
        p
        for name, p in objective.student.named_parameters()
        if p.requires_grad and (name.startswith(("cell.", "expression.")) or name == "control_cls")
    )
    if not parameters:
        raise ValueError("diagnostics need shared basal parameters")
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            objective.eval()
            if counter is not None:
                objective.auxiliary_rng_counter.zero_()
            # Graph SSL has no path to the Cell/expression parameters diagnosed
            # below. Compute its values without retaining graph-encoder activations.
            graph_metrics = {}
            if objective.lambda1:
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16),
                ):
                    graph_terms = objective.graph_loss(batch.graph_views, batch.condition_index)
                    graph_metrics = {
                        f"ssl1_{name}": float(value) for name, value in graph_terms.items()
                    }
                del graph_terms
                objective.pending.clear()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                loss, terms = objective(batch, include_ssl1=False)
                del loss
                zero = terms["prediction"] * 0
                ssl = sum(
                    objective.lambda2 * weight * terms.get(f"ssl2_{name}", zero)
                    for weight, name in zip(
                        objective.weights[1], ("dino", "ibot", "koleo"), strict=True
                    )
                )
                weighted = {"prediction": terms["prediction"], "ssl": ssl}
                if objective.auxiliary_mask_ratio:
                    weighted.update(
                        gene_mask=objective.lambda_gene_mask * terms["gene_mask"],
                        cls_mask=objective.lambda_cls_mask * terms["cls_mask"],
                    )
            gradients = {
                name: torch.cat(
                    [
                        (torch.zeros_like(p) if g is None else g).float().flatten()
                        for p, g in zip(
                            parameters,
                            torch.autograd.grad(
                                value,
                                parameters,
                                retain_graph=index + 1 < len(weighted),
                                allow_unused=True,
                            ),
                            strict=True,
                        )
                    ]
                )
                for index, (name, value) in enumerate(weighted.items())
            }
            components = {name: float(value.detach()) for name, value in terms.items()}
            components.update(graph_metrics)
            # Release every differentiable objective before the residual-only read.
            del terms, weighted, ssl, zero
            objective.pending.clear()
            norms = {name: float(value.norm()) for name, value in gradients.items()}
            denominator = norms["prediction"] * norms["ssl"]
            cosine = (
                float(torch.dot(gradients["prediction"], gradients["ssl"]) / denominator)
                if denominator
                else None
            )
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16),
            ):
                graph, conditions = objective._graph(objective.student, batch.graph, False)
                output = objective.student.encode_response(
                    graph[batch.query_positions], batch.control, conditions[batch.condition_index]
                )
                residual_matrix = (output["prediction"].float() - batch.truth.float()).abs()
                contributions = residual_matrix.square()
                if objective.prediction_error_power == 4:
                    contributions = contributions.square()
                row_weights = population_weights(
                    batch.condition_index,
                    torch.ones_like(batch.condition_index, dtype=torch.bool),
                    objective.prediction_strategy,
                )
                contributions = (
                    contributions * row_weights[:, None] / residual_matrix.shape[1]
                ).flatten()
                residual = residual_matrix.flatten()
                tail = max(1, (len(contributions) + 99) // 100)
                total = float(contributions.sum())
                tail_fraction = (
                    float(contributions.topk(tail).values.sum()) / total if total else None
                )
                quantiles = residual.quantile(residual.new_tensor([0.5, 0.9, 0.99])).tolist()
            copy_mse, model_mse = (
                components["control_copy_mse"],
                components["prediction_mse"],
            )
            return {
                "split": "train",
                "mode": "fixed_training_batch_eval",
                "input_sha256": sha256_json(
                    {
                        "query": batch.query_positions.cpu().tolist(),
                        "conditions": batch.condition_index.cpu().tolist(),
                        "control_sha256": sha256_json(batch.control.float().cpu().tolist()),
                        "truth_sha256": sha256_json(batch.truth.float().cpu().tolist()),
                    }
                ),
                "cells": len(batch.control),
                "query_genes": batch.control.shape[1],
                "components": components,
                "control_copy_mse": copy_mse,
                "prediction_mse": model_mse,
                "mse_gain_over_copy": 1 - model_mse / copy_mse if copy_mse else None,
                "shared_basal_weighted_gradient_l2": norms,
                "prediction_ssl_gradient_cosine": cosine,
                "absolute_residual_quantiles": dict(
                    zip(("p50", "p90", "p99"), quantiles, strict=True)
                ),
                "top_one_percent_loss_fraction": tail_fraction,
                "selection_use": "none",
            }
    finally:
        objective.pending.clear()
        if counter is not None:
            objective.auxiliary_rng_counter.copy_(counter)
        objective.train(was_training)
