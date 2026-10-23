"""Server-only train-population AU calibration, with zero parameter updates.

Each endpoint is an independent single-GPU observation, not distributed resume.
Two-pass vector-Jacobian replay retains the complete effective AU population
without retaining every microbatch's encoder graph. All saved outputs aggregate
training data; no representation matrices or expression values are exported.

Run on the server with the original sealed RI configuration and published source:
  PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/v2/calibrate_alignment_uniformity.py
    --mode gradients --config RI.yaml --data-root /data/yilangliu/data
    --output /data/yilangliu/calibration/init --publication publication.json
    --publication-sha256 HASH --source-commit COMMIT --run-id UNIQUE_ID
    --shared-policy policy.json --shared-policy-sha256 HASH --gpu 0 --stop-after 1
Use absolute server paths. Omit --stop-after and add --resume for the same eight
populations. The checkpoint endpoint also needs --checkpoint, --checkpoint-sha256,
--training-manifest and --training-manifest-sha256. CPU --mode census needs no
policy or GPU. Concurrent endpoints must use their common --resource-owner-pid.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def metadata_census(conditions, controls, contexts):
    """Count exact first-occurrence (control,p) dedup and compatible pairs."""
    from collections import Counter

    if not (len(conditions) == len(controls) == len(contexts)) or not conditions:
        raise ValueError("nonempty aligned condition/control/context metadata required")
    control_context = {}
    for control, context in zip(controls, contexts, strict=True):
        if control in control_context and control_context[control] != context:
            raise ValueError("one control cannot have multiple compatible contexts")
        control_context[control] = context
    seen, keep = set(), []
    for i, key in enumerate(zip(controls, conditions, strict=True)):
        if key not in seen:
            keep.append(i)
            seen.add(key)
    rows = []
    for context in sorted(set(contexts)):
        counts = Counter(conditions[i] for i in keep if contexts[i] == context)
        n = sum(counts.values())
        pairs = (n * n - sum(k * k for k in counts.values())) // 2
        rows.append(
            dict(
                context_id=str(context),
                input_count=n,
                condition_count=len(counts),
                pair_count=pairs,
            )
        )
    total = sum(row["pair_count"] for row in rows)
    for row in rows:
        row["pair_weight"] = row["pair_count"] / total if total else None
    return dict(
        input_count=len(conditions),
        unique_count=len(keep),
        duplicate_count=len(conditions) - len(keep),
        pair_count=total,
        context_count=len(rows),
        contexts=rows,
    ), keep


def spectrum(values):
    """Raw centered feature covariance; no identity jitter or fabricated rank."""
    import torch

    x = values.detach().double().cpu()
    if len(x) < 2:
        return dict(effective_rank=None, normalized_rank=None, relative_variance=None)
    centered = x - x.mean(0)
    covariance = centered.T @ centered / len(x)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    energy = eigenvalues.sum()
    total = x.square().sum(-1).mean()
    positive = eigenvalues[eigenvalues > 0]
    probabilities = positive / energy if energy > 0 else positive
    rank = float((-(probabilities * probabilities.log()).sum()).exp()) if energy > 0 else 0.0
    return dict(
        effective_rank=rank,
        normalized_rank=rank / min(x.shape[1], len(x) - 1),
        relative_variance=float(energy / total) if total > 0 else None,
        covariance_trace=float(energy),
        mean_squared_norm=float(total - energy),
        sample_count=len(x),
        width=x.shape[1],
    )


def rng_capture(device):
    import torch

    return (
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


def rng_restore(state, device):
    import torch

    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state(state[3], device)


@contextmanager
def preserved_observer(runtime, seed):
    """Restore observational RNG/pending/modes, detect all model/state mutations."""
    import torch
    from capacity_probe import validation_state

    objective = runtime.objective
    before = validation_state(runtime)
    rng = rng_capture(runtime.device)
    generator = copy.deepcopy(runtime.generator.bit_generator.state)
    pending = copy.deepcopy(objective.pending)
    modes = [
        (module, module.training, getattr(module, "randomize_relay_order", None))
        for module in objective.modules()
    ]
    try:
        objective.train()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if runtime.device.type == "cuda":
            torch.cuda.manual_seed(seed)
        objective.pending.clear()
        yield before
    finally:
        objective.pending.clear()
        objective.pending.update(pending)
        runtime.generator.bit_generator.state = generator
        rng_restore(rng, runtime.device)
        for module, mode, order in modes:
            module.training = mode
            if order is not None:
                module.randomize_relay_order = order
        after = validation_state(runtime)
        if before != after:
            changed = [key for key in before if before[key] != after[key]]
            raise RuntimeError("calibration mutated forbidden state: " + ", ".join(changed))


MECHANISM_PROFILE = "mechanism-v1"
MECHANISM_SCOPE = dict(
    joint="prediction+weighted_SSL1+weighted_SSL2; excludes active AU",
    CR="negative uncentered coding rate; full deduplicated population; no context filtering",
    SSL1="lambda1-weighted original graph self-distillation",
    SSL2="lambda2-weighted original response self-distillation; derived micro joint minus MSE",
    SSL2_activity="union of parent-bank autograd participation; not direct structural reachability",
    SSL2_numerics="FP32 gradient subtraction can leave tiny cancellation residuals",
    inactive_parameters="autograd-unused parameters; SSL2 uses parent-bank union",
    zero_gradient_parameters="exact zeros in accumulated diagnostic gradient bank",
    interpretation="raw parameter gradients only; not optimizer displacement",
    epsilon_squared=0.05,
)


def negative_coding_rate(global0, global1, epsilon_squared=0.05):
    """Native diagnostic candidate; normalize each Global separately, without centering."""
    import torch

    rates = []
    for values in (global0, global1):
        if values.ndim != 2 or not all(values.shape):
            raise ValueError("coding rate requires nonempty feature matrices")
        x = values.double()
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        if not torch.isfinite(x).all() or (norms <= 0).any():
            raise ValueError("coding rate requires finite nonzero unit norms")
        if epsilon_squared != 0.05:
            raise ValueError("diagnostic epsilon_squared is fixed at 0.05")
        unit = x / norms
        n, d = unit.shape
        gram = unit.T @ unit
        matrix = torch.eye(d, dtype=x.dtype, device=x.device) + d / (n * epsilon_squared) * gram
        matrix = (matrix + matrix.T) / 2
        factor, info = torch.linalg.cholesky_ex(matrix)
        if info.any() or not torch.isfinite(factor).all():
            raise ValueError("coding rate matrix is not finite positive definite")
        rates.append(-factor.diagonal().log().sum())
    return sum(rates) / 2


def gradient_summary(banks, names, *, profile=None, activity=None):
    """FP64 dot products of complete accumulated parameter gradients."""
    import torch

    result = {}
    for group in (
        ("response", "graph", "cell", "prediction", "response_cls")
        if profile
        else ("response", "graph")
    ):
        indices = [
            i for i, name in enumerate(names) if name == group or name.startswith(group + ".")
        ]
        if not indices and profile:
            continue
        if not indices:
            raise ValueError("missing calibration parameter group: " + group)
        norm2 = {
            label: sum(float(grad[i].double().square().sum()) for i in indices)
            for label, grad in banks.items()
        }
        pairs = {}
        for a, b in itertools.combinations(sorted(banks) if profile else banks, 2):
            dot = sum(
                float(torch.dot(banks[a][i].double().reshape(-1), banks[b][i].double().reshape(-1)))
                for i in indices
            )
            denominator = math.sqrt(norm2[a] * norm2[b])
            cosine = max(-1.0, min(1.0, dot / denominator)) if denominator else None
            pairs[a + "__" + b] = dict(
                **({"dot": dot} if profile else {}),
                cosine=cosine,
                angle_degrees=math.degrees(math.acos(cosine)) if cosine is not None else None,
            )
        result[group] = dict(
            norms={label: math.sqrt(v) for label, v in norm2.items()},
            pairs=pairs,
            parameter_count=sum(banks["A"][i].numel() for i in indices),
        )
        if profile:
            result[group]["inactive_parameters"] = {
                label: [names[i] for i in indices if not activity[label][i]] for label in banks
            }
            result[group]["zero_gradient_parameters"] = {
                label: [names[i] for i in indices if not banks[label][i].any()] for label in banks
            }
    return result


def calibrate_batch(
    runtime,
    batch,
    *,
    microbatch=80,
    temperature=2.0,
    bf16=True,
    replay_tolerance=2e-6,
    return_gradients=False,
    profile=None,
):
    """Exact chain-rule AU gradients for one full population; no optimizer call.

    First pass only saves detached CLS and pre-forward RNG per microbatch. AU
    derivatives are calculated once on both full populations. Identical replay
    forwards map those derivatives back to parameters, summing without another
    microbatch mean. Original losses use full-population row/iBOT weights.
    """
    import torch

    from gradpert.training.v2.alignment_uniformity import alignment_uniformity
    from gradpert.training.v2.engine import slice_cells
    from gradpert.training.v2.reductions import population_weights

    if profile not in (None, MECHANISM_PROFILE):
        raise ValueError("unknown diagnostic profile")
    obj, device = runtime.objective, runtime.device
    if (
        obj.loss_reduction != "row_mean"
        or obj.population_response
        or obj.weights[1][2]
        or obj.student.options.control_conditioned_graph
        or obj.masked_response_ratio
        or obj.auxiliary_mask_ratio
        or not obj.lambda2
    ):
        raise ValueError(
            "calibration requires original RI joint/row_mean without other new objectives"
        )
    if microbatch < 1 or batch.context_index is None:
        raise ValueError("positive microbatch and compatible-context metadata required")
    parameters = [
        (name, p)
        for name, p in obj.student.named_parameters()
        if p.requires_grad
        and (
            name.startswith(("response.", "graph."))
            or (profile and (name == "response_cls" or name.startswith(("cell.", "prediction."))))
        )
    ]
    names, params = zip(*parameters, strict=True)
    banks = {
        key: [torch.zeros_like(p, device="cpu", dtype=torch.float32) for p in params]
        for key in ("A", "U", "MSE", "joint")
    }
    activity = {key: [False] * len(params) for key in banks}
    if profile:
        banks["CR"] = [torch.zeros_like(p, device="cpu", dtype=torch.float32) for p in params]
        activity["CR"] = [False] * len(params)
    n = len(batch.control)
    ids = batch.condition_index
    row = population_weights(ids, torch.ones_like(ids, dtype=torch.bool), "row_mean")
    valid = [view.mask.any(-1) for view in batch.cell_views[:2]]
    active = max(1, sum(bool(v.any()) for v in valid))
    nodes = torch.stack([population_weights(ids, v, "row_mean") / active for v in valid])
    predictions = population_weights(
        ids, torch.ones_like(ids, dtype=torch.bool), obj.prediction_strategy
    )
    replay, chunks, counts = [], [], []
    losses = {name: 0.0 for name in banks}
    starts = list(range(0, n, microbatch))

    def forward(start, collect):
        end = min(start + microbatch, n)
        micro = slice_cells(batch, start, end)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            loss, terms = obj(
                micro,
                include_ssl1=False,
                reduction_weights=(row[start:end], nodes[:, start:end]),
                prediction_weights=predictions[start:end],
                row_weights=row[start:end],
                deferred_au=collect,
                checkpoint_deferred_response=True,
            )
        if len(collect) != 1:
            raise ValueError("each microbatch must return exactly two Global populations")
        return loss, terms

    for start in starts:
        replay.append(rng_capture(device))
        collect = []
        with torch.no_grad():
            forward(start, collect)
        chunks.append(tuple(x.detach().clone() for x in collect[0]))
        counts.append(len(chunks[-1][0]))
        obj.pending.clear()
    end_forward_rng = rng_capture(device)
    z = [torch.cat([chunk[v] for chunk in chunks]).float().requires_grad_(True) for v in range(2)]
    condition, context = (torch.cat([chunk[v] for chunk in chunks]) for v in (2, 3))
    au = alignment_uniformity(z[0], z[1], condition, context, temperature=temperature)
    da = torch.autograd.grad(au.alignment, z, retain_graph=True)
    du = torch.autograd.grad(au.uniformity, z)
    if profile:
        cr = negative_coding_rate(*z)
        dc = torch.autograd.grad(cr, z)
        losses["CR"] = float(cr.detach())
    losses.update(A=float(au.alignment.detach()), U=float(au.uniformity.detach()))
    statistics = {f"global{v}": spectrum(z[v]) for v in range(2)}

    def accumulate(label, scalar, retain_graph=False):
        if not torch.isfinite(scalar):
            raise FloatingPointError("nonfinite calibration " + label)
        gradients = torch.autograd.grad(
            scalar, params, allow_unused=True, retain_graph=retain_graph
        )
        for i, gradient in enumerate(gradients):
            if gradient is not None:
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("nonfinite calibration gradient " + label)
                activity[label][i] = True
                banks[label][i].add_(gradient.detach().float().cpu())
        return gradients

    offset, maximum_error = 0, 0.0
    for index, start in enumerate(starts):
        rng_restore(replay[index], device)
        collect = []
        joint, terms = forward(start, collect)
        values = collect[0]
        for v in range(4):
            if values[v].shape != chunks[index][v].shape:
                raise ValueError("replay changed population shape")
            if v < 2:
                error = (
                    float((values[v].detach().float() - chunks[index][v].float()).abs().max())
                    if values[v].numel()
                    else 0.0
                )
                maximum_error = max(maximum_error, error)
                if error > replay_tolerance:
                    raise ValueError("no-grad/replay CLS mismatch exceeds fixed tolerance")
            elif not torch.equal(values[v], chunks[index][v]):
                raise ValueError("replay changed population identities")
        count = counts[index]
        a = sum((values[v].float() * da[v][offset : offset + count]).sum() for v in range(2))
        u = sum((values[v].float() * du[v][offset : offset + count]).sum() for v in range(2))
        accumulate("MSE", terms["prediction_mse"], True)
        accumulate("joint", joint, True)
        accumulate("A", a, True)
        accumulate("U", u, bool(profile))
        if profile:
            c = sum((values[v].float() * dc[v][offset : offset + count]).sum() for v in range(2))
            accumulate("CR", c)
        losses["MSE"] += float(terms["prediction_mse"].detach())
        losses["joint"] += float(joint.detach())
        offset += count
        obj.pending.clear()
        del collect, values, joint, terms, a, u
    rng_restore(end_forward_rng, device)
    if profile:
        if obj.prediction_strategy != "row_mean":
            raise ValueError("mechanism decomposition requires row_mean prediction")
        banks["SSL2"] = [a - b for a, b in zip(banks["joint"], banks["MSE"], strict=True)]
        activity["SSL2"] = [a or b for a, b in zip(activity["joint"], activity["MSE"], strict=True)]
        losses["SSL2"] = losses["joint"] - losses["MSE"]
        graph_gradients = [None] * len(params)
        losses["SSL1"] = 0.0
    if obj.lambda1:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            graph_terms = obj.graph_loss(
                batch.graph_views,
                batch.condition_index,
                control=batch.control,
                query_gene_ids=batch.graph.ids[batch.query_positions],
            )
            graph_loss = obj.lambda1 * sum(
                w * value for w, value in zip(obj.weights[0], graph_terms.values(), strict=True)
            )
        graph_gradients = accumulate("joint", graph_loss)
        losses["joint"] += float(graph_loss.detach())
        if profile:
            losses["SSL1"] = float(graph_loss.detach())
    obj.pending.clear()
    if profile:
        banks["SSL1"] = [
            torch.zeros_like(p, device="cpu", dtype=torch.float32)
            if g is None
            else g.detach().float().cpu()
            for p, g in zip(params, graph_gradients, strict=True)
        ]
        activity["SSL1"] = [g is not None for g in graph_gradients]
    result = dict(
        losses=losses,
        gradients=gradient_summary(banks, names, profile=profile, activity=activity),
        spectrum=statistics,
        pair_count=au.pair_count,
        unique_count=len(z[0]),
        replay_max_abs_error=maximum_error,
        replay_tolerance=replay_tolerance,
        updates=0,
    )
    if profile:
        result.update(profile=profile, loss_scope=MECHANISM_SCOPE)
    if return_gradients:
        return result, banks
    return result


def _spec_census(data, spec):
    from gradpert.hashing import sha256_json

    contexts = [str(data.context_ids[data._row_index[row]]) for row in spec.control_row_ids]
    truth_context = [str(data.context_ids[data._row_index[row]]) for row in spec.perturbed_row_ids]
    if contexts != truth_context:
        raise ValueError("training control and perturbed contexts differ")
    if not set(spec.condition_ids) <= set(data.split.train_conditions):
        raise ValueError("calibration touched a nontraining condition")
    result, _ = metadata_census(list(spec.condition_ids), list(spec.control_row_ids), contexts)
    result["ordered_identity_sha256"] = sha256_json(
        dict(
            control=list(spec.control_row_ids),
            truth=list(spec.perturbed_row_ids),
            condition=list(spec.condition_ids),
            context=contexts,
        )
    )
    return result


def require_raw_identity(raw, spec):
    """Bind the real sampled tensors to the sealed ordered metadata schedule."""
    for name in ("control_row_ids", "perturbed_row_ids", "condition_ids"):
        if tuple(getattr(raw, name)) != tuple(getattr(spec, name)):
            raise ValueError("actual training batch differs from census: " + name)


def validate_batch_receipt(record, *, index, plan_hash, census, profile=None):
    """Partial or mismatched observations cannot satisfy calibration completion."""
    if (
        record.get("plan_sha256") != plan_hash
        or record.get("batch_index") != index
        or record.get("census") != census
        or record.get("state_unchanged") is not True
        or record.get("updates") != 0
        or record.get("unique_count") != census["unique_count"]
        or record.get("pair_count") != census["pair_count"]
        or not record.get("frozen_state")
        or not record.get("environment")
        or record.get("replay_tolerance") != 2e-6
        or not 0 <= record.get("replay_max_abs_error", math.inf) <= 2e-6
    ):
        raise ValueError("batch receipt does not establish complete unchanged observation")
    expected_losses = {"A", "U", "MSE", "joint"} | ({"CR", "SSL1", "SSL2"} if profile else set())
    if record.get("profile") != profile or (
        profile and record.get("loss_scope") != MECHANISM_SCOPE
    ):
        raise ValueError("batch receipt diagnostic profile differs")
    if set(record.get("losses", {})) != expected_losses:
        raise ValueError("batch receipt lacks all four calibrated losses")
    if not all(math.isfinite(value) for value in record["losses"].values()):
        raise ValueError("batch receipt contains nonfinite losses")
    groups = set(record.get("gradients", {}))
    expected_groups = (
        {"graph", "response", "cell", "prediction", "response_cls"}
        if profile
        else {"graph", "response"}
    )
    if groups != expected_groups or set(record.get("spectrum", {})) != {"global0", "global1"}:
        raise ValueError("batch receipt lacks parameter groups or representation statistics")
    if profile:
        expected_pairs = {
            a + "__" + b for a, b in itertools.combinations(sorted(expected_losses), 2)
        }
        for group in record["gradients"].values():
            if (
                set(group)
                != {
                    "norms",
                    "pairs",
                    "parameter_count",
                    "inactive_parameters",
                    "zero_gradient_parameters",
                }
                or set(group["norms"]) != expected_losses
                or set(group["pairs"]) != expected_pairs
                or set(group["inactive_parameters"]) != expected_losses
                or set(group["zero_gradient_parameters"]) != expected_losses
                or not all(math.isfinite(v) and v >= 0 for v in group["norms"].values())
                or not isinstance(group["parameter_count"], int)
                or group["parameter_count"] < 1
            ):
                raise ValueError("mechanism receipt lacks complete gradient banks")
            for pair in group["pairs"].values():
                if (
                    set(pair) != {"dot", "cosine", "angle_degrees"}
                    or not math.isfinite(pair["dot"])
                    or (pair["cosine"] is not None and not -1 <= pair["cosine"] <= 1)
                    or (pair["angle_degrees"] is not None and not 0 <= pair["angle_degrees"] <= 180)
                ):
                    raise ValueError("mechanism receipt contains invalid gradient comparison")


def verified_batch_index(output, plan_hash, census, *, profile=None):
    """Use a separate checksum index; an orphan batch is not silently accepted."""
    from cap40_posthoc import read, sha, write

    path = output / "BATCH_INDEX.json"
    if not path.exists():
        if list(output.glob("batch-*.json")):
            raise ValueError("unindexed batch receipt requires explicit recovery")
        write(path, dict(plan_sha256=plan_hash, receipts={}))
    index = read(path)
    if index.get("plan_sha256") != plan_hash or not isinstance(index.get("receipts"), dict):
        raise ValueError("batch checksum index belongs to another plan")
    expected_names = {f"batch-{i:02d}.json" for i in range(8)}
    observed_names = {p.name for p in output.glob("batch-*.json")}
    if observed_names != set(index["receipts"]) or not observed_names <= expected_names:
        raise ValueError("batch checksum index and files differ")
    for name, checksum in index["receipts"].items():
        if sha(output / name) != checksum:
            raise ValueError("batch receipt checksum mismatch: " + name)
        number = int(name[6:8])
        validate_batch_receipt(
            read(output / name),
            index=number,
            plan_hash=plan_hash,
            census=census[number],
            profile=profile,
        )
    return index


def _census_data(config, options, data_root, run_seed):
    """CPU-only canonical metadata census; does not instantiate a model."""
    from gradpert.training.data import CanonicalTrainingData
    from gradpert.training.v2.train_selection import apply_train_selection

    with CanonicalTrainingData(
        dataset_id=config.dataset_id,
        protocol_id=config.data.protocol_id,
        data_root=data_root,
        run_seed=run_seed,
    ) as data:
        data.require_experiment_data_contract(
            registry_version=config.data.registry_version, split_policy=config.data.split_policy
        )
        if options.train_selection_path:
            apply_train_selection(
                data, Path(options.train_selection_path), options.train_selection_sha256
            )
        specs = data.training_batch_identity_specs(
            epoch=0, batch_size=320, max_unique_conditions=options.max_conditions
        )
        if len(specs) < 8 or any(len(s.condition_ids) != 320 for s in specs[:8]):
            raise ValueError("eight complete original training populations of 320 required")
        return [_spec_census(data, spec) for spec in specs[:8]]


def execute(args):
    import torch
    from cap40_posthoc import read, server_path, sha, write

    from gradpert.config import load_experiment_config
    from gradpert.config.v2 import V2Options
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.hashing import sha256_json

    for path in (args.config, args.data_root, args.output, args.publication):
        server_path(path)
    profile = getattr(args, "profile", None)
    config = load_experiment_config(args.config)
    _, options = V2Options.parse_parameters(config.model.parameters)
    if int(config.training.train_batch_size.value) != 320 or options.microbatch != 80:
        raise ValueError("calibration contract is original global320, micro80")
    source = inspect_source_identity(
        Path(__file__).resolve().parents[2],
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    if source.commit != args.source_commit:
        raise ValueError("calibration source differs from planned clean published commit")
    if not profile and any(
        getattr(options, name) for name in ("au_alignment_weight", "au_uniformity_weight")
    ):
        raise ValueError("calibration must use sealed unmodified RI base configuration")
    checkpoint_flags = [
        bool(value)
        for value in (
            args.checkpoint,
            args.checkpoint_sha256,
            args.training_manifest,
            args.training_manifest_sha256,
        )
    ]
    if any(checkpoint_flags) and not all(checkpoint_flags):
        raise ValueError(
            "checkpoint, hash and original training manifest must be supplied together"
        )
    checkpoint_identity = None
    if args.checkpoint:
        for path in (args.checkpoint, args.training_manifest):
            server_path(path)
        if (
            sha(args.checkpoint) != args.checkpoint_sha256
            or sha(args.training_manifest) != args.training_manifest_sha256
        ):
            raise ValueError("checkpoint/training manifest checksum mismatch")
        checkpoint_identity = read(args.training_manifest)
        if checkpoint_identity["resolved_config_sha256"] != sha256_json(
            config.model_dump(mode="json")
        ):
            raise ValueError("old checkpoint requires its original exact resolved config")
    plan = dict(
        schema="gradpert-au-calibration-1",
        run_id=args.run_id,
        mode=args.mode,
        source=source.payload(),
        config_sha256=sha(args.config),
        seed=args.seed,
        checkpoint_sha256=args.checkpoint_sha256,
        training_identity=checkpoint_identity,
        batches=8,
        effective_population=320,
        microbatch=80,
        temperature=args.temperature,
        rng_seed=args.calibration_seed,
        topology="single_gpu_diagnostic_not_ddp_resume",
        updates=0,
        split="train_only",
        gpu=args.gpu,
        shared_policy_sha256=args.shared_policy_sha256,
        cpu_environment=inspect_environment(
            Path(__file__).resolve().parents[2], device_name="cpu", require_cuda=False
        ).payload(),
    )
    if profile:
        plan.update(
            schema="gradpert-mechanism-calibration-1", profile=profile, loss_scope=MECHANISM_SCOPE
        )
    plan_hash = sha256_json(plan)
    if args.output.exists():
        if not args.resume or read(args.output / "PLAN.json") != plan:
            raise ValueError("fresh output required or exact immutable plan for resume")
    else:
        args.output.mkdir(parents=True)
        write(args.output / "PLAN.json", plan)
    census = _census_data(config, options, args.data_root, args.seed)
    census_receipt = dict(
        plan_sha256=plan_hash, batches=census, complete=True, model_forward=False, updates=0
    )
    census_path = args.output / "CENSUS.json"
    if census_path.exists():
        if read(census_path) != census_receipt:
            raise ValueError("current census differs from immutable archived census")
    else:
        write(census_path, census_receipt)
    if args.mode == "census":
        return dict(
            census_complete=True,
            calibration_complete=False,
            scientific_complete=False,
            plan_sha256=plan_hash,
            batches=len(census),
            updates=0,
        )
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator contract missing")
    if not args.shared_policy or not args.shared_policy_sha256:
        raise ValueError("gradient calibration requires the sealed resource policy")
    receipt_index = verified_batch_index(args.output, plan_hash, census, profile=profile)
    server_path(args.shared_policy)
    if sha(args.shared_policy) != args.shared_policy_sha256:
        raise ValueError("shared resource policy checksum mismatch")
    from run_group import lock
    from shared_gpu import admit, descendant, parent_map, snapshot

    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.runtime import _move_batch_tree, prepare_runtime
    from gradpert.training.v2.views import assemble_batch

    policy_doc = read(args.shared_policy)
    policy = policy_doc.get("shared_gpu_policy", policy_doc)
    owner_pid = args.resource_owner_pid or os.getpid()
    if owner_pid != os.getpid() and not descendant(os.getpid(), owner_pid, parent_map()):
        raise ValueError("resource owner must be this process or its verified ancestor")
    with lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{args.gpu}.lock"):
        admission = snapshot(owner_pid)
        admit(policy, admission)
        torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], 0)
        torch.set_num_threads(1)
        environment = inspect_environment(
            Path(__file__).resolve().parents[2], device_name="cuda:0"
        ).payload()
        environment_path = args.output / "ENVIRONMENT.json"
        if environment_path.exists() and read(environment_path) != environment:
            raise ValueError("calibration resume environment changed")
        if not environment_path.exists():
            write(environment_path, environment)
        write(args.output / "ADMISSION.json", dict(policy=policy, observed=admission))
        with prepare_runtime(
            config,
            data_root=args.data_root,
            run_seed=args.seed,
            device=torch.device("cuda:0"),
            purpose="evaluation",
        ) as runtime:
            if checkpoint_identity is not None:
                if runtime.identity != checkpoint_identity["data"]:
                    raise ValueError("checkpoint and current runtime data identity differ")
                load_evaluation_checkpoint(
                    args.checkpoint,
                    runtime.objective,
                    training_identity=checkpoint_identity,
                    checkpoint_sha256=args.checkpoint_sha256,
                )
                payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                runtime.optimizer.load_state_dict(payload["optimizer"])
                del payload
            specs = runtime.data.training_batch_identity_specs(
                epoch=0, batch_size=320, max_unique_conditions=options.max_conditions
            )
            generator = np.random.default_rng(args.seed)
            context_ids = {
                name: i for i, name in enumerate(sorted(set(map(str, runtime.data.context_ids))))
            }
            raw_batches = runtime.data.iter_train_epoch(
                epoch=0,
                device=torch.device("cpu"),
                batch_size=320,
                max_unique_conditions=options.max_conditions,
            )
            for index, raw in enumerate(itertools.islice(raw_batches, 8)):
                require_raw_identity(raw, specs[index])
                contexts = torch.tensor(
                    [
                        context_ids[str(runtime.data.context_ids[runtime.data._row_index[row]])]
                        for row in raw.control_row_ids
                    ],
                    dtype=torch.long,
                )
                batch = assemble_batch(
                    raw,
                    runtime.index,
                    options,
                    generator,
                    allowed_expression_ids=runtime.allowed_expression_ids,
                    context_index=contexts,
                )
                expected = _spec_census(runtime.data, specs[index])
                if expected != census[index]:
                    raise ValueError("CPU census and GPU calibration schedule differ")
                output = args.output / f"batch-{index:02d}.json"
                if output.exists():
                    continue
                if args.stop_after is not None and index >= args.stop_after:
                    break
                admit(policy, snapshot(owner_pid))
                batch = _move_batch_tree(batch, runtime.device)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with preserved_observer(runtime, args.calibration_seed + index) as before:
                    result = calibrate_batch(
                        runtime, batch, temperature=args.temperature, profile=profile
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if (
                    result["unique_count"] != expected["unique_count"]
                    or result["pair_count"] != expected["pair_count"]
                ):
                    raise ValueError("observed AU population differs from CPU metadata census")
                result.update(
                    plan_sha256=plan_hash,
                    batch_index=index,
                    census=expected,
                    state_unchanged=True,
                    frozen_state=before,
                    environment=environment,
                    elapsed_seconds=elapsed,
                    cells_per_second=320 / elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
                validate_batch_receipt(
                    result, index=index, plan_hash=plan_hash, census=expected, profile=profile
                )
                write(output, result)
                receipt_index["receipts"][output.name] = sha(output)
                write(args.output / "BATCH_INDEX.json", receipt_index)
                write(
                    args.output / "progress.json",
                    dict(
                        batches_completed=index + 1,
                        expected=8,
                        updates=0,
                        calibration_complete=False,
                    ),
                )
                del batch
    if args.checkpoint and (
        sha(args.checkpoint) != args.checkpoint_sha256
        or sha(args.training_manifest) != args.training_manifest_sha256
    ):
        raise ValueError("checkpoint or training manifest changed during observation")
    completed = verified_batch_index(args.output, plan_hash, census, profile=profile)["receipts"]
    result = dict(
        calibration_complete=len(completed) == 8,
        scientific_complete=False,
        updates=0,
        plan_sha256=plan_hash,
        batches_completed=len(completed),
        expected=8,
        batch_receipts=completed,
    )
    write(
        args.output / ("COMPLETE.json" if result["calibration_complete"] else "PARTIAL.json"),
        result,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("census", "gradients"), required=True)
    for name in ("config", "data-root", "output", "publication"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("publication-sha256", "source-commit", "run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--profile", choices=(MECHANISM_PROFILE,))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--calibration-seed", type=int, default=41239)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--gpu", choices=("0", "1"), default="0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--training-manifest-sha256")
    parser.add_argument("--shared-policy", type=Path)
    parser.add_argument("--shared-policy-sha256")
    parser.add_argument("--resource-owner-pid", type=int)
    parser.add_argument("--stop-after", type=int, choices=range(1, 9))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.mode == "gradients" else ""
    try:
        result = execute(args)
    except Exception as error:
        from cap40_posthoc import read, server_path, write

        failure = dict(
            run_id=args.run_id,
            failed=True,
            calibration_complete=False,
            scientific_complete=False,
            error_type=type(error).__name__,
            error=str(error),
            timestamp_ns=time.time_ns(),
        )
        # Do not mutate an unrelated output when plan verification itself fails.
        try:
            server_path(args.output)
            plan = read(args.output / "PLAN.json")
            if plan["run_id"] == args.run_id and plan["source"]["commit"] == args.source_commit:
                prior = args.output / "FAILURE.json"
                if prior.exists():
                    write(args.output / f"FAILURE-{failure['timestamp_ns']}.json", read(prior))
                write(prior, failure)
        except (OSError, ValueError, KeyError):
            pass
        print(json.dumps(failure, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
