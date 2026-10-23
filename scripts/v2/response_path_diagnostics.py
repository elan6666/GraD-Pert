"""Bounded, truth-free F/B/D path audit on an exact crossed input panel.

Only one context's activations survive between groups. The caller owns native
assembly, resource locks, eval/no_grad, checkpoint and complete runtime state.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
from cap40_posthoc import main_response
from cap40_posthoc_stats import digest_json

TOLERANCE = 2e-6
CONTEXT_KEYS = ("cell_line", "cell_type", "context_id", "batch")
FEATURES = ("control_cls", "response_cls", "response_tokens", "score", "prediction")


def decomposition(values):
    """Equal-weight condition x control ANOVA energies, per scalar coordinate."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim < 3 or not x.size or min(x.shape[:2]) < 2 or not np.isfinite(x).all():
        raise ValueError("finite complete condition x control x feature grid required")
    mean = x.mean((0, 1), keepdims=True)
    condition = x.mean(1, keepdims=True) - mean
    control = x.mean(0, keepdims=True) - mean
    interaction = x - mean - condition - control
    energies = {
        "condition": float(np.square(condition).mean()),
        "control": float(np.square(control).mean()),
        "interaction": float(np.square(interaction).mean()),
    }
    total = float(np.square(x - mean).mean())
    return {
        "total_energy": total,
        **{name + "_energy": value for name, value in energies.items()},
        **{
            name + "_fraction": value / total if total else None for name, value in energies.items()
        },
        "fraction_unavailable_reason": None if total else "zero_centered_energy",
    }


def pair_statistics(left, right):
    """Pooled per-coordinate RMS; cosine averaged over valid last-axis vectors."""
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 2 or not a.size:
        raise ValueError("aligned nonempty paired feature matrices required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite paired features")
    product = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    valid = product > 0
    cosine = (a * b).sum(-1)[valid] / product[valid]
    return {
        "rms": float(np.sqrt(np.square(a - b).mean())),
        "cosine": float(np.clip(cosine, -1, 1).mean()) if valid.any() else None,
        "valid_vector_pairs": int(valid.sum()),
        "vector_pairs": int(valid.size),
        "cosine_unavailable_reason": None if valid.any() else "zero_vector_norm",
    }


def _fingerprint(values):
    return hashlib.sha256(values.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def _finite(values):
    import torch

    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError("nonfinite path-audit activation")


def decode_cached(model, tokens, condition, control, metadata):
    """Call native decoding only; never rerun or modify the encoder in D."""
    import torch

    joint = torch.cat((tokens, condition[:, None, :].expand_as(tokens)), dim=-1)
    score = model.decode_score(joint, **metadata)
    prediction, _ = model.prediction_outputs(score, control)
    result = {"score": score, "prediction": prediction}
    _finite(result)
    return result


def collect_group(rt, batch, cached, group):
    """Three sealed forwards: F, deterministic F replay, all-layer blocked B."""
    import torch

    model, obj = rt.objective.student, rt.objective
    options = model.options
    if torch.is_grad_enabled() or obj.training or model.training:
        raise ValueError("path audit requires eval and no_grad")
    if (
        options.attention != "relay_full"
        or options.control_conditioned_graph
        or options.prediction_parameterization != "softplus"
        or options.direct_target_flag
        or isinstance(cached, dict)
    ):
        raise ValueError("path audit supports static relay softplus without direct-target flags")
    ids = batch.graph.ids[batch.query_positions]
    if ids.tolist() != group["queries"] or len(batch.control) != len(group["rows"]):
        raise ValueError("assembled path-audit row/query alignment differs")
    if batch.condition_index.unique().numel() != 1:
        raise ValueError("one sealed condition per path-audit group required")
    target_ids = batch.graph.ids[batch.graph.target_positions.clamp_min(0)]
    row = int(batch.condition_index[0])
    actual = target_ids[row][batch.graph.target_valid[row]].tolist()
    if actual != list(rt.data.anchors_by_condition[group["condition_id"]]):
        raise ValueError("assembled perturbation targets differ from panel condition")
    target = cached[target_ids] * batch.graph.target_valid.unsqueeze(-1)
    condition = target.sum(1) / batch.graph.target_valid.sum(1).clamp_min(1).unsqueeze(-1)
    condition = condition[batch.condition_index]
    metadata = obj.response_metadata(batch)
    normal = main_response(rt, batch, cached)
    replay = main_response(rt, batch, cached)
    replay_error = max(float((normal[k] - replay[k]).abs().max()) for k in normal)
    decoded = decode_cached(model, normal["response_tokens"], condition, batch.control, metadata)
    decode_error = float((decoded["prediction"] - normal["prediction"]).abs().max())
    if max(replay_error, decode_error) > TOLERANCE:
        raise ValueError("path audit normal replay/cached decoder parity failed")
    blocked = model.encode_response(
        cached[ids],
        batch.control,
        condition,
        block_response_cls_to_gene=True,
        **metadata,
    )
    normal["score"] = decoded["score"]
    blocked_decoded = decode_cached(
        model, blocked["response_tokens"], condition, batch.control, metadata
    )
    blocked_error = float((blocked_decoded["prediction"] - blocked["prediction"]).abs().max())
    if blocked_error > TOLERANCE:
        raise ValueError("path audit blocked cached decoder parity failed")
    blocked["score"] = blocked_decoded["score"]
    _finite(normal)
    _finite(blocked)
    return {
        "F": {key: normal[key].detach().cpu() for key in FEATURES},
        "B": {key: blocked[key].detach().cpu() for key in FEATURES},
        "condition": condition.detach().cpu(),
        "control": batch.control.detach().cpu(),
        "metadata": {key: value.detach().cpu() for key, value in metadata.items()},
        "parity": {
            "normal_replay_max_abs": replay_error,
            "normal_decode_max_abs": decode_error,
            "blocked_decode_max_abs": blocked_error,
            "tolerance": TOLERANCE,
        },
    }


class PathAuditBank:
    """Reject identity mismatches and release each complete two-condition grid."""

    def __init__(self, panel, *, required_shape=None):
        if panel["panel_sha256"] != digest_json(panel["schedule"]):
            raise ValueError("frozen path-audit panel content changed")
        self.schedule = deepcopy(panel["schedule"])
        self.contexts = defaultdict(list)
        previous, closed = None, set()
        for group in self.schedule:
            context = tuple(group["context"])
            if len(context) != len(CONTEXT_KEYS):
                raise ValueError("complete context identity required")
            if context != previous:
                if context in closed:
                    raise ValueError("path-audit contexts must be contiguous")
                if previous is not None:
                    closed.add(previous)
                previous = context
            rows, condition = group["rows"], group["condition_id"]
            if len(group["queries"]) < 2 or len(set(group["queries"])) != len(group["queries"]):
                raise ValueError("unique ordered query genes required")
            if digest_json(group["queries"]) != panel["query_sha256"]:
                raise ValueError("frozen query order differs")
            if any(
                tuple(r[k] for k in CONTEXT_KEYS) != context or r["condition_id"] != condition
                for r in rows
            ):
                raise ValueError("panel record/context/condition identity differs")
            controls = [r["control_row_id"] for r in rows]
            if len(controls) < 2 or len(set(controls)) != len(controls):
                raise ValueError("unique controls required")
            self.contexts[context].append(group)
        for groups in self.contexts.values():
            if len(groups) != 2 or len({g["condition_id"] for g in groups}) != 2:
                raise ValueError("complete two-condition crossed grid required")
            if groups[0]["queries"] != groups[1]["queries"] or [
                r["control_row_id"] for r in groups[0]["rows"]
            ] != [r["control_row_id"] for r in groups[1]["rows"]]:
                raise ValueError("identical ordered query/control keys required")
        shape = (len(self.contexts), 2, len(self.schedule[0]["rows"])) if self.schedule else ()
        if not self.schedule or (required_shape is not None and shape != required_shape):
            raise ValueError("path audit panel shape differs from frozen scope")
        if required_shape is not None and any(
            len(g["rows"]) != required_shape[2] for g in self.schedule
        ):
            raise ValueError("path audit controls differ from frozen scope")
        self.next = 0
        self.pending = []
        self.rows = {name: [] for name in ("decomposition", "pairs", "interventions")}
        self.receipts = []

    def add(self, rt, batch, cached, group):
        if self.next >= len(self.schedule) or group != self.schedule[self.next]:
            raise ValueError("missing, reordered or duplicate path-audit group/row keys")
        if not self.pending:
            self.context_started = time.perf_counter()
        values = collect_group(rt, batch, cached, group)
        if self.pending:
            import torch

            if not torch.equal(self.pending[0]["control"], values["control"]):
                raise ValueError("shared control IDs have unequal ordered expression")
        self.pending.append(values)
        self.next += 1
        if len(self.pending) == 2:
            self._finish(rt, group)
            self.pending.clear()

    def _finish(self, rt, group):
        import torch

        values = self.pending
        context = dict(zip(CONTEXT_KEYS, group["context"], strict=True))
        groups = self.contexts[tuple(group["context"])]
        context.update(
            condition_ids=[g["condition_id"] for g in groups],
            ordered_control_ids=[r["control_row_id"] for r in groups[0]["rows"]],
            query_sha256=digest_json(group["queries"]),
        )
        swap = []
        fingerprints = [
            (_fingerprint(v["F"]["response_tokens"]), _fingerprint(v["condition"])) for v in values
        ]
        for i, v in enumerate(values):
            swap.append(
                decode_cached(
                    rt.objective.student,
                    v["F"]["response_tokens"].to(rt.device),
                    values[1 - i]["condition"].to(rt.device),
                    v["control"].to(rt.device),
                    {key: value.to(rt.device) for key, value in v["metadata"].items()},
                )
            )
        if fingerprints != [
            (_fingerprint(v["F"]["response_tokens"]), _fingerprint(v["condition"])) for v in values
        ]:
            raise ValueError("decoder intervention mutated cached encoder state")
        for mode in ("F", "B"):
            arrays = {
                key: np.stack([v[mode][key].float().numpy() for v in values]) for key in FEATURES
            }
            for key in ("control_cls", "response_cls"):
                norm = np.linalg.norm(arrays[key].astype(np.float64), axis=-1, keepdims=True)
                if (norm == 0).any():
                    # Whole unit-feature row is unavailable, never replace a zero direction.
                    self.rows["decomposition"].append(
                        {
                            **context,
                            "mode": mode,
                            "feature": key + "_unit",
                            "status": "zero_norm",
                            **dict.fromkeys(decomposition(np.zeros((2, 2, 1)))),
                            "fraction_unavailable_reason": "zero_vector_norm",
                        }
                    )
                else:
                    arrays[key + "_unit"] = arrays[key] / norm
            for key, x in arrays.items():
                base = {**context, "mode": mode, "feature": key}
                self.rows["decomposition"].append({**base, "status": "ok", **decomposition(x)})
                for stratum, pairs in (
                    (
                        "same_condition_different_control",
                        [
                            ((p, a), (p, b))
                            for p in range(2)
                            for a, b in itertools.combinations(range(x.shape[1]), 2)
                        ],
                    ),
                    (
                        "different_condition_same_control",
                        [((0, c), (1, c)) for c in range(x.shape[1])],
                    ),
                ):
                    self.rows["pairs"].append(
                        {
                            **base,
                            "stratum": stratum,
                            "pair_count": len(pairs),
                            **pair_statistics(
                                np.stack([x[a] for a, _ in pairs]),
                                np.stack([x[b] for _, b in pairs]),
                            ),
                        }
                    )
        for key in FEATURES:
            normal = np.stack([v["F"][key].float().numpy() for v in values])
            blocked = np.stack([v["B"][key].float().numpy() for v in values])
            self.rows["interventions"].append(
                {**context, "mode": "B_minus_F", "feature": key, **pair_statistics(blocked, normal)}
            )
            if key in ("score", "prediction"):
                donor = torch.stack([v[key].detach().float().cpu() for v in swap]).numpy()
                stats = pair_statistics(donor, normal)
                full_swap = pair_statistics(normal[::-1], normal)["rms"]
                self.rows["interventions"].append(
                    {
                        **context,
                        "mode": "D_minus_F",
                        "feature": key,
                        **stats,
                        "full_condition_swap_rms": full_swap,
                        "ratio_to_full_condition_swap": stats["rms"] / full_swap
                        if full_swap > TOLERANCE
                        else None,
                        "ratio_unavailable_reason": None
                        if full_swap > TOLERANCE
                        else "denominator_at_replay_floor",
                    }
                )
        self.receipts.append(
            {
                **context,
                "parity": [v["parity"] for v in values],
                "cached_tokens_and_encoder_condition_unchanged": True,
                "elapsed_seconds": time.perf_counter() - self.context_started,
                "cuda_allocated_bytes": (
                    torch.cuda.memory_allocated(rt.device) if rt.device.type == "cuda" else None
                ),
                "cuda_peak_allocated_since_process_start_bytes": (
                    torch.cuda.max_memory_allocated(rt.device) if rt.device.type == "cuda" else None
                ),
            }
        )

    def result(self):
        if self.pending or self.next != len(self.schedule):
            raise ValueError("incomplete path-audit population")
        summaries = []
        for kind, rows in self.rows.items():
            grouped = defaultdict(list)
            for row in rows:
                grouped[(row["mode"], row["feature"], row.get("stratum"))].append(row)
            for (mode, feature, stratum), observations in grouped.items():
                summary = dict(
                    kind=kind,
                    mode=mode,
                    feature=feature,
                    stratum=stratum,
                    contexts=len(observations),
                )
                for key in sorted(set().union(*(r.keys() for r in observations))):
                    if key.endswith(("energy", "fraction")) or key in (
                        "rms",
                        "cosine",
                        "ratio_to_full_condition_swap",
                        "full_condition_swap_rms",
                    ):
                        numbers = [
                            r[key] for r in observations if isinstance(r.get(key), (int, float))
                        ]
                        summary[key] = float(np.mean(numbers)) if numbers else None
                        summary[key + "_finite_contexts"] = len(numbers)
                summaries.append(summary)
        return {
            "schema": "representation-response-path-audit-1",
            "updates": 0,
            "truth_use": "none; no truth scoring or perturbed expression read",
            "prediction_parameterization": "softplus_absolute_expression",
            "delta_use": "omitted; prediction-minus-control arithmetic is not learned sensitivity",
            "forward_policy": "three sealed forwards per group: F, F replay, all-self-layer B",
            "decoder_intervention": "decoder e_p from other condition at the identical control",
            "matrix_retention": "one context; released after scalar aggregation; no matrix export",
            "aggregation": "full context grid; equal-weight context means; contexts are not seeds",
            "rms_definition": "sqrt(mean squared difference across input pairs and coordinates)",
            "cosine_definition": "mean valid last-axis vector cosine; aligned genes for tokens",
            "tolerance": TOLERANCE,
            "contexts": len(self.contexts),
            "input_rows": sum(len(g["rows"]) for g in self.schedule),
            "forward_row_evaluations": 3 * sum(len(g["rows"]) for g in self.schedule),
            **self.rows,
            "context_means": summaries,
            "receipts": self.receipts,
            "interpretation": (
                "eval path dependence only; edge deletion is not proof of training harm; "
                "decoder mismatch is not a biological counterfactual"
            ),
        }
