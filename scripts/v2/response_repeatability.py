"""Frozen V2 response repeatability: metadata-first, allowed-column-only, zero models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from gradpert.data import DatasetLayout
from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2 import response_grounding_metadata_census as census

B = 199
SALTS = dict(
    treated="gradpert-response-repeatability-v2-20261027-treated-primary",
    control="gradpert-response-repeatability-v2-20261027-controls-fourfold",
    null="gradpert-response-repeatability-v2-20261027-context-slot-null",
    within="gradpert-response-repeatability-v2-20261027-within-context-sensitivity",
)
DESIGN_SHA = "7760998018eb252ff13d0d9cd083ae8a3bdeb81696dd77f74816efcc44c3302e"
PROTOCOL_SHA = "aab0feacfe092fcf2813672fe5558e6f6f6786ed66bfd36760db7a6383719655"
CENSUS_SHA = "f1e5fffff87d0edea95c5dd814865c250e10b146addfb72b363d78c3fc9e70be"
SCALE_SHA = "89c06418c180a91d0eda26f936540c8709ed318facab2bef4652ca84bfb65cb8"
ALLOWED_SHA = "fd70e611f2403cc805b47dedf4608a23f0103c9e123ea320ef363db665d48150"
FORBIDDEN_SHA = "6d18ae5dad48d50d251d17a46cfbdf2b883d99bde8ad2234952fc9805c2c0fc7"
AMENDMENT_SHA = "390af566adb531f21300dfe698671b9904b2557bc727d82260c6459ad96371dc"
PARTITION_SHA = "c4e4500f816adce8d9357c2a1263c60a0bad5aa9a898df290dde239a9325b99a"
FOLDS = ("TA", "TB", "VA", "VB")


def _rank(rows: Sequence[int], ids: Sequence[str], fields: Sequence[Any]) -> list[int]:
    return sorted(rows, key=lambda i: (sha256_json([*fields, ids[i]]), i))


def fixed_halves(
    rows: Sequence[int], ids: Sequence[str], fields: Sequence[Any]
) -> tuple[list[int], list[int]]:
    ranked = _rank(rows, ids, fields)
    half = len(ranked) // 2
    return sorted(ranked[:half]), sorted(ranked[half:])


def make_plan(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """No expression access; preserve full condition axes, including all NA entries."""
    census_result = census.summarize_metadata(metadata)
    ids, conditions, contexts = (metadata[k] for k in ("row_ids", "condition_ids", "context_ids"))
    by_id = {r: i for i, r in enumerate(ids)}
    controls = {}
    for k, pool in metadata["control_pools"].items():
        ranked = _rank([by_id[r] for r in pool], ids, [SALTS["control"], k])
        controls[k] = {fold: sorted(ranked[j::4]) for j, fold in enumerate(FOLDS)}
    val_set = set(metadata["val_conditions"])
    split_rows = dict(
        selected_train=list(metadata["train_row_indices"]),
        validation=[i for i, p in enumerate(conditions) if p in val_set],
    )
    splits = {}
    for split, rows in split_rows.items():
        axis = metadata["train_conditions" if split == "selected_train" else "val_conditions"]
        members: dict[str, list[int]] = defaultdict(list)
        for i in rows:
            members[conditions[i]].append(i)
        entries = []
        for p in axis:
            r = members[p]
            ca, cb = fixed_halves(r, ids, [SALTS["treated"], split, p])
            counts = Counter(contexts[i] for i in r)
            a, b = Counter(contexts[i] for i in ca), Counter(contexts[i] for i in cb)
            bad = {k for k in counts if any(len(controls[k][f]) < 2 for f in FOLDS)}
            reasons = (["treated_n_lt4"] if len(r) < 4 else []) + (
                ["control_fold_lt2"] if bad else []
            )
            entries.append(
                dict(
                    condition_id=p,
                    rows=r,
                    A=ca,
                    B=cb,
                    eligible=not reasons,
                    reasons=reasons,
                    A_contexts=len(a),
                    B_contexts=len(b),
                    A_context_counts=dict(a),
                    B_context_counts=dict(b),
                    lost_context_mass=sum(counts[k] for k in bad) / len(r),
                    half_context_TV=(
                        sum(abs(a[k] / len(ca) - b[k] / len(cb)) for k in counts) / 2
                        if ca and cb
                        else None
                    ),
                    shared_context_original_mass=sum(n for k, n in counts.items() if a[k] and b[k])
                    / len(r),
                )
            )
        selected = [e for e in entries if e["eligible"]]
        null = slot_plan(selected, contexts, ids, split)
        splits[split] = dict(axis=entries, eligible=selected, null=null)
    within = []
    for p_index, entry in enumerate(splits["selected_train"]["eligible"]):
        groups: dict[str, list[int]] = defaultdict(list)
        for i in entry["rows"]:
            groups[contexts[i]].append(i)
        for k, rows in groups.items():
            if len(rows) >= 4 and all(len(controls[k][f]) >= 2 for f in FOLDS):
                a, b = fixed_halves(
                    rows, ids, [SALTS["within"], "selected_train", entry["condition_id"], k]
                )
                within.append(
                    dict(
                        condition_id=entry["condition_id"],
                        context_id=k,
                        main_index=p_index,
                        rows=rows,
                        A=a,
                        B=b,
                    )
                )
    used_contexts = {contexts[i] for s in splits.values() for e in s["eligible"] for i in e["rows"]}
    used_rows = sorted(
        {i for s in splits.values() for e in s["eligible"] for i in e["rows"]}
        | {i for k in used_contexts for f in FOLDS for i in controls[k][f]}
    )
    return dict(
        control_condition_id=metadata["control_condition_id"],
        splits=splits,
        controls=controls,
        within=within,
        used_rows=used_rows,
        census_summary=census_result["summary"],
    )


def slot_plan(
    entries: Sequence[Mapping[str, Any]], contexts: Sequence[str], ids: Sequence[str], split: str
) -> dict[str, Any]:
    """Fixed slots; duplicates are legal. Unique hashes describe labels, not row order."""
    slots: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for p, entry in enumerate(entries):
        for i in entry["B"]:  # canonical condition-axis, then canonical row order
            slots[contexts[i]].append((p, i))
    n = len(entries)
    q = np.zeros(n, dtype=np.float64)
    log_l = 0.0
    for members in slots.values():
        counts = Counter(p for p, _ in members)
        total = len(members)
        log_l += math.lgamma(total + 1) - sum(math.lgamma(c + 1) for c in counts.values())
        for p, count in counts.items():
            q[p] += count * (1 - count / total) / len(entries[p]["B"])
    source_rows = sorted(i for members in slots.values() for _, i in members)
    source_position = {i: j for j, i in enumerate(source_rows)}
    original_owner = {i: p for members in slots.values() for p, i in members}
    assignments, draws = [], []
    for draw in range(1, B + 1):
        owners = np.empty(len(source_rows), dtype=np.int32)
        for k, members in slots.items():
            permuted = _rank([i for _, i in members], ids, [SALTS["null"], split, draw, k])
            for (p, _), i in zip(members, permuted, strict=True):
                owners[source_position[i]] = p
        assignments.append(owners)
        changed = Counter(
            original_owner[i] for j, i in enumerate(source_rows) if owners[j] != original_owner[i]
        )
        draws.append(
            dict(
                draw=draw,
                label_assignment_sha256=sha256_json(owners.tolist()),
                actual_micro_mobility=sum(changed.values()) / len(source_rows)
                if source_rows
                else None,
                actual_macro_mobility=float(
                    np.mean([changed[p] / len(e["B"]) for p, e in enumerate(entries)])
                )
                if n
                else None,
            )
        )
    total_b = sum(len(e["B"]) for e in entries)
    macro = float(q.mean()) if n else None
    fraction = float((q >= 0.5).mean()) if n else None
    gate = n > 0 and macro >= 0.5 and fraction >= 0.5 and log_l >= math.log(20)
    return dict(
        source_rows=source_rows,
        owners=assignments,
        draws=draws,
        q_p=q,
        mobility=dict(
            micro=float(sum(q[p] * len(e["B"]) for p, e in enumerate(entries)) / total_b)
            if total_b
            else None,
            macro=macro,
            fraction_q_ge_half=fraction,
            log_label_assignment_count=log_l,
            q_min=float(q.min()) if n else None,
            q_max=float(q.max()) if n else None,
            q_p_histogram=np.histogram(q, bins=[0, 0.1, 0.25, 0.5, 0.75, 1.00000001])[0].tolist(),
            coverage_gate=bool(gate),
            unique_label_assignments=len({d["label_assignment_sha256"] for d in draws}),
        ),
    )


class AllowedReader:
    """Only one fancy axis: strict consecutive selected-row slices, allowed columns.

    Cache allowed columns once in the original dtype; FP64 conversion is blockwise.
    Physical HDF5 internal I/O bytes are not measured by this logical read boundary.
    """

    def __init__(
        self,
        adata: Any,
        rows: Sequence[int],
        columns: Sequence[int],
        *,
        row_block: int = 2048,
        expected_dtype: np.dtype | None = None,
    ):
        self.adata, self.rows, self.columns = adata, list(rows), list(columns)
        if (
            not self.rows
            or self.rows != sorted(set(self.rows))
            or not self.columns
            or self.columns != sorted(set(self.columns))
            or row_block < 1
        ):
            raise ValueError(
                "reader needs sorted unique authorized rows/columns and positive block"
            )
        self.row_block, self.expected_dtype = row_block, expected_dtype
        self.calls = self.logical_values = 0
        self.cache: np.ndarray | None = None

    def read(self) -> np.ndarray:
        if self.cache is not None:
            raise RuntimeError("allowed expression may only be cached once")
        start = 0
        while start < len(self.rows):
            stop = start + 1
            while (
                stop < len(self.rows)
                and self.rows[stop] == self.rows[stop - 1] + 1
                and stop - start < self.row_block
            ):
                stop += 1
            self.calls += 1
            selection = self.adata[
                slice(self.rows[start], self.rows[stop - 1] + 1),
                np.asarray(self.columns, dtype=np.int64),
            ]
            values = selection.X  # column projection precedes numerical array exposure
            values = values.toarray() if sparse.issparse(values) else np.asarray(values)
            self.logical_values += values.size
            if values.shape != (stop - start, len(self.columns)) or not np.isfinite(values).all():
                raise ValueError("allowed expression block has unexpected shape/nonfinite values")
            if self.expected_dtype is not None and values.dtype != self.expected_dtype:
                raise ValueError("canonical storage dtype differs before cache allocation")
            if self.cache is None:
                self.cache = np.empty((len(self.rows), len(self.columns)), dtype=values.dtype)
            if values.dtype != self.cache.dtype:
                raise ValueError("canonical numerical dtype changed across reads")
            self.cache[start:stop] = values
            start = stop
        assert self.cache is not None
        return self.cache


def loo(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < 2:
        raise ValueError("TRAIN center needs at least two eligible conditions")
    center = values.mean(axis=0, dtype=np.float64)
    return (len(values) / (len(values) - 1)) * (values - center), center


class Moments:
    """FP64 per-condition exact sums plus Chan centered covariance for Pearson."""

    def __init__(self, draws: int, conditions: int):
        self.n = np.zeros((draws, conditions), dtype=np.int64)
        self.sums = np.zeros((draws, conditions, 3), dtype=np.float64)  # cross, energy, diff
        self.means = np.zeros((draws, conditions, 2), dtype=np.float64)
        self.m2 = np.zeros((draws, conditions, 3), dtype=np.float64)
        self.minimum = np.full((draws, conditions, 2), np.inf)
        self.maximum = np.full((draws, conditions, 2), -np.inf)

    def add(self, draw: int, a: np.ndarray, b: np.ndarray, keep: np.ndarray | None = None) -> None:
        if (
            a.dtype != np.float64
            or b.dtype != np.float64
            or a.shape != b.shape
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError("finite FP64 paired arrays required")
        keep = np.ones(a.shape, dtype=bool) if keep is None else keep
        if keep.shape != a.shape or keep.dtype != bool:
            raise ValueError("scoring mask must be boolean and aligned")
        count = keep.sum(axis=1)
        # Exact constant detection uses scored values, not rounded means/M2.
        block_min = np.column_stack([np.min(x, axis=1, where=keep, initial=np.inf) for x in (a, b)])
        block_max = np.column_stack(
            [np.max(x, axis=1, where=keep, initial=-np.inf) for x in (a, b)]
        )
        self.minimum[draw] = np.minimum(self.minimum[draw], block_min)
        self.maximum[draw] = np.maximum(self.maximum[draw], block_max)
        aa, bb = np.where(keep, a, 0), np.where(keep, b, 0)
        self.sums[draw, :, 0] += (aa * bb).sum(axis=1, dtype=np.float64)
        self.sums[draw, :, 1] += ((aa * aa + bb * bb) / 2).sum(axis=1, dtype=np.float64)
        self.sums[draw, :, 2] += ((aa - bb) ** 2).sum(axis=1, dtype=np.float64)
        mean = np.column_stack(
            [
                np.divide(x.sum(axis=1), count, out=np.zeros(len(a)), where=count > 0)
                for x in (aa, bb)
            ]
        )
        da, db = np.where(keep, a - mean[:, 0, None], 0), np.where(keep, b - mean[:, 1, None], 0)
        centered = np.column_stack(
            [(da * da).sum(axis=1), (db * db).sum(axis=1), (da * db).sum(axis=1)]
        )
        old = self.n[draw].copy()
        total = old + count
        delta = mean - self.means[draw]
        weight = np.divide(old * count, total, out=np.zeros(len(a)), where=total > 0)
        self.m2[draw] += (
            centered
            + np.column_stack([delta[:, 0] ** 2, delta[:, 1] ** 2, delta[:, 0] * delta[:, 1]])
            * weight[:, None]
        )
        self.means[draw] += (
            delta * np.divide(count, total, out=np.zeros(len(a)), where=total > 0)[:, None]
        )
        self.n[draw] = total
        if not all(
            np.isfinite(v).all() for v in (self.sums[draw], self.means[draw], self.m2[draw])
        ):
            raise ValueError("nonfinite FP64 statistical accumulation")

    def finish(self, draw: int) -> dict[str, Any]:
        n = self.n[draw]
        valid = n > 0
        scalar = np.divide(
            self.sums[draw],
            n[:, None],
            out=np.full_like(self.sums[draw], np.nan),
            where=valid[:, None],
        )
        pearson_valid = (
            (n >= 2)
            & np.all(self.minimum[draw] < self.maximum[draw], axis=1)
            & (self.m2[draw, :, 0] > 0)
            & (self.m2[draw, :, 1] > 0)
        )
        # Avoid overflowing the product of two individually finite variances.
        pearson = np.divide(
            self.m2[draw, :, 2],
            np.sqrt(self.m2[draw, :, 0]),
            out=np.full(len(n), np.nan),
            where=pearson_valid,
        )
        np.divide(pearson, np.sqrt(self.m2[draw, :, 1]), out=pearson, where=pearson_valid)
        pearson = np.clip(pearson, -1, 1)
        aggregate = [float(scalar[valid, i].mean()) if valid.any() else None for i in range(3)]
        s, e, q = aggregate
        if s is not None and not math.isclose(
            q, 2 * e - 2 * s, rel_tol=1e-10, abs_tol=1e-12 * max(1, e)
        ):
            raise ArithmeticError("Q=2E-2S FP64 closure failed")
        finite = pearson[np.isfinite(pearson)]
        return dict(
            S=s,
            E=e,
            Q=q,
            ratio=s / e if e is not None and e > 0 else None,
            scored_conditions=int(valid.sum()),
            NA_conditions=int((~valid).sum()),
            Pearson_mean=float(finite.mean()) if len(finite) else None,
            Pearson_median=float(np.median(finite)) if len(finite) else None,
            Pearson_finite_conditions=len(finite),
            Pearson_NA_conditions=len(n) - len(finite),
            per_condition=dict(
                gene_count=n.tolist(),
                S=scalar[:, 0].tolist(),
                E=scalar[:, 1].tolist(),
                Q=scalar[:, 2].tolist(),
                Pearson=pearson.tolist(),
            ),
        )


def null_inference(
    observed: Mapping[str, Any], scores: Sequence[float | None], coverage: bool
) -> dict[str, Any]:
    if len(scores) != B:
        raise ValueError("V2 requires exactly199 frozen null draws")
    s, e = observed["S"], observed["E"]
    if s is None or any(x is None for x in scores):
        return dict(
            p=None,
            route_support=None,
            positive_rule=None,
            MC_borderline=None,
            reason="statistics_NA",
        )
    tolerance = 1e-10 * max(e, abs(s))
    count = sum(x >= s - tolerance for x in scores)
    tail = count / B
    z = 1.959963984540054
    denominator = 1 + z * z / B
    center = (tail + z * z / (2 * B)) / denominator
    half = z * math.sqrt(tail * (1 - tail) / B + z * z / (4 * B * B)) / denominator
    interval = [max(0.0, center - half), min(1.0, center + half)]
    borderline = interval[0] <= 0.05 <= interval[1]
    p = (1 + count) / (B + 1) if coverage else None
    positive = bool(s > 0 and p <= 0.05) if p is not None else None
    return dict(
        p=p,
        tail_count=count,
        tail_proportion=tail,
        tie_tolerance=tolerance,
        Wilson95=interval,
        MC_borderline=bool(borderline),
        positive_rule=positive,
        route_support=positive if coverage else None,
        reason=None if coverage else "metadata_mobility_coverage_failed",
    )


def _matrix(
    entries: Sequence[Mapping[str, Any]],
    half: str,
    positions: Mapping[int, int],
    context_index: Mapping[str, int],
    contexts: Sequence[str],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rr, cc, vv = [], [], []
    weights = np.zeros((len(entries), len(context_index)), dtype=np.float64)
    for p, entry in enumerate(entries):
        rows = entry[half]
        for i in rows:
            rr.append(p)
            cc.append(positions[i])
            vv.append(1 / len(rows))
            weights[p, context_index[contexts[i]]] += 1 / len(rows)
    return sparse.csr_matrix(
        (vv, (rr, cc)), shape=(len(entries), len(positions)), dtype=np.float64
    ), weights


def analyze_blocks(
    plan: Mapping[str, Any],
    contexts: Sequence[str],
    gene_ids: Sequence[str],
    reader: Callable[[Sequence[int]], np.ndarray],
    *,
    block_size: int = 512,
) -> dict[str, Any]:
    """Exact FP64 streaming; caller's reader exposes only ordered allowed genes."""
    if block_size < 1 or not gene_ids or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("positive block size and unique ordered allowed genes required")
    within_coverage = dict(
        groups=len(plan["within"]),
        conditions=len({g["main_index"] for g in plan["within"]}),
        contexts=len({g["context_id"] for g in plan["within"]}),
        treated_rows=sum(len(g["rows"]) for g in plan["within"]),
        independent_validation=False,
        full_train_axis_conditions=len(plan["splits"]["selected_train"]["axis"]),
        primary_eligible_conditions=len(plan["splits"]["selected_train"]["eligible"]),
        NA_full_axis_conditions=len(plan["splits"]["selected_train"]["axis"])
        - len({g["main_index"] for g in plan["within"]}),
    )
    train = plan["splits"]["selected_train"]["eligible"]
    if len(train) < 2:
        return dict(
            status="insufficient_eligible_TRAIN_for_center",
            splits={},
            within={},
            numeric_blocks=0,
            within_coverage=within_coverage,
            next_anchor_design_eligible=False,
        )
    positions = {i: j for j, i in enumerate(plan["used_rows"])}
    keys = sorted(
        {contexts[i] for s in plan["splits"].values() for e in s["eligible"] for i in e["rows"]}
    )
    k_index = {k: i for i, k in enumerate(keys)}
    matrices, banks, within_matrices = {}, {}, {}
    for split, p in plan["splits"].items():
        entries = p["eligible"]
        a, wa = _matrix(entries, "A", positions, k_index, contexts)
        b, wb = _matrix(entries, "B", positions, k_index, contexts)
        null_rows = np.array([positions[i] for i in p["null"]["source_rows"]], dtype=np.int64)
        nulls = [
            sparse.csr_matrix(
                (
                    np.array([1 / len(entries[int(owner)]["B"]) for owner in owners]),
                    (owners, null_rows),
                ),
                shape=b.shape,
            )
            for owners in p["null"]["owners"]
        ]
        matrices[split] = a, b, wa, wb, nulls
        banks[split] = {
            key: Moments(B + 1 if key != "raw" else 1, len(entries))
            for key in ("primary", "direct_target_excluded", "raw_direct_target_excluded", "raw")
        }
    group = plan["within"]
    if group:
        within_matrices = {h: _matrix(group, h, positions, k_index, contexts) for h in ("A", "B")}
        parents = sorted({g["main_index"] for g in group})
        sizes = Counter()
        for g in group:
            sizes[g["main_index"]] += len(g["rows"])
        pool = sparse.csr_matrix(
            (
                [len(g["rows"]) / sizes[g["main_index"]] for g in group],
                ([parents.index(g["main_index"]) for g in group], list(range(len(group)))),
            ),
            shape=(len(parents), len(group)),
        )
        within_banks = dict(group=Moments(1, len(group)), condition=Moments(1, len(parents)))
    for start in range(0, len(gene_ids), block_size):
        stop = min(start + block_size, len(gene_ids))
        x = np.asarray(reader(range(start, stop)), dtype=np.float64)
        if x.shape != (len(positions), stop - start) or not np.isfinite(x).all():
            raise ValueError("streamed allowed block is nonfinite/misaligned")
        control = {
            f: np.stack(
                [
                    x[[positions[i] for i in plan["controls"][k][f]]].mean(axis=0, dtype=np.float64)
                    for k in keys
                ]
            )
            for f in FOLDS
        }
        values = {}
        for split in ("selected_train", "validation"):
            a, b, wa, wb, _ = matrices[split]
            fa, fb = ("TA", "TB") if split == "selected_train" else ("VA", "VB")
            da, db = a @ x - wa @ control[fa], b @ x - wb @ control[fb]
            values[split] = da, db
        ta, tb = values["selected_train"]
        ra, ma = loo(ta)
        rb, mb = loo(tb)
        for split, p in plan["splits"].items():
            da, db = values[split]
            aa, bb = (ra, rb) if split == "selected_train" else (da - ma, db - mb)
            keep = np.array(
                [
                    [
                        g
                        not in (set(e["condition_id"].split("+")) - {plan["control_condition_id"]})
                        for g in gene_ids[start:stop]
                    ]
                    for e in p["eligible"]
                ],
                dtype=bool,
            ).reshape(len(p["eligible"]), stop - start)
            banks[split]["raw"].add(0, da, db)
            banks[split]["raw_direct_target_excluded"].add(0, da, db, keep)
            for key, mask in (("primary", None), ("direct_target_excluded", keep)):
                banks[split][key].add(0, aa, bb, mask)
            _, _, _, wb, nulls = matrices[split]
            ctrl = control["TB" if split == "selected_train" else "VB"]
            for draw, null in enumerate(nulls, 1):
                null_b = null @ x - wb @ ctrl
                null_r = loo(null_b)[0] if split == "selected_train" else null_b - mb
                banks[split]["raw_direct_target_excluded"].add(draw, da, null_b, keep)
                banks[split]["primary"].add(draw, aa, null_r)
                banks[split]["direct_target_excluded"].add(draw, aa, null_r, keep)
        if group:
            a, wa = within_matrices["A"]
            b, wb = within_matrices["B"]
            ga, gb = a @ x - wa @ control["TA"], b @ x - wb @ control["TB"]
            ia = np.array([g["main_index"] for g in group])
            ca, cb = ta - ra, tb - rb  # main TRAIN leave-p-out centers
            within_banks["group"].add(0, ga - ca[ia], gb - cb[ia])
            within_banks["condition"].add(0, pool @ ga - ca[parents], pool @ gb - cb[parents])
    result: dict[str, Any] = dict(
        status="completed_fixed_V2",
        splits={},
        within={},
        numeric_blocks=math.ceil(len(gene_ids) / block_size),
        within_coverage=within_coverage,
    )
    for split, bank in banks.items():
        result["splits"][split] = {}
        coverage = plan["splits"][split]["null"]["mobility"]["coverage_gate"]
        for key, moments in bank.items():
            observed = moments.finish(0)
            nulls = [moments.finish(d) for d in range(1, B + 1)] if key != "raw" else []
            inference = (
                null_inference(observed, [x["S"] for x in nulls], coverage) if nulls else None
            )
            result["splits"][split][key] = dict(observed=observed, nulls=nulls, inference=inference)
    if group:
        result["within"] = {k: v.finish(0) for k, v in within_banks.items()}
    primary = [
        result["splits"][s]["primary"]["inference"] for s in ("selected_train", "validation")
    ]
    direct = [
        result["splits"][s]["raw_direct_target_excluded"]["inference"]
        for s in ("selected_train", "validation")
    ]
    result["next_anchor_design_eligible"] = bool(
        all(x["positive_rule"] and not x["MC_borderline"] for x in primary + direct)
    )
    return result


def membership_receipts(
    plan: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    ids, contexts = metadata["row_ids"], metadata["context_ids"]
    private = dict(splits={}, controls={})
    coverage = {}
    for k, folds in plan["controls"].items():
        private["controls"][k] = {
            f: dict(rows=[ids[i] for i in rows], ordered_sha256=sha256_json([ids[i] for i in rows]))
            for f, rows in folds.items()
        }
    for split, p in plan["splits"].items():
        private["splits"][split] = []
        for e in p["axis"]:
            private["splits"][split].append(
                {
                    **{k: v for k, v in e.items() if k not in ("rows", "A", "B")},
                    **{
                        h: dict(
                            rows=[ids[i] for i in e[h]],
                            ordered_sha256=sha256_json([ids[i] for i in e[h]]),
                        )
                        for h in ("rows", "A", "B")
                    },
                }
            )
        entries = p["eligible"]
        slots = sorted(
            [e["condition_id"], k, n]
            for e in entries
            for k, n in Counter(contexts[i] for i in e["B"]).items()
        )
        rank_hashes = {}
        canonical_hashes = {}
        for h in ("A", "B"):
            rank_hashes[h] = sha256_json(
                [
                    [
                        e["condition_id"],
                        [
                            ids[i]
                            for i in _rank(e[h], ids, [SALTS["treated"], split, e["condition_id"]])
                        ],
                    ]
                    for e in entries
                ]
            )
            canonical_hashes[h] = sha256_json(
                [[e["condition_id"], [ids[i] for i in e[h]]] for e in entries]
            )
        coverage[split] = dict(
            full_axis_conditions=len(p["axis"]),
            eligible_conditions=len(entries),
            NA_conditions=len(p["axis"]) - len(entries),
            NA_reasons=dict(Counter(r for e in p["axis"] for r in e["reasons"])),
            eligible_rows=sum(len(e["rows"]) for e in entries),
            A_rows=sum(len(e["A"]) for e in entries),
            B_rows=sum(len(e["B"]) for e in entries),
            A_membership_hash=rank_hashes["A"],
            B_membership_hash=rank_hashes["B"],
            A_canonical_order_hash=canonical_hashes["A"],
            B_canonical_order_hash=canonical_hashes["B"],
            B_slot_count_hash=sha256_json(slots),
            null_B_source_canonical_membership_sha256=sha256_json(
                [ids[i] for i in p["null"]["source_rows"]]
            ),
            mean_half_context_TV=float(np.mean([e["half_context_TV"] for e in entries]))
            if entries
            else None,
            max_half_context_TV=max((e["half_context_TV"] for e in entries), default=None),
            mean_shared_context_original_mass=float(
                np.mean([e["shared_context_original_mass"] for e in entries])
            )
            if entries
            else None,
            **p["null"]["mobility"],
        )
    private["within_context"] = [
        {
            **{k: v for k, v in g.items() if k not in ("rows", "A", "B")},
            **{
                h: dict(
                    rows=[ids[i] for i in g[h]], ordered_sha256=sha256_json([ids[i] for i in g[h]])
                )
                for h in ("rows", "A", "B")
            },
        }
        for g in plan["within"]
    ]
    return private, coverage


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _public_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in score.items() if k != "per_condition"}


def publish_result(
    output: Path,
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
    coverage: Mapping[str, Any],
    receipt: Mapping[str, Any],
    closure: Callable[[], None],
) -> None:
    private = output / "server-only"
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("completed repeatability output is immutable")
    condition_rows, null_rows, private_null = [], [], []
    small = {k: v for k, v in result.items() if k not in ("splits", "within")}
    small["population_coverage"] = coverage
    small["splits"], small["within"] = (
        {},
        {k: _public_score(v) for k, v in result["within"].items()},
    )
    for key, score in small["within"].items():
        score["statistical_axis"] = (
            "condition_context_groups" if key == "group" else "restricted_condition_mixtures"
        )
        if key == "group":
            for old, new in (
                ("scored_conditions", "scored_groups"),
                ("NA_conditions", "NA_groups"),
                ("Pearson_finite_conditions", "Pearson_finite_groups"),
                ("Pearson_NA_conditions", "Pearson_NA_groups"),
            ):
                score[new] = score.pop(old)
    for split, p in plan["splits"].items():
        small["splits"][split] = {}
        banks = result["splits"].get(split, {})
        eligible_index = {e["condition_id"]: i for i, e in enumerate(p["eligible"])}
        for key in ("primary", "direct_target_excluded", "raw_direct_target_excluded", "raw"):
            bank = banks.get(key)
            if bank:
                small["splits"][split][key] = dict(
                    observed={
                        **_public_score(bank["observed"]),
                        "full_axis_conditions": len(p["axis"]),
                        "metadata_NA_conditions": len(p["axis"]) - len(p["eligible"]),
                        "total_axis_NA_conditions": len(p["axis"])
                        - bank["observed"]["scored_conditions"],
                    },
                    inference=bank["inference"],
                )
            for entry in p["axis"]:
                i = eligible_index.get(entry["condition_id"])
                scores = {
                    k: bank["observed"]["per_condition"][k][i] if bank and i is not None else None
                    for k in ("gene_count", "S", "E", "Q", "Pearson")
                }
                na_reason = ";".join(entry["reasons"])
                if not na_reason and not bank:
                    na_reason = "insufficient_TRAIN_for_center"
                if bank and i is not None and scores["gene_count"] == 0:
                    na_reason = "empty_scoring_gene_mask"
                pearson_na = (
                    "fewer_than_two_scored_genes"
                    if scores["gene_count"] is not None and scores["gene_count"] < 2
                    else "zero_gene_variance"
                    if scores["Pearson"] is not None and not math.isfinite(scores["Pearson"])
                    else na_reason
                )
                condition_rows.append(
                    dict(
                        split=split,
                        bank=key,
                        condition_id=entry["condition_id"],
                        treated_n=len(entry["rows"]),
                        expected_label_mobility_q_p=float(p["null"]["q_p"][i])
                        if i is not None
                        else None,
                        eligible=entry["eligible"],
                        NA_reason=na_reason,
                        Pearson_NA_reason=pearson_na,
                        **scores,
                    )
                )
            if key == "raw":
                continue
            score_hashes = []
            for d in range(B):
                score = bank["nulls"][d] if bank else dict(S=None, E=None, Q=None, ratio=None)
                null_rows.append(
                    dict(split=split, bank=key, **p["null"]["draws"][d], **_public_score(score))
                )
                score_hashes.append(sha256_json(_clean(_public_score(score))))
                if bank:
                    for e in p["axis"]:
                        i = eligible_index.get(e["condition_id"])
                        private_null.append(
                            dict(
                                split=split,
                                bank=key,
                                draw=d + 1,
                                condition_id=e["condition_id"],
                                **{
                                    k: score["per_condition"][k][i] if i is not None else None
                                    for k in ("gene_count", "S", "E", "Q", "Pearson")
                                },
                            )
                        )
            if bank:
                small["splits"][split][key]["unique_S_scores"] = len(
                    {score["S"] for score in bank["nulls"]}
                )
                small["splits"][split][key]["unique_scalar_tuples"] = len(set(score_hashes))
    atomic_json(output / "summary.json", _clean(small))
    census._csv(
        output / "coverage.csv",
        [
            dict(
                split=s,
                **{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in c.items()},
            )
            for s, c in coverage.items()
        ],
    )
    census._csv(output / "null_199_scalars.csv", _clean(null_rows))
    census._csv(private / "conditions.csv", _clean(condition_rows))
    if private_null:
        census._csv(private / "null_condition_scores.csv", _clean(private_null))
    if result["within"]:
        atomic_json(private / "within_context_scores.json", _clean(result["within"]))
        parents = sorted({g["main_index"] for g in plan["within"]})
        train = plan["splits"]["selected_train"]
        indices = {train["eligible"][p]["condition_id"]: i for i, p in enumerate(parents)}
        scores = result["within"]["condition"]["per_condition"]
        census._csv(
            private / "within_condition_scores.csv",
            _clean(
                [
                    dict(
                        condition_id=e["condition_id"],
                        NA_reason=""
                        if e["condition_id"] in indices
                        else "no_eligible_within_context_group_or_primary_NA",
                        **{
                            k: scores[k][indices[e["condition_id"]]]
                            if e["condition_id"] in indices
                            else None
                            for k in ("gene_count", "S", "E", "Q", "Pearson")
                        },
                    )
                    for e in train["axis"]
                ]
            ),
        )
    files = {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}
    closure()
    if any(output.rglob("*.pkl")):
        raise ValueError("unexpected PKL in fresh diagnostic output")
    atomic_json(
        output / "COMPLETE.json",
        _clean(
            dict(
                **receipt,
                output_sha256=files,
                closure_verified=True,
                zero_PKL_within_output=True,
                scientific_complete=False,
                transfer_allowlist=[
                    "COMPLETE.json",
                    "PRE_NUMERICAL_METADATA.json",
                    "summary.json",
                    "coverage.csv",
                    "null_199_scalars.csv",
                ],
                server_only=(
                    "all server-only/ membership/condition/draw records; "
                    "no raw expression persisted"
                ),
            )
        ),
    )


def _extra_source(
    binding: Mapping[str, Any], policy_function: Callable[..., Any]
) -> dict[str, Any]:
    identity = census._source(binding)
    root = Path(binding["root"])
    for relative, path in (
        ("scripts/v2/response_repeatability.py", Path(__file__)),
        (
            "src/gradpert/training/expression_policy.py",
            Path(__import__(policy_function.__module__, fromlist=["x"]).__file__),
        ),
    ):
        raw = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{identity['commit']}:{relative}"], timeout=30
        )
        if path.resolve() != root.resolve() / relative or hashlib.sha256(
            raw
        ).hexdigest() != sha256_file(path):
            raise ValueError("repeatability/policy executable differs from published source")
    return identity


def _claim_output(spec: Mapping[str, Any], original: Mapping[str, Any]) -> Path:
    output = census._server_path(spec["output"])
    protected = [
        census.DATA_ROOT,
        census.SELECTION.parent,
        Path(spec["analysis_source"]["root"]),
        Path(original["source"]["repository_root"]),
        Path(original["historical_training_source"]["repository_root"]),
        Path(original["input_seals"][0]["path"]).parent,
        Path(spec["census_complete"]["path"]).parent,
    ]
    protected += [
        Path(b["path"]).parent
        for b in original["input_seals"]
        if Path(b["path"]).name in ("run_manifest.json", "resolved_config.json")
    ]
    if any(output.resolve().is_relative_to(p.resolve()) for p in protected):
        raise ValueError(
            "output must be independent of canonical/source/historical/census directories"
        )
    output.mkdir(parents=True, exist_ok=False)
    return output


def run(spec_path: Path, expected_sha: str) -> None:
    if sha256_file(spec_path) != expected_sha:
        raise ValueError("spec hash differs")
    spec = json.loads(spec_path.read_text())
    if (
        set(spec)
        != {
            "schema",
            "analysis_source",
            "design",
            "protocol",
            "census_complete",
            "preprocessing",
            "partition_boundary",
            "amendment",
            "gene_block_size",
            "row_block_size",
            "cpu_budget",
            "output",
        }
        or spec["schema"] != "gradpert-response-repeatability-execution-v2"
    ):
        raise ValueError("complete frozen V2 execution spec required")
    if spec["cpu_budget"] != {"threads": 1, "max_allowed_cache_bytes": 4 * 1024**3} or any(
        type(v) is not int for v in spec["cpu_budget"].values()
    ):
        raise ValueError("sealed one-thread/4GiB allowed-cache CPU budget required")
    if (spec["gene_block_size"], spec["row_block_size"]) != (512, 2048):
        raise ValueError("fixed V2 execution blocks must be sealed512/2048")
    if spec["census_complete"]["sha256"] != CENSUS_SHA:
        raise ValueError("exact accepted census required before claiming output")
    original = json.loads(census._checked(spec["census_complete"]).read_text())
    output = _claim_output(spec, original)
    audit = census.new_audit()
    cache_reader: AllowedReader | None = None
    data = object.__new__(census.CanonicalTrainingData)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU-only requires CUDA_VISIBLE_DEVICES='' before process start")
        if any(
            os.environ.get(k) != "1"
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        ):
            raise ValueError("CPU thread environment must be set before process startup")
        from gradpert.training.expression_policy import expression_policy

        source = _extra_source(spec["analysis_source"], expression_policy)
        seals = [
            dict(path=str(spec_path), sha256=expected_sha),
            spec["analysis_source"]["publication"],
        ]
        for name, digest in (
            ("design", DESIGN_SHA),
            ("protocol", PROTOCOL_SHA),
            ("census_complete", CENSUS_SHA),
            ("preprocessing", SCALE_SHA),
            ("amendment", AMENDMENT_SHA),
            ("partition_boundary", PARTITION_SHA),
        ):
            if spec[name]["sha256"] != digest:
                raise ValueError("frozen design/census/preprocessing binding differs")
        seals += [
            spec[n]
            for n in (
                "design",
                "protocol",
                "census_complete",
                "preprocessing",
                "partition_boundary",
                "amendment",
            )
        ]
        for seal in seals:
            census._checked(seal)
        original = json.loads(Path(spec["census_complete"]["path"]).read_text())
        seals += original["input_seals"]
        for seal in original["input_seals"]:
            census._checked(seal)
        old_spec = json.loads(Path(original["input_seals"][0]["path"]).read_text())
        old_summary_path = Path(spec["census_complete"]["path"]).parent / "summary.json"
        seals.append(
            dict(path=str(old_summary_path), sha256=original["outputs_sha256"]["summary.json"])
        )
        census._checked(seals[-1])
        old_summary = json.loads(old_summary_path.read_text())
        historical = json.loads(Path(old_spec["historical_run_manifest"]["path"]).read_text())
        preprocessing = json.loads(Path(spec["preprocessing"]["path"]).read_text())
        if preprocessing["report"]["expression_scale_action"] != "normalize_total_4000_then_log1p":
            raise ValueError("canonical scale evidence differs; no normalization fallback")
        layout = DatasetLayout(census.DATA_ROOT, census.DATASET, census.PROTOCOL)
        canonical = json.loads(layout.canonical_manifest.read_text())
        if canonical["preprocessing_manifest_sha256"] != SCALE_SHA:
            raise ValueError("preprocessing manifest linkage differs")
        h5ad_stat = census._file_stat(layout.canonical_adata)
        closed = False
        with census.metadata_only_guard(census.CanonicalTrainingData, audit):
            try:
                census.CanonicalTrainingData.__init__(
                    data,
                    dataset_id=census.DATASET,
                    protocol_id=census.PROTOCOL,
                    data_root=census.DATA_ROOT,
                    run_seed=1,
                )
                selected = census.apply_train_selection(
                    data, census.SELECTION, census.SELECTION_SHA
                )
                if selected != original["selection_receipt"]:
                    raise ValueError("native selected membership changed")
                metadata = dict(
                    row_ids=data.row_ids,
                    condition_ids=data.condition_ids,
                    context_ids=data.context_ids,
                    control_pools=data.control_pools,
                    train_row_indices=data.train_row_indices,
                    train_conditions=data.split.train_conditions,
                    val_conditions=data.split.val_conditions,
                    control_condition_id=data.split.control_condition_id,
                )
                plan = make_plan(metadata)
                if plan["census_summary"] != old_summary:
                    raise ValueError("fresh complete-axis census differs")
                allowed, policy = expression_policy(
                    tuple(data.expression_gene_ids),
                    tuple(data.split.test_conditions),
                    data.split.control_condition_id,
                    enabled=True,
                )
                if (
                    allowed is None
                    or len(allowed) != 4775
                    or len(policy["excluded_expression_gene_ids"]) != 225
                    or policy != historical["data"]["training_expression_policy"]
                    or policy["allowed_expression_gene_ids_sha256"] != ALLOWED_SHA
                    or policy["excluded_expression_gene_ids_sha256"] != FORBIDDEN_SHA
                ):
                    raise ValueError("native4775/225 expression policy differs from historical run")
                private, coverage = membership_receipts(plan, metadata)
                expected = json.loads(Path(spec["partition_boundary"]["path"]).read_text())[
                    "splits"
                ]
                for split, actual in coverage.items():
                    for key in (
                        "A_membership_hash",
                        "B_membership_hash",
                        "A_rows",
                        "B_rows",
                        "B_slot_count_hash",
                        "eligible_conditions",
                    ):
                        if actual[key] != expected[split][key]:
                            raise ValueError("independent partition/slot replay differs")
                    for our, other in (
                        ("macro", "macro_q"),
                        ("micro", "micro_q"),
                        ("log_label_assignment_count", "log_label_assignment_count"),
                    ):
                        if not math.isclose(
                            actual[our], expected[split][other], rel_tol=1e-12, abs_tol=1e-10
                        ):
                            raise ValueError("independent mobility replay differs")
                private_dir = output / "server-only"
                private_dir.mkdir()
                atomic_json(private_dir / "memberships.json", _clean(private))
                atomic_json(private_dir / "gene_policy.json", policy)
                environment = dict(
                    python=platform.python_version(),
                    packages={
                        name: importlib.metadata.version(name)
                        for name in ("numpy", "scipy", "anndata", "torch")
                    },
                )
                if environment["packages"]["anndata"] != "0.13.2":
                    raise ValueError(
                        "installed AnnData differs from the reviewed slice/fancy boundary"
                    )
                pre = dict(
                    environment=environment,
                    schema="response-repeatability-pre-numerical-v2",
                    source=source,
                    spec_sha256=expected_sha,
                    coverage=coverage,
                    membership_bank_sha256=sha256_file(private_dir / "memberships.json"),
                    metadata_input_sha256=sha256_json(metadata),
                    gene_policy_bank_sha256=sha256_file(private_dir / "gene_policy.json"),
                    ordered_used_row_hash=sha256_json([data.row_ids[i] for i in plan["used_rows"]]),
                    allowed_gene_count=4775,
                    forbidden_gene_count=225,
                    allowed_gene_hash=ALLOWED_SHA,
                    forbidden_gene_hash=FORBIDDEN_SHA,
                    salts=SALTS,
                    B=B,
                    scale_action="existing_canonical_normalize_total_4000_then_log1p; no refit",
                    logical_cache_bytes_float32=len(plan["used_rows"]) * 4775 * 4,
                    cpu_budget=spec["cpu_budget"],
                    numerical_read_calls=0,
                    audit_counts=audit.copy(),
                    scientific_complete=False,
                )
                atomic_json(output / "PRE_NUMERICAL_METADATA.json", _clean(pre))
                if (
                    pre["logical_cache_bytes_float32"]
                    > spec["cpu_budget"]["max_allowed_cache_bytes"]
                ):
                    raise ValueError("metadata cache estimate exceeds frozen CPU budget; no reads")
                cache_reader = (
                    AllowedReader(
                        data._adata,
                        plan["used_rows"],
                        allowed.tolist(),
                        row_block=spec["row_block_size"],
                        expected_dtype=np.dtype("float32"),
                    )
                    if len(plan["splits"]["selected_train"]["eligible"]) >= 2
                    else None
                )
                cache = cache_reader.read() if cache_reader is not None else None
                if cache is not None and cache.dtype != np.float32:
                    raise ValueError("canonical storage must preserve declared float32 scale")
                data.close()
                closed = True
                genes = [data.expression_gene_ids[int(i)] for i in allowed]
                result = analyze_blocks(
                    plan,
                    data.context_ids,
                    genes,
                    lambda cols: cache[:, list(cols)],
                    block_size=spec["gene_block_size"],
                )

                def closure() -> None:
                    for seal in seals:
                        census._checked(seal)
                    if (
                        source != _extra_source(spec["analysis_source"], expression_policy)
                        or census._file_stat(layout.canonical_adata) != h5ad_stat
                        or any(audit.values())
                    ):
                        raise ValueError("source/input/stat/forbidden-call closure failed")
                    if (
                        sha256_file(private_dir / "memberships.json")
                        != pre["membership_bank_sha256"]
                        or sha256_file(private_dir / "gene_policy.json")
                        != pre["gene_policy_bank_sha256"]
                        or sha256_json(metadata) != pre["metadata_input_sha256"]
                    ):
                        raise ValueError("metadata/policy/membership bank closure failed")
                    if (
                        sha256_json(list(data.row_ids)) != census.OBS_ORDER_SHA
                        or sha256_json([data.row_ids[i] for i in data.train_row_indices])
                        != census.SELECTED_IDS_SHA
                    ):
                        raise ValueError("native metadata membership closure failed")

                receipt = dict(
                    schema="response-repeatability-complete-v2",
                    environment=environment,
                    source=source,
                    spec_sha256=expected_sha,
                    input_seals=seals,
                    preprocessing_manifest_sha256=SCALE_SHA,
                    pre_numerical_sha256=sha256_file(output / "PRE_NUMERICAL_METADATA.json"),
                    audit_counts=audit,
                    allowed_numerical_read_calls=cache_reader.calls if cache_reader else 0,
                    logical_allowed_values=cache_reader.logical_values if cache_reader else 0,
                    forbidden_full_width_loader_calls=audit["expression_loader_calls"],
                    physical_HDF5_internal_bytes="unmeasured",
                    no_normalization_or_model_update=True,
                    cpu_budget=spec["cpu_budget"],
                    B=B,
                    salts=SALTS,
                )
                publish_result(output, result, plan, coverage, receipt, closure)
            finally:
                if not closed:
                    data.close()
    except Exception as error:
        private = output / "server-only"
        private.mkdir(exist_ok=True)
        atomic_json(
            private / "FAILURE_DETAILS.json",
            dict(error_type=type(error).__name__, error=str(error)),
        )
        atomic_json(
            output / "FAILURE.json",
            dict(
                status="failed",
                error_type=type(error).__name__,
                details="server-only",
                allowed_numerical_read_calls=cache_reader.calls if cache_reader else 0,
                logical_allowed_values=cache_reader.logical_values if cache_reader else 0,
                audit_counts=audit,
                scientific_complete=False,
            ),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    run(args.spec, args.spec_sha256)


if __name__ == "__main__":
    main()
