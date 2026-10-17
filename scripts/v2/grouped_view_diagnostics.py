"""Condition/context-stratified Global/Local feature comparisons, not loss targets.

Raw row features stay in the server worker/recovery state. Only aggregate tables
leave the server. Views and repeated controls never increase independent N.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np


def feature_statistics(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite row-by-feature matrix required")
    if len(x) < 2:
        return dict(
            rows=len(x),
            effective_rank=None,
            relative_variance=None,
            status="undefined:fewer_than_two_independent_rows",
        )
    centered = x - x.mean(0)
    gram = centered @ centered.T if len(x) < x.shape[1] else centered.T @ centered
    eig = np.linalg.eigvalsh((gram + gram.T) / (2 * len(x))).clip(0)
    trace, energy = float(eig.sum()), float(np.square(x).sum() / len(x))
    p = eig / trace if trace > max(energy, 1e-30) * 1e-12 else np.zeros_like(eig)
    p = p[p > 0]
    return dict(
        rows=len(x),
        width=x.shape[1],
        rank_ceiling=min(len(x) - 1, x.shape[1]),
        effective_rank=float(np.exp(-(p * np.log(p)).sum())) if len(p) else 0.0,
        relative_variance=trace / max(energy, 1e-30),
        status="ok",
    )


def _sample_key(row, unit):
    return tuple(
        str(row.get(k, ""))
        for k in (
            "cell_line",
            "cell_type",
            "context_id",
            "condition_id",
            *(("control_row_id",) if unit == "control_condition" else ()),
        )
    )


class GroupedViewBank:
    def __init__(self):
        self.entries = []

    def add(self, branch, role, view, records, values, *, unit="control_condition"):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or len(records) != len(x) or not np.isfinite(x).all():
            raise ValueError("view features must retain row/condition identity")
        self.entries.append(
            dict(branch=branch, role=role, view=view, unit=unit, records=records, values=x)
        )

    def result(self, *, reuse_pair_layout=True):
        banks = defaultdict(dict)
        for entry in self.entries:
            for row, value in zip(entry["records"], entry["values"], strict=True):
                context = tuple(
                    str(row.get(k, "")) for k in ("cell_line", "cell_type", "context_id")
                )
                key = (entry["branch"], entry["role"], entry["unit"], *context)
                views = banks[key]
                leaf = views.setdefault(entry["view"], {})
                sample = _sample_key(row, entry["unit"])
                # The frozen manifest order selects one augmentation realization
                # when a control repeats within a perturbation condition.
                leaf.setdefault(sample, (row, value))
        pairs, features = [], []
        for key in sorted(banks):
            branch, role, unit, line, cell_type, context = key
            views = banks[key]
            samples = sorted(set.intersection(*(set(v) for v in views.values())))
            if not samples:
                continue
            names = sorted(views)
            records = [views[names[0]][sample][0] for sample in samples]
            arrays = {v: np.stack([views[v][s][1] for s in samples]) for v in names}
            conditions = np.array([str(r["condition_id"]) for r in records])
            controls = np.array([str(r.get("control_row_id", "")) for r in records])
            base = dict(
                branch=branch,
                role=role,
                sample_unit=unit,
                cell_line=line,
                cell_type=cell_type,
                context_id=context,
                independent_samples=len(samples),
                independent_controls=len(set(controls) - {""}),
                conditions=len(set(conditions)),
            )
            globals_ = [v for v in names if v.startswith("global")]
            center = np.concatenate([arrays[v] for v in globals_ or names]).mean(0)
            for view, x in arrays.items():
                per_condition = [x[conditions == p] for p in sorted(set(conditions))]
                within = [feature_statistics(a) for a in per_condition if len(a) >= 2]
                features.append(
                    {
                        **base,
                        "view": view,
                        **feature_statistics(x),
                        "between_condition_mean_rank": feature_statistics(
                            np.stack([a.mean(0) for a in per_condition])
                        )["effective_rank"],
                        "within_condition_rank_mean": float(
                            np.mean([a["effective_rank"] for a in within])
                        )
                        if within
                        else None,
                        "within_condition_relative_variance_mean": float(
                            np.mean([a["relative_variance"] for a in within])
                        )
                        if within
                        else None,
                        "within_condition_valid_conditions": len(within),
                    }
                )
            n = len(samples)
            diagonal = np.eye(n, dtype=bool)
            upper = np.triu(np.ones((n, n), dtype=bool), 1)
            same_p = conditions[:, None] == conditions[None, :]
            same_c = controls[:, None] == controls[None, :]
            masks = {
                "same_sample": diagonal,
                "same_condition_different_control": upper & same_p & ~same_c,
                "different_condition_different_control": upper & ~same_p & ~same_c,
                "different_condition_same_control": upper & ~same_p & same_c,
            }
            if unit == "perturbation_condition":
                masks = {"same_sample": diagonal, "different_condition": upper & ~same_p}

            def layout(mask, conditions=conditions):
                i, j = np.nonzero(mask)
                groups = defaultdict(list)
                for k, (left, right) in enumerate(zip(i, j, strict=True)):
                    p = tuple(sorted((conditions[left], conditions[right])))
                    groups[p].append(k)
                return i, j, groups

            # Identity strata are identical across all view pairs. Build once,
            # retaining exact pair/group order and reduction arithmetic.
            layouts = (
                {name: layout(mask) for name, mask in masks.items()} if reuse_pair_layout else {}
            )
            for va, vb in combinations_with_replacement(names, 2):
                a, b = arrays[va], arrays[vb]
                an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
                ac, bc = a - center, b - center
                acn, bcn = np.linalg.norm(ac, axis=1), np.linalg.norm(bc, axis=1)
                raw_den, centered_den = an[:, None] * bn[None, :], acn[:, None] * bcn[None, :]
                dot = a @ b.T
                raw = np.divide(dot, raw_den, out=np.full((n, n), np.nan), where=raw_den > 1e-30)
                centered = np.divide(
                    ac @ bc.T, centered_den, out=np.full((n, n), np.nan), where=centered_den > 1e-30
                )
                mse = np.square(a).sum(1)[:, None] + np.square(b).sum(1)[None, :] - 2 * dot
                mse = mse.clip(0) / a.shape[1]
                family = (
                    "GG"
                    if va.startswith("global") and vb.startswith("global")
                    else ("LL" if va.startswith("local") and vb.startswith("local") else "GL")
                )
                for stratum, mask in masks.items():
                    if va == vb and stratum == "same_sample":
                        continue
                    i, j, groups = layouts[stratum] if reuse_pair_layout else layout(mask)
                    fields = {}
                    for name, matrix in (
                        ("cosine", raw),
                        ("centered_cosine", centered),
                        ("mean_squared_difference", mse),
                    ):
                        values = matrix[i, j]
                        finite = values[np.isfinite(values)]
                        gm = [
                            float(values[ix][np.isfinite(values[ix])].mean())
                            for ix in groups.values()
                            if np.isfinite(values[ix]).any()
                        ]
                        fields.update(
                            {
                                name + "_pair_mean": float(finite.mean()) if len(finite) else None,
                                name + "_condition_balanced_mean": float(np.mean(gm))
                                if gm
                                else None,
                                name + "_valid_pairs": len(finite),
                            }
                        )
                    pairs.append(
                        {
                            **base,
                            "view_a": va,
                            "view_b": vb,
                            "view_family": family,
                            "stratum": stratum,
                            "pair_count": len(i),
                            "condition_groups": len(groups),
                            "status": "ok" if len(i) else "undefined:no_eligible_pairs",
                            **fields,
                        }
                    )
        return dict(
            schema="condition-stratified-global-local-1",
            pairs=pairs,
            features=features,
            limits=[
                "Pairs/views are correlated; no independent-pair significance claim.",
                "Cell line/type/context are matched; cross-cell-line effects are not pooled.",
                "Centered cosine uses one shared Global centroid per branch/role/context.",
                "Same-control/different-perturbation is separate from different-control pairs.",
            ],
        )
