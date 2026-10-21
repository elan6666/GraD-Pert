"""Read-only, context-matched prototype pair diagnostics using supplied scaled logits.

No temperature or center is inferred. Log probabilities are retained only in memory;
result() returns scalar aggregates, never logits or probability matrices. Repeated
condition/control rows select the first realization in manifest order.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

_METRICS = ("ce", "teacher_entropy", "kl")


def _log_softmax(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("finite row-by-prototype scaled logits required")
    shifted = x - x.max(axis=1, keepdims=True)
    result = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    if not np.isfinite(result).all():
        raise ValueError("scaled logit range exceeds float64 diagnostic precision")
    return result


def _summary(groups):
    """Reduce sums first within each ordered condition pair, then equally across pairs."""
    valid = [g for g in groups if g[0] > 0]
    count = sum(g[0] for g in valid)
    result = {
        "status": "ok" if count else "undefined:no_eligible_pairs",
        "pair_count": count,
        "valid_condition_pairs": len(valid),
    }
    for offset, label in ((1, ""), (4, "matched_correct_")):
        for index, metric in enumerate(_METRICS):
            result[label + metric + "_pair_mean"] = (
                float(sum(g[offset + index] for g in valid) / count) if count else None
            )
            result[label + metric + "_condition_balanced_mean"] = (
                float(np.mean([g[offset + index] / g[0] for g in valid])) if valid else None
            )
    for metric in _METRICS:
        for reduction in ("pair_mean", "condition_balanced_mean"):
            name = metric + "_" + reduction
            a, b = result[name], result["matched_correct_" + name]
            result["wrong_minus_correct_" + name] = a - b if a is not None else None
    return result


class PrototypePairBank:
    """Compare Teacher Global targets to Student cross-view distributions.

    Records require condition_id and context_id; control_condition also requires
    control_row_id. Optional cell_line/cell_type further isolate contexts. Teacher
    role accepts global* views only. Inputs must already include any centering and
    temperature scaling. Exact all-pair means are calculated through group sums,
    without materializing an N by N by K pair tensor or serializing arrays.
    """

    def __init__(self):
        self._banks = defaultdict(lambda: defaultdict(dict))
        self._widths = {}
        self._duplicates = defaultdict(int)

    def add(self, branch, role, view, records, scaled_logits, unit="control_condition"):
        if role not in ("student", "teacher"):
            raise ValueError("role must be student or teacher")
        if unit not in ("control_condition", "perturbation_condition"):
            raise ValueError("unsupported statistical unit")
        if not isinstance(view, str) or not view.startswith(("global", "local")):
            raise ValueError("view must start with global or local")
        if role == "teacher" and not view.startswith("global"):
            raise ValueError("Teacher accepts Global views only")
        rows = list(records)
        values = _log_softmax(scaled_logits)
        if len(rows) != len(values):
            raise ValueError("record/logit row counts differ")
        required = ("condition_id", "context_id") + (
            ("control_row_id",) if unit == "control_condition" else ()
        )
        for row in rows:
            if any(k not in row or row[k] is None or str(row[k]) == "" for k in required):
                raise ValueError("records need nonempty condition/context/control identities")
        width_key = (branch, unit)
        if width_key in self._widths and self._widths[width_key] != values.shape[1]:
            raise ValueError("prototype width differs within branch/unit")
        self._widths[width_key] = values.shape[1]
        for row, value in zip(rows, values, strict=True):
            context = tuple(str(row.get(k, "")) for k in ("cell_line", "cell_type", "context_id"))
            key = (branch, unit, *context)
            sample = (str(row["condition_id"]),) + (
                (str(row["control_row_id"]),) if unit == "control_condition" else ()
            )
            leaf = self._banks[key][role, view]
            if sample in leaf:
                self._duplicates[key] += 1
            else:
                leaf[sample] = value.copy()

    def result(self):
        output = []
        for key in sorted(self._banks):
            branch, unit, line, cell_type, context = key
            views = self._banks[key]
            teachers = sorted(v for r, v in views if r == "teacher")
            students = sorted(v for r, v in views if r == "student")
            for tv in teachers:
                for sv in students:
                    if tv == sv:
                        continue  # Exclude matching Global views, as in cross-view CLS loss.
                    tl, sl = views["teacher", tv], views["student", sv]
                    samples = sorted(tl.keys() & sl.keys())
                    base = dict(
                        branch=branch,
                        sample_unit=unit,
                        cell_line=line,
                        cell_type=cell_type,
                        context_id=context,
                        teacher_view=tv,
                        student_view=sv,
                        view_family="GG" if sv.startswith("global") else "GL",
                        independent_samples=len(samples),
                        independent_controls=len({s[1] for s in samples})
                        if unit == "control_condition"
                        else None,
                        conditions=len({s[0] for s in samples}),
                        teacher_only_samples=len(tl.keys() - sl.keys()),
                        student_only_samples=len(sl.keys() - tl.keys()),
                        duplicate_rows_ignored=self._duplicates[key],
                        prototype_count=self._widths[branch, unit],
                        weighting="ordered_condition_pair_equal",
                    )
                    strata = (
                        ("same_sample", "different_condition")
                        if unit == ("perturbation_condition")
                        else (
                            "same_sample",
                            "same_condition_different_control",
                            "different_condition_same_control",
                            "different_condition_different_control",
                        )
                    )
                    groups = {name: [] for name in strata}
                    if samples:
                        qt = np.exp(np.stack([tl[s] for s in samples]))
                        lt = np.stack([tl[s] for s in samples])
                        ls = np.stack([sl[s] for s in samples])
                        entropy = -np.einsum("ij,ij->i", qt, lt)
                        correct = -np.einsum("ij,ij->i", qt, ls)
                        by_condition = defaultdict(list)
                        for i, sample in enumerate(samples):
                            by_condition[sample[0]].append(i)

                        def selected(left, right, qt=qt, ls=ls, entropy=entropy, correct=correct):
                            """Sum explicitly paired rows, not their Cartesian product."""
                            if not left:
                                return np.zeros(7, dtype=np.float64)
                            ce = float(-np.einsum("ij,ij->", qt[left], ls[right]))
                            h = float(entropy[left].sum())
                            c = float(correct[left].sum())
                            return np.array([len(left), ce, h, ce - h, c, h, c - h])

                        for pa, left in by_condition.items():
                            groups["same_sample"].append(selected(left, left))
                            for pb, right in by_condition.items():
                                if unit == "perturbation_condition" and pa == pb:
                                    continue
                                ce = float(-qt[left].sum(0) @ ls[right].sum(0))
                                h = float(entropy[left].sum() * len(right))
                                c = float(correct[left].sum() * len(right))
                                total = np.array(
                                    [
                                        len(left) * len(right),
                                        ce,
                                        h,
                                        ce - h,
                                        c,
                                        h,
                                        c - h,
                                    ]
                                )
                                if unit == "perturbation_condition":
                                    groups["different_condition"].append(total)
                                    continue
                                control_right = {samples[j][1]: j for j in right}
                                same_left = [i for i in left if samples[i][1] in control_right]
                                same_right = [control_right[samples[i][1]] for i in same_left]
                                same = selected(same_left, same_right)
                                if pa == pb:
                                    groups["same_condition_different_control"].append(total - same)
                                else:
                                    groups["different_condition_same_control"].append(same)
                                    groups["different_condition_different_control"].append(
                                        total - same
                                    )
                    for stratum in strata:
                        result = _summary(groups[stratum])
                        result["pair_count"] = int(result["pair_count"])
                        if stratum == "same_sample":
                            # Correct pairs are the reference, not "wrong" pairs.
                            for name in list(result):
                                if name.startswith("wrong_minus_correct_"):
                                    result[name] = None
                        output.append({**base, "stratum": stratum, **result})
        return {
            "schema": "prototype-cross-view-context-pairs-1",
            "pairs": output,
            "limits": [
                "Inputs are caller-scaled logits; no temperature or center is inferred.",
                "Correct and mismatch means use the same eligible anchors and multiplicity.",
                "Condition pairs are ordered: Teacher-to-Student cross entropy is directional.",
                "Repeated condition/control/view rows retain the first manifest realization.",
                "All pairs stay within cell line, cell type and context; no cross-context pooling.",
                "Wrong-minus-correct is a descriptive gap, not accuracy or pair significance.",
                "Static perturbation units ignore repeated controls and expose no control strata.",
                "Common Teacher/Student identities ensure every mismatch has a correct reference.",
            ],
        }
