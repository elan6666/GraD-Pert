"""Frozen train-only crossed control/condition panel for mechanism diagnosis.

This supplements the historical medium population; it never replaces it.
The rows are input interventions, not observed paired counterfactual truths.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace

from cap40_posthoc_stats import digest_json, seeded_int


def assemble_control_only(runtime, records, queries, seed, *, deep):
    """Use native assembly with real controls and no perturbed-expression read.

    The objective API expects a truth-shaped tensor; its zero placeholder may
    enter an unreported prediction-loss calculation, never a representation.
    """
    import numpy as np
    import torch

    from gradpert.training.batch import GraDPertTrainingBatch
    from gradpert.training.v2.views import assemble_batch

    data = runtime.data
    control = torch.from_numpy(
        data._read_expression_indices([data._row_index[r["control_row_id"]] for r in records])
    )
    conditions = tuple(r["condition_id"] for r in records)
    raw = GraDPertTrainingBatch(
        control_expression=control,
        target_expression=torch.zeros_like(control),
        condition_ids=conditions,
        anchors_by_condition={p: data.anchors_by_condition[p] for p in set(conditions)},
        perturbed_row_ids=tuple(r["truth_row_id"] for r in records),
        control_row_ids=tuple(r["control_row_id"] for r in records),
    )
    options = replace(runtime.options, lambda1=float(deep), lambda2=float(deep))
    return assemble_batch(
        raw,
        runtime.index,
        options,
        np.random.default_rng(seed),
        allowed_expression_ids=np.array(queries),
    )


class MatchedCellViews:
    """Freeze actual expression crops/masks within a crossed-input context.

    Graph sampling may consume a condition-dependent amount of RNG. Reusing its
    seed does not by itself match expression views. This diagnostic-only adapter
    retains the first native recipe and never alters training or graph views.
    """

    def __init__(self):
        self._contexts = {}

    @staticmethod
    def signature(views):
        return digest_json(
            [{"positions": v.positions.tolist(), "mask": v.mask.tolist()} for v in views]
        )

    def bind(self, batch, records):
        contexts = {
            tuple(r[k] for k in ("cell_line", "cell_type", "context_id", "batch")) for r in records
        }
        if len(contexts) != 1:
            raise ValueError("one compatible context per expression-view binding")
        context = contexts.pop()
        ids = batch.graph.ids[batch.query_positions].tolist()
        controls = [r["control_row_id"] for r in records]
        original = self.signature(batch.cell_views)
        if context not in self._contexts:
            views = tuple(
                replace(v, positions=v.positions.clone(), mask=v.mask.clone())
                for v in batch.cell_views
            )
            self._contexts[context] = (ids, controls, views, self.signature(views))
        saved_ids, saved_controls, views, fingerprint = self._contexts[context]
        if ids != saved_ids or controls != saved_controls:
            raise ValueError("crossed expression views require identical ordered genes/controls")
        if self.signature(views) != fingerprint:
            raise ValueError("frozen expression-view recipe was mutated")
        if len(views) != len(batch.cell_views):
            raise ValueError("expression-view count changed within context")
        return replace(batch, cell_views=views), {
            "context": list(context),
            "native_cell_views_sha256": original,
            "matched_cell_views_sha256": fingerprint,
            "query_sha256": digest_json(ids),
            "controls_sha256": digest_json(controls),
        }


def panel_schedule(parent, *, seed=20261021, conditions=32, controls=4):
    if parent.get("analysis_scope") != "medium_cap40_stratified":
        raise ValueError("a frozen medium parent is required")
    if conditions < 2 or controls < 2:
        raise ValueError("crossed panel needs at least two conditions and controls")
    grouped = defaultdict(lambda: defaultdict(list))
    for group in parent["schedule"]:
        for row in group["rows"]:
            if row["condition_id"] != group["condition_id"]:
                raise ValueError("parent row/condition mismatch")
            context = tuple(row[k] for k in ("cell_line", "cell_type", "context_id", "batch"))
            grouped[context][row["condition_id"]].append(row)
    eligible, excluded = {}, []
    for context, groups in sorted(grouped.items()):
        pool = sorted({r["control_row_id"] for rows in groups.values() for r in rows})
        if len(groups) < 2 or len(pool) < controls:
            excluded.append(
                {"context": list(context), "conditions": len(groups), "controls": len(pool)}
            )
            continue
        keys = sorted(groups, key=lambda p: (seeded_int([context, p], seed), p))
        pool.sort(key=lambda c: (seeded_int([context, c], seed), c))
        eligible[context] = (groups, keys, pool[:controls])
    # Add two conditions per context before any extra round-robin allocation.
    chosen = []
    for context in sorted(eligible, key=lambda c: (seeded_int(c, seed), c)):
        if len(chosen) + 2 <= conditions:
            chosen.extend((context, p) for p in eligible[context][1][:2])
    represented = list(dict.fromkeys(c for c, _ in chosen))
    for rank in range(2, max((len(eligible[c][1]) for c in represented), default=2)):
        for context in represented:
            keys = eligible[context][1]
            if rank < len(keys) and len(chosen) < conditions:
                chosen.append((context, keys[rank]))
    if not chosen:
        raise ValueError("no context has sufficient distinct controls and conditions")
    # One unchanged panel for every condition and checkpoint. Eligibility is
    # verified against the runtime's allowed-expression set before inference.
    queries = deepcopy(parent["schedule"][0]["queries"])
    if len(queries) != len(set(queries)) or len(queries) < 2:
        raise ValueError("invalid common query panel")
    schedule = []
    for index, (context, condition) in enumerate(chosen):
        groups, _, pool = eligible[context]
        representative = min(groups[condition], key=lambda r: r["truth_row_id"])
        records = [{**representative, "control_row_id": control} for control in pool]
        schedule.append(
            dict(
                index=index,
                condition_id=condition,
                context=list(context),
                rows=records,
                queries=queries,
                view_seed=seeded_int(["views", context], seed) % (2**32),
            )
        )
    return dict(
        schema="representation-crossed-panel-1",
        seed=seed,
        schedule=schedule,
        requested_conditions=conditions,
        condition_context_groups=len(schedule),
        conditions=len({g["condition_id"] for g in schedule}),
        controls_per_context=controls,
        rows=sum(len(g["rows"]) for g in schedule),
        excluded_contexts=excluded,
        panel_sha256=digest_json(schedule),
        query_sha256=digest_json(queries),
        statistical_unit="control_condition_input_intervention",
        truth_use="none; representative truth ID is assembly metadata only",
        selection="train metadata and fixed hash only; no outcome-based selection",
    )


def validate_panel_runtime(panel, runtime):
    """Reject test rows, hidden expression genes and incompatible controls."""
    if panel["panel_sha256"] != digest_json(panel["schedule"]):
        raise ValueError("panel content changed")
    data = runtime.data
    train = set(data.train_row_indices)
    controls = set(data.control_row_indices)
    allowed = set(map(int, runtime.allowed_expression_ids))
    for group in panel["schedule"]:
        if not set(group["queries"]).issubset(allowed):
            raise ValueError("panel includes excluded expression genes")
        if len({r["control_row_id"] for r in group["rows"]}) != len(group["rows"]):
            raise ValueError("repeated control in crossed group")
        for row in group["rows"]:
            ti, ci = data._row_index[row["truth_row_id"]], data._row_index[row["control_row_id"]]
            if ti not in train or ci not in controls:
                raise ValueError("panel left training/control population")
            if (
                str(data.context_ids[ti]) != row["context_id"]
                or str(data.context_ids[ci]) != row["context_id"]
            ):
                raise ValueError("panel control context mismatch")
            if str(data.condition_ids[ti]) != row["condition_id"]:
                raise ValueError("panel perturbation mismatch")
