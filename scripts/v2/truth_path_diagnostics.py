"""Training-population truth scoring for the sealed evaluation-only F/B panel."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from cap40_posthoc import records_from_data
from cap40_posthoc_stats import digest_json
from response_path_diagnostics import CONTEXT_KEYS, PathAuditBank, collect_group


def population_errors(normal, blocked, control, truth, deg):
    """Reduce controls before scoring absolute expression against one population mean."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (normal, blocked, control)]
    target = np.asarray(truth, dtype=np.float64)
    if any(x.ndim != 2 or x.shape != arrays[0].shape for x in arrays):
        raise ValueError("aligned control-by-query predictions required")
    if target.shape != arrays[0].shape[1:] or not all(
        np.isfinite(x).all() for x in [*arrays, target]
    ):
        raise ValueError("finite aligned population truth required")
    f, b, c = [x.mean(0) for x in arrays]
    result = {}
    for label, indices in (("query", np.arange(len(target))), ("deg", np.asarray(deg, dtype=int))):
        if not len(indices):
            result.update(
                {
                    label + "_" + k: None
                    for k in (
                        "F_mse",
                        "B_mse",
                        "control_copy_mse",
                        "B_minus_F_mse",
                        "F_rmse",
                        "B_rmse",
                        "prediction_effect_rmse",
                        "truth_effect_rmse",
                        "F_effect_rmse",
                        "B_effect_rmse",
                    )
                }
            )
            continue
        if (
            len(set(indices.tolist())) != len(indices)
            or indices.min() < 0
            or indices.max() >= len(target)
        ):
            raise ValueError("DEG indices outside frozen query axis")
        mse = [float(np.square((x - target)[indices]).mean()) for x in (f, b, c)]
        result.update(
            {
                label + "_F_mse": mse[0],
                label + "_B_mse": mse[1],
                label + "_control_copy_mse": mse[2],
                label + "_B_minus_F_mse": mse[1] - mse[0],
                label + "_F_rmse": float(np.sqrt(mse[0])),
                label + "_B_rmse": float(np.sqrt(mse[1])),
                label + "_prediction_effect_rmse": float(
                    np.sqrt(np.square((b - f)[indices]).mean())
                ),
                label + "_truth_effect_rmse": float(
                    np.sqrt(np.square((target - c)[indices]).mean())
                ),
                label + "_F_effect_rmse": float(np.sqrt(np.square((f - c)[indices]).mean())),
                label + "_B_effect_rmse": float(np.sqrt(np.square((b - c)[indices]).mean())),
            }
        )
    return result


def train_deg(truth, controls, genes, condition, control_id):
    """Native v2 non-dropout/top20 helper, Scanpy absolute t-test ranking on TRAIN only."""
    if len(truth) < 2 or len(controls) < 2:
        return [], "insufficient_train_cells_for_t_test"
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    from gradpert.evaluation.state import _ranked_de_indices

    data = ad.AnnData(
        np.concatenate((truth, controls)),
        obs=pd.DataFrame(
            {"condition": pd.Categorical(["truth"] * len(truth) + ["control"] * len(controls))}
        ),
    )
    data.var_names = list(genes)
    sc.tl.rank_genes_groups(
        data,
        groupby="condition",
        groups=["truth"],
        reference="control",
        rankby_abs=True,
        n_genes=len(genes),
        method="t-test",
    )
    ranking = data.uns["rank_genes_groups"]
    scores = np.asarray(ranking["scores"]["truth"])
    if not np.isfinite(scores).all():
        return [], "nonfinite_train_t_test_scores"
    # Explicit deterministic tie policy; ranking never sees model outputs.
    ranked = [
        str(g) for _, g in sorted(zip(-np.abs(scores), ranking["names"]["truth"], strict=True))
    ]
    t, c = truth.mean(0), controls.mean(0)
    non_dropout = (t != 0) | ((t == 0) & (c == 0))
    result = _ranked_de_indices(
        ranked,
        dict(zip(genes, range(len(genes)), strict=True)),
        non_dropout,
        condition=condition,
        control_id=control_id,
        evaluation_protocol="v2",
    )
    return result, None if result else "no_eligible_train_deg"


class TruthPathBank:
    """Read only selected training populations; retain scalar receipts, never row IDs."""

    def __init__(self, panel, rt, *, required_shape=(16, 2, 4), query_count=1000):
        validator = PathAuditBank(panel, required_shape=required_shape)
        self.schedule = validator.schedule
        self.next, self.rows, self.receipts = 0, [], []
        self.panel_hash = panel["panel_sha256"]
        self.queries = self.schedule[0]["queries"]
        if len(self.queries) != query_count or not set(self.queries).issubset(
            set(map(int, rt.allowed_expression_ids))
        ):
            raise ValueError(
                "truth audit requires frozen query axis within allowed expression genes"
            )
        d = rt.data
        self.data = d
        self.records = records_from_data(d)
        self.populations = defaultdict(list)
        train = set(d.train_row_indices)
        seen = set()
        for r in self.records:
            i = d._row_index[r["truth_row_id"]]
            if i not in train or r["truth_row_id"] in seen:
                raise ValueError("truth record left unique cap40 training selection")
            seen.add(r["truth_row_id"])
            self._metadata(r, i)
            self.populations[(tuple(r[k] for k in CONTEXT_KEYS), r["condition_id"])].append(i)
        if seen != {d.row_ids[i] for i in train}:
            raise ValueError("truth records differ from complete cap40 membership")
        for indices in self.populations.values():
            if len(indices) > 40:
                raise ValueError("truth population exceeds sealed cap40 selection")
        self.control_pools = defaultdict(list)
        for i in d.control_row_indices:
            if str(d.condition_ids[i]) != d.split.control_condition_id:
                raise ValueError("non-control in training control pool")
            self.control_pools[self._context(i)].append(i)
        for group in self.schedule:
            context = tuple(group["context"])
            if group["queries"] != self.queries:
                raise ValueError("truth audit query order differs")
            for row in group["rows"]:
                ti, ci = d._row_index[row["truth_row_id"]], d._row_index[row["control_row_id"]]
                if ti not in train or ci not in self.control_pools[context]:
                    raise ValueError("panel left selected train truth/control membership")
                self._metadata(row, ti)
                if self._context(ci) != context:
                    raise ValueError("panel control context differs")

    def _context(self, i):
        d = self.data
        return (
            "Jurkat",
            str(d._adata.obs.iloc[i]["cell_type"]),
            str(d.context_ids[i]),
            str(d.batch_ids[i]),
        )

    def _metadata(self, row, i):
        if (
            self._context(i) != tuple(row[k] for k in CONTEXT_KEYS)
            or str(self.data.condition_ids[i]) != row["condition_id"]
        ):
            raise ValueError("truth metadata condition/context differs from runtime")

    def add(self, rt, batch, cached, group):
        if self.next >= len(self.schedule) or group != self.schedule[self.next]:
            raise ValueError("missing, reordered or duplicate truth-audit group")
        values = collect_group(rt, batch, cached, group)
        self.next += 1
        context = tuple(group["context"])
        indices = self.populations.get((context, group["condition_id"]), [])
        controls = self.control_pools[context]
        base = dict(zip(CONTEXT_KEYS, context, strict=True))
        base.update(
            condition_id=group["condition_id"],
            truth_count=len(indices),
            train_control_count=len(controls),
            crossed_controls=len(group["rows"]),
            query_count=len(self.queries),
            query_sha256=digest_json(self.queries),
            ordered_crossed_control_rows_sha256=digest_json(
                [r["control_row_id"] for r in group["rows"]]
            ),
            ordered_truth_rows_sha256=digest_json([self.data.row_ids[i] for i in indices]),
            ordered_train_control_rows_sha256=digest_json([self.data.row_ids[i] for i in controls]),
        )
        if not indices:
            self.rows.append({**base, "status": "missing_matched_train_truth"})
        else:
            truth = self.data._read_expression_indices(indices)[:, self.queries]
            control = self.data._read_expression_indices(controls)[:, self.queries]
            truth_mean = truth.mean(0, dtype=np.float64)
            base["query_population_mean_sha256"] = digest_json(truth_mean.tolist())
            base["query_train_control_pool_mean_sha256"] = digest_json(
                control.mean(0, dtype=np.float64).tolist()
            )
            genes = [self.data.expression_gene_ids[i] for i in self.queries]
            deg, reason = train_deg(
                truth, control, genes, group["condition_id"], self.data.split.control_condition_id
            )
            scores = population_errors(
                values["F"]["prediction"].float().numpy(),
                values["B"]["prediction"].float().numpy(),
                values["control"].float().numpy(),
                truth_mean,
                deg,
            )
            self.rows.append(
                {
                    **base,
                    "status": "ok",
                    "deg_count": len(deg),
                    "deg_unavailable_reason": reason,
                    "deg_query_positions_sha256": digest_json(deg),
                    **scores,
                }
            )
        self.receipts.append(
            {
                **dict(zip(CONTEXT_KEYS, context, strict=True)),
                "condition_id": group["condition_id"],
                "parity": values["parity"],
            }
        )

    def result(self):
        if self.next != len(self.schedule):
            raise ValueError("incomplete truth-audit population")
        contexts = defaultdict(list)
        for row in self.rows:
            contexts[tuple(row[k] for k in CONTEXT_KEYS)].append(row)
        summaries = []
        for context, rows in contexts.items():
            result = dict(zip(CONTEXT_KEYS, context, strict=True))
            for k in sorted({k for row in rows for k in row if k.endswith(("mse", "rmse"))}):
                numbers = [r[k] for r in rows if r.get(k) is not None]
                result[k] = float(np.mean(numbers)) if numbers else None
                result[k + "_valid_groups"] = len(numbers)
            summaries.append(result)
        return dict(
            schema="representation-truth-path-audit-1",
            updates=0,
            panel_sha256=self.panel_hash,
            query_count=len(self.queries),
            contexts=len(contexts),
            input_rows=sum(len(g["rows"]) for g in self.schedule),
            forward_row_evaluations=3 * sum(len(g["rows"]) for g in self.schedule),
            scoring=(
                "mean four absolute-expression predictions before population-mean MSE; "
                "no delta scoring"
            ),
            deg_policy=(
                "TRAIN-only matched context; native v2 top20/non-dropout; "
                "absolute Scanpy t-test; ties gene ID; nonfinite or n<2 NA"
            ),
            truth_scope=(
                "all selected cap40 TRAIN rows per condition/context; "
                "frozen allowed query axis only"
            ),
            interpretation=(
                "frozen eval edge deletion only; no observed per-control counterfactual "
                "and no training-quality inference"
            ),
            rows=self.rows,
            context_means=summaries,
            receipts=self.receipts,
        )
