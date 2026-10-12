"""Freeze the same cap40 rows and remove held-out expression before upstream access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gradpert.hashing import sha256_json
from gradpert.training.data import CanonicalTrainingData
from gradpert.training.expression_policy import expression_policy
from gradpert.training.v2.train_selection import apply_train_selection

from .contract import PreparedData


def read_allowed_rows(adata, rows, allowed, expression_count: int, chunk_rows: int = 1024):
    """Bounded one-array backed IO; only allowed values leave the reader.

    Older AnnData/h5py cannot apply two fancy index vectors. Read selected
    rows on the contiguous expression prefix, then discard excluded columns
    before retaining a block. No test rows, graph-only expression, numeric
    statistics or excluded values enter the prepared upstream object.
    """
    from scipy import sparse

    blocks = []
    for offset in range(0, len(rows), chunk_rows):
        block = adata[np.asarray(rows[offset : offset + chunk_rows]), :expression_count].X
        block = block[:, allowed] if sparse.issparse(block) else np.asarray(block)[:, allowed]
        blocks.append(sparse.csr_matrix(block, dtype=np.float32))
    return sparse.vstack(blocks, format="csr")


def prepare_data(config: dict[str, Any], data_root: Path) -> PreparedData:
    """Only a bounded server invocation may materialize canonical expression."""
    if not data_root.resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("scientific data preparation is server-only")
    anchor = config["data"]
    with CanonicalTrainingData(
        dataset_id=anchor["dataset_id"],
        protocol_id=anchor["protocol_id"],
        data_root=data_root,
        run_seed=config["seed"],
    ) as data:
        data.require_experiment_data_contract(
            registry_version=anchor["registry_version"],
            split_policy=anchor["split_policy"],
        )
        selection = apply_train_selection(
            data,
            Path(anchor["train_selection_path"]),
            anchor["train_selection_sha256"],
        )
        allowed, policy = expression_policy(
            data.expression_gene_ids,
            tuple(data.split.test_conditions),
            data.split.control_condition_id,
            enabled=True,
        )
        assert allowed is not None
        omitted = set(anchor.get("omit_condition_ids", ()))
        known = set(data.split.train_conditions)
        if omitted - known or data.split.control_condition_id in omitted:
            raise ValueError("authorized prior exclusions must name canonical training conditions")
        train = tuple(i for i in data.train_row_indices if data.condition_ids[i] not in omitted)
        val_set = set(data.split.val_conditions) - omitted
        val = tuple(i for i, c in enumerate(data.condition_ids) if c in val_set)
        targets = train + val
        contexts = {data.context_ids[i] for i in targets}
        controls = tuple(
            i
            for i, c in enumerate(data.condition_ids)
            if c == data.split.control_condition_id and data.context_ids[i] in contexts
        )
        selected = tuple(sorted((*train, *val, *controls)))
        # Only allowed values are retained across the fitter boundary. The
        # bounded reader discards excluded columns before assembling any matrix.
        matrix = read_allowed_rows(data._adata, selected, allowed, len(data.expression_gene_ids))
        return build_prepared(
            matrix=matrix,
            allowed_indices=tuple(map(int, allowed)),
            gene_ids=data.expression_gene_ids,
            graph_gene_ids=data.graph_gene_ids,
            row_ids=tuple(data.row_ids[i] for i in selected),
            conditions=tuple(data.condition_ids[i] for i in selected),
            contexts=tuple(data.context_ids[i] for i in selected),
            train_row_ids=tuple(data.row_ids[i] for i in train),
            val_row_ids=tuple(data.row_ids[i] for i in val),
            control_row_ids=tuple(data.row_ids[i] for i in controls),
            train_conditions=tuple(c for c in data.split.train_conditions if c not in omitted),
            val_conditions=tuple(c for c in data.split.val_conditions if c not in omitted),
            control_condition=data.split.control_condition_id,
            receipt={
                "canonical_data_sha256": data.manifest.canonical_adata_sha256,
                "split_content_sha256": data.split.split_content_sha256,
                "expression_gene_order_sha256": data.manifest.expression_gene_order_sha256,
                "graph_gene_order_sha256": data.manifest.graph_gene_order_sha256,
                "train_selection": selection,
                "expression_policy": policy,
                "test_condition_ids_sha256": sha256_json(list(data.split.test_conditions)),
                "test_truth_present_during_fit": False,
                "authorized_omitted_conditions": sorted(omitted),
                "omitted_train_row_count": len(data.train_row_indices) - len(train),
            },
        )


def build_prepared(
    *,
    matrix: Any,
    allowed_indices: tuple[int, ...],
    gene_ids: tuple[str, ...],
    graph_gene_ids: tuple[str, ...],
    row_ids: tuple[str, ...],
    conditions: tuple[str, ...],
    contexts: tuple[str, ...],
    train_row_ids: tuple[str, ...],
    val_row_ids: tuple[str, ...],
    control_row_ids: tuple[str, ...],
    train_conditions: tuple[str, ...],
    val_conditions: tuple[str, ...],
    control_condition: str = "ctrl",
    receipt: dict[str, Any] | None = None,
) -> PreparedData:
    """Construct from an already restricted matrix; also usable with synthetic tests."""
    import anndata
    import pandas as pd
    from scipy import sparse

    n = len(gene_ids)
    if (
        not gene_ids
        or len(set(gene_ids)) != n
        or len(set(row_ids)) != len(row_ids)
        or graph_gene_ids[:n] != gene_ids
        or not allowed_indices
        or tuple(sorted(set(allowed_indices))) != allowed_indices
        or any(i < 0 or i >= n for i in allowed_indices)
        or matrix.shape != (len(row_ids), len(allowed_indices))
        or not len(row_ids) == len(conditions) == len(contexts)
    ):
        raise ValueError("invalid sanitized row/gene axis")
    train_set, val_set, ctrl_set = map(set, (train_row_ids, val_row_ids, control_row_ids))
    if (
        train_set & val_set
        or train_set & ctrl_set
        or val_set & ctrl_set
        or train_set | val_set | ctrl_set != set(row_ids)
    ):
        raise ValueError("rows must partition exact train/validation/control populations")
    indices = {row: i for i, row in enumerate(row_ids)}
    if (
        {conditions[indices[r]] for r in train_row_ids} != set(train_conditions)
        or {conditions[indices[r]] for r in val_row_ids} != set(val_conditions)
        or any(conditions[indices[r]] != control_condition for r in control_row_ids)
    ):
        raise ValueError("row assignments differ from condition split")
    values = sparse.coo_matrix(matrix, dtype=np.float32)
    if not np.isfinite(values.data).all():
        raise ValueError("nonfinite allowed expression")
    safe = sparse.csr_matrix(
        (values.data, (values.row, np.asarray(allowed_indices)[values.col])),
        shape=(len(row_ids), n),
        dtype=np.float32,
    )
    excluded = tuple(i for i in range(n) if i not in set(allowed_indices))
    pools: dict[str, list[int]] = {}
    for row in control_row_ids:
        i = indices[row]
        pools.setdefault(contexts[i], []).append(i)

    def candidates(rows: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
        result = {row: tuple(pools.get(contexts[indices[row]], ())) for row in rows}
        if any(not pool for pool in result.values()):
            raise ValueError("target row lacks context-compatible control")
        return result

    obs = pd.DataFrame(index=pd.Index(row_ids))
    obs["condition"] = list(conditions)
    obs["cell_type"] = [c.split("::", 1)[0] for c in contexts]
    obs["batch"] = [c.split("::", 1)[1] if "::" in c else "" for c in contexts]
    obs["control"] = [c == control_condition for c in conditions]
    obs["dose_val"] = [
        "1+1" if "+" in c and "ctrl" not in c.split("+") else "1" for c in conditions
    ]
    obs["condition_name"] = [
        f"{ct}_{p}_{d}" for ct, p, d in zip(obs.cell_type, obs.condition, obs.dose_val, strict=True)
    ]
    var = pd.DataFrame({"gene_name": list(gene_ids)}, index=pd.Index(gene_ids))
    adata = anndata.AnnData(X=safe, obs=obs, var=var)
    evidence = {
        **(receipt or {}),
        "schema": "official-expression-holdout-data-v1",
        "train_row_ids_sha256": sha256_json(list(train_row_ids)),
        "val_row_ids_sha256": sha256_json(list(val_row_ids)),
        "control_row_ids_sha256": sha256_json(list(control_row_ids)),
        "allowed_gene_ids_sha256": sha256_json([gene_ids[i] for i in allowed_indices]),
        "excluded_gene_ids_sha256": sha256_json([gene_ids[i] for i in excluded]),
        "sanitization": (
            "selected_rows_only; excluded_columns_discarded_before_upstream; "
            "full_axis_placeholders_are_not_targets"
        ),
        "control_pairing": "cell_type::batch",
        "raw_auxiliary_layers_obsm_uns_absent": True,
        "primary_X_layer_alias": "AnnData>=0.13 may expose layers[None] as identical X",
    }
    return PreparedData(
        adata,
        gene_ids,
        graph_gene_ids,
        train_conditions,
        val_conditions,
        allowed_indices,
        excluded,
        train_row_ids,
        val_row_ids,
        control_row_ids,
        contexts,
        candidates(train_row_ids),
        candidates(val_row_ids),
        evidence,
    )
