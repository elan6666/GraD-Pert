"""Expose only audited train, validation and source-control data to official runners."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from benchmarks.expression_holdout.data import build_prepared, read_allowed_rows
from gradpert.hashing import sha256_json


def restrict_fit_rows(rows, condition_ids, test_ids, expected_row_hashes):
    """Remove each held target wherever it appears, including combinations."""
    held = {g for c in test_ids for g in c.split("+") if g != "ctrl"}
    if not held:
        raise ValueError("the frozen fit partition needs non-control test targets")
    kept = {
        role: np.asarray(
            [row for row in rows[role] if held.isdisjoint(str(condition_ids[row]).split("+"))],
            dtype=np.int64,
        )
        for role in ("train", "val", "source_control")
    }
    if any(
        not kept[role].size or sha256_json(kept[role].tolist()) != expected_row_hashes[role]
        for role in kept
    ):
        raise ValueError("filtered fit row order does not match the frozen protocol")
    combined = np.concatenate(list(kept.values()))
    if np.unique(combined).size != combined.size:
        raise ValueError("filtered fitting roles overlap")
    return kept


def allowed_expression_indices(gene_ids, held_genes, *, masked):
    if len(set(gene_ids)) != len(gene_ids):
        raise ValueError("expression panel contains duplicate gene identities")
    if not set(held_genes) <= set(gene_ids):
        raise ValueError("masked genes are outside the frozen expression panel")
    held = set(held_genes) if masked else set()
    allowed = tuple(i for i, gene in enumerate(gene_ids) if gene not in held)
    if not allowed:
        raise ValueError("no expression columns remain for the fit")
    return allowed


def open_crosscell_fit_partition(
    *, source: Path, verified_rows, cohort, mask_expression: bool, metadata_only: bool = False
):
    """Read verified fixed-scale cache. Test truth and K562 controls stay outside fit."""

    import anndata

    adata = anndata.read_h5ad(source, backed="r")
    try:
        conditions = adata.obs["condition"].astype(str).to_numpy()
        lines = adata.obs["cell_line"].astype(str).to_numpy()
        batches = adata.obs["batch"].astype(str).to_numpy()
        panel = tuple(map(str, adata.var_names))
        roles = restrict_fit_rows(
            verified_rows, conditions, cohort["condition_ids"], cohort["fit_row_sha256"]
        )
        ordered = np.sort(np.concatenate(list(roles.values())))
        if set(lines[ordered]) != {"RPE1", "hepg2", "jurkat"}:
            raise ValueError("target or unsupported cell line is present in fit")
        if not np.all(conditions[roles["source_control"]] == "ctrl"):
            raise ValueError("control role includes treatment rows")
        cols = allowed_expression_indices(
            panel, cohort["masked_genes_in_expression_order"], masked=mask_expression
        )
        control_contexts = {
            (lines[position], batches[position]) for position in roles["source_control"]
        }
        if not set(zip(lines[ordered], batches[ordered], strict=True)) <= control_contexts:
            raise ValueError("fit context lacks matched source controls")
        fit_targets = {
            gene
            for condition in set(conditions[ordered])
            for gene in condition.split("+")
            if gene != "ctrl"
        }
        test_targets = {
            gene
            for condition in cohort["condition_ids"]
            for gene in condition.split("+")
            if gene != "ctrl"
        }
        graph_genes = (*panel, *sorted((fit_targets | test_targets) - set(panel)))
        receipt = {
            "schema": "crosscell-fixed-panel-fit-v1",
            "counts": {name: len(values) for name, values in roles.items()},
            "fit_row_position_sha256": {
                name: sha256_json(values.tolist()) for name, values in roles.items()
            },
            "fit_line_counts": dict(Counter(lines[ordered])),
            "condition_ids_sha256": cohort["condition_ids_sha256"],
            "frozen_expression_sha256": sha256_json(list(panel)),
            "allowed_genes_count": len(cols),
            "allowed_gene_ids_sha256": sha256_json([panel[i] for i in cols]),
            "masked_genes_count": len(panel) - len(cols),
            "graph_gene_order_sha256": sha256_json(list(graph_genes)),
            "any_target_line_expression_read": False,
            "test_truth_present": False,
            "scale_scope": "existing fixed-panel dataset; preprocessing precedes this partition",
        }
        if metadata_only:
            return receipt
        # Readers project before returning any block to the upstream process.
        matrix = read_allowed_rows(adata, ordered, cols, len(panel))
        ids = {position: f"crosscell:{position}" for position in ordered}
        return build_prepared(
            matrix=matrix,
            allowed_indices=cols,
            gene_ids=panel,
            graph_gene_ids=graph_genes,
            row_ids=tuple(ids[position] for position in ordered),
            conditions=tuple(map(str, conditions[ordered])),
            contexts=tuple(
                f"{line}::{batch}"
                for line, batch in zip(lines[ordered], batches[ordered], strict=True)
            ),
            train_row_ids=tuple(ids[position] for position in roles["train"]),
            val_row_ids=tuple(ids[position] for position in roles["val"]),
            control_row_ids=tuple(ids[position] for position in roles["source_control"]),
            train_conditions=tuple(sorted(set(conditions[roles["train"]]))),
            val_conditions=tuple(sorted(set(conditions[roles["val"]]))),
            receipt=receipt,
        )
    finally:
        adata.file.close()
