"""Expose only audited train, validation and source-control data to official runners."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from benchmarks.expression_holdout.data import build_prepared, read_allowed_rows
from gradpert.hashing import sha256_file, sha256_json


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


def crosscell_fit_rows(rows, conditions, cohort, *, compatibility=None):
    """Keep the sealed holdout, optionally applying the shared C2 support manifest."""
    baseline = restrict_fit_rows(
        rows, conditions, cohort["condition_ids"], cohort["fit_row_sha256"]
    )
    if compatibility is None:
        return baseline
    exclusions = set(compatibility["exclusion_ids"])
    if (
        compatibility["schema"] != "crosscell-shared-fit-support-v1"
        or compatibility["condition_ids_sha256"] != cohort["condition_ids_sha256"]
        or compatibility["paired_rows_sha256"] != cohort["per_condition_pairing_sha256"]
        or compatibility["parent_fit_row_sha256"] != cohort["fit_row_sha256"]
        or not exclusions
        or not exclusions.isdisjoint((*cohort["condition_ids"], "ctrl"))
    ):
        raise ValueError("C2 shared support manifest does not match the frozen cohort")
    retained = {
        role: indices[~np.isin(conditions[indices], sorted(exclusions))]
        for role, indices in baseline.items()
    }
    for role, indices in retained.items():
        if (
            len(indices) != compatibility["fit_counts"][role]
            or sha256_json(indices.tolist()) != compatibility["fit_row_sha256"][role]
        ):
            raise ValueError(f"C2 support manifest does not match the retained {role} rows")
    return retained


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
    *,
    source: Path,
    verified_rows,
    cohort,
    mask_expression: bool,
    metadata_only: bool = False,
    fit_compatibility=None,
):
    """Read verified fixed-scale cache. Test truth and K562 controls stay outside fit."""

    import anndata

    adata = anndata.read_h5ad(source, backed="r")
    try:
        conditions = adata.obs["condition"].astype(str).to_numpy()
        lines = adata.obs["cell_line"].astype(str).to_numpy()
        batches = adata.obs["batch"].astype(str).to_numpy()
        panel = tuple(map(str, adata.var_names))
        roles = crosscell_fit_rows(
            verified_rows, conditions, cohort, compatibility=fit_compatibility
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
        if fit_compatibility is not None:
            receipt["shared_fit_support_sha256"] = sha256_json(fit_compatibility)
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


def open_jurkat_double_holdout_fit(data, *, cohort_path: Path, cohort_sha256: str):
    """Use the study's frozen Jurkat identities and project expression before fit."""
    import json

    if sha256_file(cohort_path) != cohort_sha256:
        raise ValueError("sealed double-holdout cohort hash mismatch")
    cohort = json.loads(cohort_path.read_text())["jurkat"]
    split = data.split
    panel = tuple(data.expression_gene_ids)
    control = split.control_condition_id
    test_targets = {gene for condition in split.test_conditions for gene in condition.split("+")}
    test_targets.discard(control)
    held = [gene for gene in panel if gene in test_targets]
    fit_targets = {
        gene
        for condition in (*split.train_conditions, *split.val_conditions)
        for gene in condition.split("+")
        if gene != control
    }
    fit_roles = {
        "train": set(split.train_conditions),
        "val": set(split.val_conditions),
        "control": {control},
    }
    test_controls = data.layout.manifests / "evaluation_controls.test.json"
    if (
        data.manifest.dataset_id != "nadig_jurkat"
        or data.manifest.canonical_adata_sha256 != cohort["canonical_h5ad_sha256"]
        or split.split_content_sha256 != cohort["split_content_sha256"]
        or sha256_json(list(split.test_conditions)) != cohort["condition_ids_sha256"]["test"]
        or len(panel) != cohort["expression_panel_genes"]
        or sha256_json(list(panel)) != cohort["expression_gene_order_sha256"]
        or sha256_file(test_controls) != cohort["evaluation_controls_sha256"]
        or not held
        or held != cohort["masked_genes_in_expression_order"]
        or not test_targets.isdisjoint(fit_targets)
    ):
        raise ValueError("Jurkat fit split, panel, controls or held targets differ from the cohort")
    positions = {
        role: tuple(i for i, condition in enumerate(data.condition_ids) if condition in conditions)
        for role, conditions in fit_roles.items()
    }
    selected = tuple(sorted(i for indices in positions.values() for i in indices))
    allowed = allowed_expression_indices(panel, held, masked=True)
    return build_prepared(
        matrix=read_allowed_rows(data._adata, selected, allowed, len(panel)),
        allowed_indices=allowed,
        gene_ids=panel,
        graph_gene_ids=tuple(data.graph_gene_ids),
        row_ids=tuple(data.row_ids[i] for i in selected),
        conditions=tuple(data.condition_ids[i] for i in selected),
        contexts=tuple(data.context_ids[i] for i in selected),
        train_row_ids=tuple(data.row_ids[i] for i in positions["train"]),
        val_row_ids=tuple(data.row_ids[i] for i in positions["val"]),
        control_row_ids=tuple(data.row_ids[i] for i in positions["control"]),
        train_conditions=tuple(split.train_conditions),
        val_conditions=tuple(split.val_conditions),
        control_condition=control,
        receipt={
            "scenario": "J2",
            "cohort_sha256": cohort_sha256,
            "canonical_data_sha256": data.manifest.canonical_adata_sha256,
            "split_content_sha256": split.split_content_sha256,
            "evaluation_controls_sha256": cohort["evaluation_controls_sha256"],
            "test_conditions_sha256": cohort["condition_ids_sha256"]["test"],
            "expression_gene_order_sha256": cohort["expression_gene_order_sha256"],
            "fit_row_counts": {role: len(indices) for role, indices in positions.items()},
            "fit_allowed_gene_count": len(allowed),
            "held_genes": held,
            "held_target_count_outside_panel": len(test_targets) - len(held),
            "fit_target_intersection_with_held": [],
            "test_truth_present_during_fit": False,
            "test_scoring_panel_gene_count": len(panel),
            "test_control_expression": "full_canonical_panel",
            "scale_scope": "existing fixed-panel dataset; preprocessing precedes this partition",
        },
    )
