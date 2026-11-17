"""Freeze paired K562/Jurkat test cohorts without reading expression values."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import anndata
import numpy as np
from evaluate_crosscell_k562 import SOURCE

from gradpert.data._io import atomic_json, read_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.v2.crosscell import FOLDS_SHA, SOURCE_SHA, frozen_rows

ROOT = Path("/data/yilangliu/GraD-Pert")
CROSS = ROOT / "crosscell-fixed-v1"
JURKAT = ROOT / "data-vnext-a942114/nadig_jurkat/within_cell_unseen_single"
REFERENCE = ROOT / "development/crosscell-k562-small-single-10ep-20261114-a1/eval-epoch5-k562-a1"
PRIOR_EVAL_SHA = "79274f56bac9b595828de4df08490ff5888e0de2178f4920f9905a3fecf61a09"
SELECTION_SEED = "gradpert-k562-holdout-v1"


def targets(condition: str) -> frozenset[str]:
    return frozenset(part for part in condition.split("+") if part != "ctrl")


def genes_in_panel(genes: tuple[str, ...], condition_ids: list[str]) -> list[str]:
    held = set().union(*(targets(condition) for condition in condition_ids))
    return [gene for gene in genes if gene in held]


def cross_cell() -> dict:
    source = CROSS / "source/K562_cross_cell_lines/de_adata_test.h5ad"
    folds_path = CROSS / "folds-baadb85/manifest.json"
    done = REFERENCE / "COMPLETE.json"
    if (
        sha256_file(folds_path) != FOLDS_SHA
        or sha256_file(source) != SOURCE_SHA
        or sha256_file(done) != PRIOR_EVAL_SHA
    ):
        raise ValueError("cross-cell source, folds or prior evaluation identity changed")
    manifest = read_json(folds_path)
    fold = manifest["folds"]["K562"]
    index = folds_path.parent / fold["row_index_file"]
    if sha256_file(index) != fold["row_index_sha256"]:
        raise ValueError("cross-cell row index changed")
    controls = REFERENCE / "FROZEN_ROWS.json"
    controls_sha = sha256_file(controls)
    prior = read_json(done)
    if controls_sha != prior["frozen_rows_sha256"]:
        raise ValueError("prior frozen pairing changed")
    draws = read_json(controls)["draws"]
    adata = anndata.read_h5ad(source, backed="r")
    try:
        genes = tuple(map(str, adata.var_names))
        if sha256_json(list(genes)) != manifest["gene_order_sha256"]:
            raise ValueError("cross-cell gene order changed")
        rows = frozen_rows(index, fold, adata.obs)
        conditions = adata.obs["condition"].astype(str).to_numpy()
        cell_lines = adata.obs["cell_line"].astype(str).to_numpy()
    finally:
        adata.file.close()
    truth_counts = Counter(conditions[rows["test"]])
    source_conditions = set(conditions[np.r_[rows["train"], rows["val"]]])
    natural = sorted(set(truth_counts) - source_conditions)
    source_line_conditions = set(
        zip(cell_lines[rows["train"]], conditions[rows["train"]], strict=True)
    )
    line_counts = Counter(condition for _, condition in source_line_conditions)
    gene_set = set(genes)
    eligible = [
        condition
        for condition, count in truth_counts.items()
        if line_counts[condition] >= 2
        and count >= 20
        and "ctrl" in condition.split("+")
        and len(targets(condition)) == 1
        and targets(condition) <= gene_set
    ]
    if len(eligible) < 128:
        raise ValueError("engineered cohort cannot reach 128 eligible perturbations")
    ordered = sorted(eligible, key=lambda c: (sha256_json([SELECTION_SEED, c]), c))
    target_control_rows = set(rows["target_control"])

    def cohort(ids: list[str]) -> dict:
        held_targets = set().union(*(targets(condition) for condition in ids))
        omitted = {c for c in source_conditions if held_targets & targets(c)}
        keep = ~np.isin(conditions, sorted(omitted))
        fit = {k: rows[k][keep[rows[k]]].tolist() for k in ("train", "val", "source_control")}
        retained = set(conditions[fit["train"]]) | set(conditions[fit["val"]])
        if set(ids) & retained or any(held_targets & targets(c) for c in retained):
            raise ValueError("held-out target remains in source supervision")
        selected = {c: draws[c] for c in ids}
        for c, draw in selected.items():
            expected = rows["test"][conditions[rows["test"]] == c].tolist()
            if (
                draw["truth_rows"] != expected
                or len(draw["input_control_rows"]) != 300
                or set(draw["input_control_rows"]) - target_control_rows
            ):
                raise ValueError(f"control/truth mismatch for {c}")
        return {
            "condition_ids": ids,
            "condition_ids_sha256": sha256_json(ids),
            "truth_rows": sum(truth_counts[c] for c in ids),
            "per_condition_pairing_sha256": sha256_json(selected),
            "shared_frozen_draws_path": str(controls),
            "masked_genes_in_expression_order": genes_in_panel(genes, ids),
            "fit_row_counts": {k: len(v) for k, v in fit.items()},
            "fit_row_sha256": {k: sha256_json(v) for k, v in fit.items()},
            "removed_fit_condition_ids": sorted(omitted),
        }

    return {
        "target_cell_line": "K562",
        "source_cell_lines": fold["train_cell_lines"],
        "source_h5ad_sha256": SOURCE_SHA,
        "frozen_fold_sha256": sha256_file(index),
        "expression_panel_genes": len(genes),
        "expression_gene_order_sha256": manifest["gene_order_sha256"],
        "shared_frozen_draws_sha256": controls_sha,
        "upstream_panel_origin": "K562-selected common panel",
        "engineered_eligible_conditions": len(eligible),
        "engineered_selection_seed": SELECTION_SEED,
        "engineered_k562": cohort(ordered[:128]),
        "natural_k562": cohort(natural),
    }


def jurkat() -> dict:
    manifest_dir = JURKAT / "manifests"
    canonical = read_json(manifest_dir / "canonical.json")
    split_file = manifest_dir / "split.json"
    split = read_json(split_file)
    control_file = manifest_dir / "evaluation_controls.test.json"
    controls = read_json(control_file)
    control_hashes = {
        "test": sha256_file(control_file),
        "val": sha256_file(manifest_dir / "evaluation_controls.val.json"),
    }
    genes = tuple((JURKAT / "canonical/expression_gene_ids.txt").read_text().splitlines())
    if (
        sha256_file(split_file) != canonical["split_manifest_sha256"]
        or sha256_json(control_hashes) != canonical["evaluation_controls_sha256"]
        or controls["split_content_sha256"] != split["split_content_sha256"]
        or [draw["condition_id"] for draw in controls["draws"]] != split["test_conditions"]
        or any(len(draw["ordered_row_ids"]) != 300 for draw in controls["draws"])
        or sha256_json(list(genes)) != canonical["expression_gene_order_sha256"]
    ):
        raise ValueError("Jurkat split, expression order or evaluation controls changed")
    cohorts = {key: split[f"{key}_conditions"] for key in ("train", "val", "test")}
    train, val, test = (set(cohorts[key]) for key in ("train", "val", "test"))
    held = set().union(*(targets(c) for c in test))
    if train & val or (train | val) & test or any(targets(c) & held for c in train | val):
        raise ValueError("canonical Jurkat fit exposures overlap held-out targets")
    return {
        "split_content_sha256": canonical["split_content_sha256"],
        "canonical_h5ad_sha256": canonical["canonical_adata_sha256"],
        "evaluation_controls_sha256": control_hashes["test"],
        "expression_gene_order_sha256": canonical["expression_gene_order_sha256"],
        "expression_panel_genes": len(genes),
        "condition_counts": {k: len(v) for k, v in cohorts.items()},
        "condition_ids_sha256": {k: sha256_json(v) for k, v in cohorts.items()},
        "masked_genes_in_expression_order": genes_in_panel(genes, cohorts["test"]),
        "shared_split_path": str(split_file),
        "shared_frozen_draws_path": str(control_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--remote-ref", required=True)
    parser.add_argument("--publication", required=True, type=Path)
    parser.add_argument("--publication-sha", required=True)
    args = parser.parse_args()
    if not all(path.resolve().is_relative_to(ROOT) for path in (args.output, args.publication)):
        raise ValueError("formal study manifests stay on the server")
    identity = inspect_source_identity(
        SOURCE,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref=args.remote_ref,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    cross = cross_cell()
    inner = jurkat()
    args.output.mkdir(exist_ok=False)
    atomic_json(
        args.output / "cohorts.json",
        {
            "schema": "perturbation-holdout-study-cohorts-v1",
            "source_commit": identity.commit,
            "cross_cell": cross,
            "jurkat": inner,
            "expression_values_read": False,
            "fixed_panel_claim": "isolation downstream of the existing dataset preparation",
            "raw_strict_verified": False,
        },
    )


if __name__ == "__main__":
    main()
