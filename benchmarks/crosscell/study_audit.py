"""Verify the exact paired cross-cell fit boundaries without materializing expression."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata

from benchmarks.common.study_fit_partition import open_crosscell_fit_partition
from benchmarks.crosscell.study_paths import COHORT_SHA, SERVER, STUDY
from benchmarks.crosscell.study_paths import REPO as ROOT
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file
from gradpert.training.v2.crosscell import FOLDS_SHA, SOURCE_SHA, frozen_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", required=True, type=Path)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not all(p.resolve().is_relative_to(STUDY) for p in (args.output, args.publication)):
        raise ValueError("formal fit audit stays on the study server")
    source = inspect_source_identity(
        ROOT,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    cache = SERVER / "crosscell-fixed-v1"
    input_path = cache / "source/K562_cross_cell_lines/de_adata_test.h5ad"
    manifest_path = cache / "folds-baadb85/manifest.json"
    cohort_path = STUDY / "frozen/cohorts.json"
    for path, digest in (
        (input_path, SOURCE_SHA),
        (manifest_path, FOLDS_SHA),
        (cohort_path, COHORT_SHA),
    ):
        if sha256_file(path) != digest:
            raise ValueError(f"fixed study input changed: {path.name}")
    manifest = read_json(manifest_path)["folds"]["K562"]
    indices = manifest_path.parent / manifest["row_index_file"]
    if sha256_file(indices) != manifest["row_index_sha256"]:
        raise ValueError("K562 fold assignment changed")
    adata = anndata.read_h5ad(input_path, backed="r")
    try:
        rows = frozen_rows(indices, manifest, adata.obs)
    finally:
        adata.file.close()
    cohort = read_json(cohort_path)["cross_cell"]["engineered_k562"]
    arms = {
        name: open_crosscell_fit_partition(
            source=input_path,
            verified_rows=rows,
            cohort=cohort,
            mask_expression=mask,
            metadata_only=True,
        )
        for name, mask in (("C1", False), ("C2", True))
    }
    if arms["C1"]["fit_row_position_sha256"] != arms["C2"]["fit_row_position_sha256"]:
        raise ValueError("paired cross-cell fitting row IDs changed")
    args.output.mkdir(exist_ok=False)
    atomic_json(
        args.output / "COMPLETE.json",
        {
            "source_commit": source.commit,
            "cohorts_sha256": COHORT_SHA,
            "expression_values_read": False,
            "arms": arms,
        },
    )


if __name__ == "__main__":
    main()
