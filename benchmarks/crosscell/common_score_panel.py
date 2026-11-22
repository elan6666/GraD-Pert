"""Select the C2 scoring genes supported by all official baseline adapters."""

from __future__ import annotations

import argparse

import anndata

from benchmarks.crosscell.study_data import cohort_inputs
from benchmarks.crosscell.study_paths import COHORT_SHA, STUDY
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file, sha256_json

AUDIT = STUDY / "remaining-c2-readiness-a2.json"
AUDIT_SHA = "808fdd8e9bd7058eb5ebb7b51e38544b76db563c46046cb7dd0cea32e73887a3"
PANEL = STUDY / "frozen/crosscell-c2-common-output-3194-a1.json"


def panel_document(source, cross, cohort):
    if sha256_file(AUDIT) != AUDIT_SHA:
        raise ValueError("the verified C2 output coverage audit changed")
    adata = anndata.read_h5ad(source, backed="r")
    try:
        canonical = list(map(str, adata.var_names))
    finally:
        adata.file.close()
    coverage = read_json(AUDIT)["coverage_options"]
    unavailable = {
        "scfoundation": coverage["scfoundation_without_model_output"],
        "prescribe": coverage["prescribe_without_model_output"],
    }
    omitted = set().union(*unavailable.values())
    genes = [gene for gene in canonical if gene not in omitted]
    if (
        len(canonical) != 3352
        or len(genes) != 3194
        or len(coverage["scfoundation_without_model_output"]) != 30
        or set(unavailable["prescribe"]) != set(cohort["masked_genes_in_expression_order"])
        or genes != coverage["common_model_output_gene_names"]
        or coverage["conditions_with_resolved_target_symbols"] != 128
        or sha256_json(canonical) != cross["expression_gene_order_sha256"]
    ):
        raise ValueError("the common output panel does not match the frozen C2 audit")
    return {
        "protocol": "crosscell_k562_dual_holdout_common_output_3194",
        "cohort_sha256": COHORT_SHA,
        "audit_sha256": AUDIT_SHA,
        "canonical_gene_order_sha256": cross["expression_gene_order_sha256"],
        "score_gene_order_sha256": sha256_json(genes),
        "score_genes": genes,
        "omissions_by_model": unavailable,
        "condition_ids_sha256": cohort["condition_ids_sha256"],
        "paired_rows_sha256": cohort["per_condition_pairing_sha256"],
        "controls_per_condition": 300,
    }


def load_panel(source, cross, cohort):
    expected = panel_document(source, cross, cohort)
    if read_json(PANEL) != expected:
        raise ValueError("the frozen 3194-gene scoring panel changed")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if PANEL.exists():
        raise FileExistsError(PANEL)
    source, cross, cohort, _ = cohort_inputs()
    atomic_json(PANEL, panel_document(source, cross, cohort))
    print(f"{sha256_file(PANEL)}  {PANEL}")


if __name__ == "__main__":
    main()
