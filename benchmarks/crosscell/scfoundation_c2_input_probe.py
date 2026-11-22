"""Verify scFoundation's released pre-normalized input rule on source C2 controls."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import anndata
import numpy as np
import torch

from benchmarks.common.study_fit_partition import crosscell_fit_rows
from benchmarks.crosscell.common_score_panel import AUDIT, AUDIT_SHA, load_panel
from benchmarks.crosscell.official_fit_probe import SUPPORT_SHA
from benchmarks.crosscell.scfoundation_c2_weight_probe import COMMIT, UPSTREAM
from benchmarks.crosscell.study_data import cohort_inputs
from benchmarks.crosscell.study_paths import STUDY
from benchmarks.expression_holdout.data import read_allowed_rows
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file, sha256_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != STUDY or output.exists() or os.getenv("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("use a fresh server receipt and a CPU-only environment")
    source, cross, cohort, rows = cohort_inputs()
    panel = load_panel(source, cross, cohort)
    support = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
    if sha256_file(AUDIT) != AUDIT_SHA or sha256_file(support) != SUPPORT_SHA:
        raise ValueError("C2 fit support or vocabulary audit changed")
    git = ["git", "-C", str(UPSTREAM)]
    if (
        subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != COMMIT
        or subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
    ):
        raise ValueError("scFoundation official source changed")
    audit = read_json(AUDIT)["scfoundation"]
    vocab_path = UPSTREAM / "model/OS_scRNA_gene_index.19264.tsv"
    if sha256_file(vocab_path) != audit["vocabulary_sha256"]:
        raise ValueError("the stock scFoundation vocabulary changed")
    with vocab_path.open() as stream:
        vocab = [entry["gene_name"] for entry in csv.DictReader(stream, delimiter="\t")]
    stock = {gene: index for index, gene in enumerate(vocab)}
    if len(vocab) != len(stock) or len(vocab) != 19264:
        raise ValueError("the released vocabulary must contain 19264 unique genes")

    adata = anndata.read_h5ad(source, backed="r")
    try:
        conditions = adata.obs["condition"].astype(str).to_numpy()
        lines = adata.obs["cell_line"].astype(str).to_numpy()
        fit = crosscell_fit_rows(rows, conditions, cohort, compatibility=read_json(support))
        source_lines = ("RPE1", "hepg2", "jurkat")
        controls = np.sort(
            np.concatenate(
                [
                    fit["source_control"][lines[fit["source_control"]] == name][:100]
                    for name in source_lines
                ]
            )
        )
        if (
            len(controls) != 300
            or not (conditions[controls] == "ctrl").all()
            or np.isin(lines[controls], "K562").any()
        ):
            raise ValueError("probe requires 300 source controls distributed over three cell lines")
        genes = panel["score_genes"]
        cols = adata.var_names.get_indexer(genes)
        if np.any(cols < 0) or not set(genes).isdisjoint(
            cohort["masked_genes_in_expression_order"]
        ):
            raise ValueError("the input axis exposed held or unknown expression")
        values = read_allowed_rows(adata, controls, cols, adata.n_vars).toarray()
    finally:
        adata.file.close()
    aliases = audit["expression"]["aliases"]
    positions = [stock[aliases.get(gene, gene)] for gene in genes]
    if len(set(positions)) != len(genes):
        raise ValueError("audited aliases collide in the official vocabulary")
    expanded = np.zeros((len(values), len(vocab)), dtype=np.float32)
    expanded[:, positions] = values
    # The released get_embedding.py pre_normalized=T path uses the sum of
    # its normalized input values. GEARS highres=0 appends log10(sum) twice.
    proxy = expanded.sum(1, dtype=np.float32)
    if not np.isfinite(expanded).all() or np.any(expanded < 0) or np.any(proxy <= 0):
        raise ValueError("the masked controls do not meet the released input rule")
    sys.path.insert(0, str(UPSTREAM / "GEARS"))
    from modules.encoders import choose_binset

    gene_values = torch.from_numpy(expanded)
    input_depth = torch.from_numpy(proxy[:, None])
    model_input = choose_binset(
        torch.cat((gene_values, input_depth), dim=1), "autobin_resolution_append", 100, highres=0
    )
    expected = torch.from_numpy(np.log10(proxy))
    torch.testing.assert_close(model_input[:, :19264], gene_values, rtol=0, atol=0)
    torch.testing.assert_close(model_input[:, -1], expected)
    torch.testing.assert_close(model_input[:, -2], expected)
    atomic_json(
        output,
        {
            "model": "scfoundation",
            "status": "released_pre_normalized_input_verified",
            "formal_eligible": False,
            "official_commit": COMMIT,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "vocabulary_sha256": audit["vocabulary_sha256"],
            "score_gene_order_sha256": sha256_json(genes),
            "source_control_row_order_sha256": sha256_json(controls.tolist()),
            "source_control_count_per_cell_line": dict.fromkeys(source_lines, 100),
            "measured_raw_depth": False,
            "depth_policy": "official_pre_normalized_T_sum_on_masked_and_mapped_expression",
            "input_shape": list(model_input.shape),
            "test_conditions_accessed": 0,
            "optimizer_updates": 0,
        },
    )
    print(sha256_file(output), output)


if __name__ == "__main__":
    main()
