"""Prepare GEARS's released co-expression graph on the C2 fit-only expression panel."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from benchmarks.common.study_fit_partition import crosscell_fit_rows
from benchmarks.crosscell.common_score_panel import AUDIT, AUDIT_SHA, load_panel
from benchmarks.crosscell.official_fit_probe import SUPPORT_SHA
from benchmarks.crosscell.scfoundation_c2_weight_probe import COMMIT, UPSTREAM
from benchmarks.crosscell.study_data import cohort_inputs
from benchmarks.crosscell.study_paths import STUDY
from benchmarks.expression_holdout.data import read_allowed_rows
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file, sha256_json

SUPPORT = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    output = parser.parse_args().output.resolve()
    if output.parent != STUDY or output.exists() or os.getenv("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("use a fresh server directory and CPU-only execution")
    source, cross, cohort, rows = cohort_inputs()
    panel = load_panel(source, cross, cohort)
    if sha256_file(AUDIT) != AUDIT_SHA or sha256_file(SUPPORT) != SUPPORT_SHA:
        raise ValueError("the fit support or scFoundation alignment audit changed")
    git = ["git", "-C", str(UPSTREAM)]
    if (
        subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != COMMIT
        or subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
    ):
        raise ValueError("the official scFoundation source changed")
    audit = read_json(AUDIT)["scfoundation"]
    vocab = UPSTREAM / "model/OS_scRNA_gene_index.19264.tsv"
    if sha256_file(vocab) != audit["vocabulary_sha256"]:
        raise ValueError("the scFoundation vocabulary changed")
    with vocab.open() as stream:
        native = [record["gene_name"] for record in csv.DictReader(stream, delimiter="\t")]
    genes = panel["score_genes"]
    aliases = audit["expression"]["aliases"]
    mapped = [aliases.get(gene, gene) for gene in genes]
    node_map = {gene: index for index, gene in enumerate(native)}
    if (
        len(node_map) != 19264
        or len(set(mapped)) != len(genes)
        or not set(mapped) <= node_map.keys()
    ):
        raise ValueError("C2 expression aliases do not map bijectively into scFoundation")
    h5ad = anndata.read_h5ad(source, backed="r")
    try:
        conditions = h5ad.obs["condition"].astype(str).to_numpy()
        fit = crosscell_fit_rows(rows, conditions, cohort, compatibility=read_json(SUPPORT))
        train = np.sort(fit["train"])
        if np.intersect1d(train, np.r_[rows["test"], rows["target_control"], fit["val"]]).size:
            raise ValueError("non-training rows crossed the graph boundary")
        cols = h5ad.var_names.get_indexer(genes)
        if np.any(cols < 0) or set(genes) & set(cohort["masked_genes_in_expression_order"]):
            raise ValueError("held expression crossed the graph boundary")
        matrix = read_allowed_rows(h5ad, train, cols, h5ad.n_vars)
    finally:
        h5ad.file.close()
    adata = anndata.AnnData(
        matrix,
        obs=pd.DataFrame({"condition": conditions[train]}, index=train.astype(str)),
        var=pd.DataFrame({"gene_name": mapped}, index=mapped),
    )
    (output / "c2").mkdir(parents=True)
    sys.path[:0] = [str(UPSTREAM / "GEARS"), str(UPSTREAM / "model")]
    from gears.utils import GeneSimNetwork, get_coexpression_network_from_train

    threshold, neighbors = 0.4, 20
    print(f"C2 training co-expression rows={len(train)} genes={len(genes)}", flush=True)
    edges = get_coexpression_network_from_train(
        adata,
        threshold,
        neighbors,
        str(output),
        "c2",
        "frozen",
        1,
        0.75,
        {"train": sorted(set(conditions[train]))},
    )
    if edges.empty or not np.isfinite(edges.importance).all():
        raise ValueError("the official co-expression builder returned invalid weights")
    graph = GeneSimNetwork(edges, native, node_map)
    indices = graph.edge_index.numpy()
    weights = graph.edge_weight.numpy()
    if not np.isin(indices, [node_map[gene] for gene in mapped]).all() or indices.shape != (
        2,
        len(weights),
    ):
        raise ValueError("the graph was not lifted correctly to scFoundation's native order")
    path = output / "coexpression.npz"
    np.savez(path, edge_index=indices, edge_weight=weights)
    receipt = output / "result.json"
    atomic_json(
        receipt,
        {
            "status": "coexpression_verified",
            "official_commit": COMMIT,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "training_rows": len(train),
            "training_row_order_sha256": sha256_json(train.tolist()),
            "fit_support_sha256": SUPPORT_SHA,
            "input_genes": len(genes),
            "input_gene_order_sha256": sha256_json(genes),
            "native_genes": len(native),
            "native_gene_order_sha256": sha256_json(native),
            "edges": len(weights),
            "threshold": threshold,
            "neighbors": neighbors,
            "graph_sha256": sha256_file(path),
            "validation_rows_read": 0,
            "test_rows_read": 0,
        },
    )
    print(sha256_file(receipt), receipt)


if __name__ == "__main__":
    main()
