"""Run the released scBIG GRC and its first training stage on capped C2 source rows."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import anndata
import numpy as np

from benchmarks.common.study_fit_partition import allowed_expression_indices, crosscell_fit_rows
from benchmarks.crosscell.official_fit_probe import SUPPORT_SHA
from benchmarks.crosscell.study_data import cohort_inputs
from benchmarks.crosscell.study_paths import SERVER, STUDY
from gradpert.data._io import atomic_json, read_json
from gradpert.evaluation.crosscell import read_rows
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot

UPSTREAM = SERVER / "upstreams/scbig-c2-5659ce9"
COMMIT = "5659ce995f0f1299155ae073540b78bb3e3f3524"
RESOURCES = SERVER / "upstreams/scbig-c2-assets-a1"
ENCODER_FILES = (
    "genecompass_gene_embeddings_full.pkl",
    "9606.protein.info.v12.0.txt.gz",
    "9606.protein.links.v12.0.txt.gz",
)
BATCH_CELLS = 256
UPDATES = 10
SEED = 42


def source_inputs():
    source, _, cohort, frozen = cohort_inputs()
    support = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
    if sha256_file(support) != SUPPORT_SHA:
        raise ValueError("the frozen C2 fitting population changed")
    adata = anndata.read_h5ad(source, backed="r")
    try:
        conditions = adata.obs["condition"].astype(str).to_numpy()
        roles = crosscell_fit_rows(frozen, conditions, cohort, compatibility=read_json(support))
        visible = list(
            allowed_expression_indices(
                list(adata.var_names), cohort["masked_genes_in_expression_order"], masked=True
            )
        )
        rng = np.random.default_rng(SEED)
        train_ids = np.sort(
            rng.choice(
                np.r_[roles["train"], roles["source_control"]], UPDATES * BATCH_CELLS, replace=False
            )
        )
        val_ids = np.sort(rng.choice(roles["val"], BATCH_CELLS, replace=False))
        control_ids = np.sort(rng.choice(roles["source_control"], 300, replace=False))
        used = np.r_[train_ids, val_ids, control_ids]
        if np.intersect1d(used, np.r_[frozen["test"], frozen["target_control"]]).size:
            raise ValueError("target-line expression crossed the encoder fitting boundary")
        gene_names = [str(adata.var_names[index]) for index in visible]
        return (
            gene_names,
            [read_rows(adata, ids)[:, visible] for ids in (train_ids, val_ids, control_ids)],
            {
                "shared_fit_support_sha256": SUPPORT_SHA,
                "visible_gene_order_sha256": sha256_json(gene_names),
                "training_row_sha256": sha256_json(train_ids.tolist()),
                "validation_row_sha256": sha256_json(val_ids.tolist()),
                "control_row_sha256": sha256_json(control_ids.tolist()),
            },
        )
    finally:
        adata.file.close()


def run(output, assets, gpu_identity):
    import jax
    from flax import serialization
    from scbig.clustering.grc import (
        build_gene_embedding_matrix,
        formula1_grc_clustering,
        load_genecompass,
        load_string_ppi_adjacency,
        reorder_genes_by_cluster,
    )
    from scbig.models._gcae_impl import get_encoder_for_ablation

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("scBIG encoder probe did not receive the selected CUDA device")
    gene_names, matrices, split = source_inputs()
    compass, info, links = (str(RESOURCES / name) for name in ENCODER_FILES)
    started = time.monotonic()
    print("Computing the released GRC32 ordering from GeneCompass and STRING", flush=True)
    embeddings, coverage = build_gene_embedding_matrix(
        gene_names, *load_genecompass(compass), fill_missing=True, random_seed=SEED
    )
    adjacency, edges = load_string_ppi_adjacency(gene_names, info, links, min_score=700)
    labels, _ = formula1_grc_clustering(
        embeddings,
        adjacency,
        n_clusters=32,
        n_refinement_steps=3,
        sinkhorn_reg=0.1,
        semantic_weight=1.0,
        ppi_weight=1.0,
        random_state=SEED,
    )
    sorted_genes, _, order = reorder_genes_by_cluster(gene_names, labels, return_order=True)
    inverse = np.argsort(order)
    atomic_json(
        output / "encoder_inputs.json",
        {
            **split,
            "ordered_genes": list(sorted_genes),
            "grc_indices": order.tolist(),
            "genecompass_coverage": coverage,
            "string_edges": int(edges),
        },
    )
    train, val, controls = (x[:, order] for x in matrices)

    encoder = get_encoder_for_ablation(
        "genecompass_pe_module",
        latent_dim=50,
        embed_dim=256,
        num_layers=3,
        num_heads=8,
        mlp_dim=512,
        dropout_rate=0.1,
        chunk_size=64,
        num_inducing=8,
        pooling_type="mean",
    )
    encoder.set_priors(list(sorted_genes), genecompass_path=compass)
    fit_started = time.monotonic()
    encoder.fit(
        train,
        X_test=val,
        n_epochs=1,
        batch_size=BATCH_CELLS,
        learning_rate=1e-4,
        eval_every_epochs=1,
        eval_label="frozen source-domain validation",
    )
    if int(encoder.state.step) != UPDATES:
        raise ValueError("unexpected released GCAE optimizer update count")
    weights = output / "encoder.msgpack"
    weights.write_bytes(serialization.to_bytes(encoder.state))
    encoder.state = serialization.from_bytes(encoder.state, weights.read_bytes())
    result = np.asarray(encoder.inverse_transform(encoder.transform(controls)))[:, inverse]
    if result.shape != (300, len(gene_names)) or not np.isfinite(result).all():
        raise ValueError("scBIG did not reconstruct 300 source controls on the visible gene axis")
    atomic_json(
        output / "result.json",
        {
            "status": "encoder_smoke_complete",
            "formal_eligible": False,
            "scenario": "C2",
            "upstream_commit": COMMIT,
            "gpu": 1,
            "gpu_uuid": gpu_identity,
            "jax_version": jax.__version__,
            "optimizer_updates": int(encoder.state.step),
            "test_conditions_accessed": 0,
            "reconstruction_shape": list(result.shape),
            "full_scoring_axis_verified": False,
            "flow_stage_executed": False,
            "checkpoint_sha256": sha256_file(weights),
            "inputs_sha256": sha256_file(output / "encoder_inputs.json"),
            "asset_sha256": assets,
            "fit_validation_reload_seconds": time.monotonic() - fit_started,
            "total_seconds": time.monotonic() - started,
            "source_reconstruction_mse": float(np.mean((result - matrices[2]) ** 2)),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-receipt", type=Path, required=True)
    parser.add_argument("--assets-receipt-sha", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.parent != STUDY or any(
        os.getenv(k) != v
        for k, v in {
            "CUDA_VISIBLE_DEVICES": "1",
            "JAX_PLATFORMS": "cuda",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }.items()
    ):
        raise ValueError("select physical GPU 1 in the sealed environment")
    rev = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True)
    dirty = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True
    )
    if rev.strip() != COMMIT or dirty.strip():
        raise ValueError("the official scBIG source differs from the pinned checkout")
    if sha256_file(args.assets_receipt) != args.assets_receipt_sha:
        raise ValueError("the encoder asset receipt changed")
    seal = read_json(args.assets_receipt)
    assets = {name: seal[name]["sha256"] for name in ENCODER_FILES}
    for name, digest in assets.items():
        if sha256_file(RESOURCES / name) != digest:
            raise ValueError(f"the encoder resource differs from the downloaded release: {name}")
    with lock(SERVER / "runtime/v2-gpu-1.lock"):
        admit_selected_idle(snapshot(os.getpid()), 1, args.gpu_uuid)
        args.output.mkdir()
        atomic_json(
            args.output / "worker.json",
            {
                "stage": "grc_and_gcae",
                "owner": process_info(os.getpid()),
                "physical_gpu": 1,
            },
        )
        sys.path.insert(0, str(UPSTREAM))
        run(args.output, assets, args.gpu_uuid)


if __name__ == "__main__":
    main()
