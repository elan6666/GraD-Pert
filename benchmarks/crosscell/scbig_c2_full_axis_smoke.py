"""Check released scBIG GCAE decoding on the full, expression-masked C2 panel."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import anndata
import numpy as np

from benchmarks.crosscell.scbig_c2_encoder_smoke import ENCODER_FILES
from benchmarks.crosscell.scbig_c2_flow_probe import (
    ASSETS_RECEIPT,
    BATCH_CELLS,
    COMMIT,
    RESOURCES,
    SEED,
    UPDATES,
    UPSTREAM,
    initialize_encoder,
    source_inputs,
)
from benchmarks.crosscell.study_data import cohort_inputs
from benchmarks.crosscell.study_paths import SERVER, STUDY
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot


def fit_full_axis(output):
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from scbig.clustering.grc import (
        build_gene_embedding_matrix,
        formula1_grc_clustering,
        load_genecompass,
        load_string_ppi_adjacency,
        reorder_genes_by_cluster,
    )
    from scbig.models._gcae_impl import train_step_bio

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("expected exactly one CUDA device")
    started = time.monotonic()
    source, _, cohort, _ = cohort_inputs()
    adata = anndata.read_h5ad(source, backed="r")
    try:
        full_axis = list(map(str, adata.var_names))
    finally:
        adata.file.close()
    visible, matrices, hashes = source_inputs()
    kept = set(visible)
    held = [gene for gene in full_axis if gene not in kept]
    if (
        len(full_axis) != 3352
        or len(held) != 128
        or held != cohort["masked_genes_in_expression_order"]
    ):
        raise ValueError("C2 full and visible panels disagree")
    manifest = RESOURCES / "release-assets-a1.json"
    if sha256_file(manifest) != ASSETS_RECEIPT:
        raise ValueError("released asset manifest changed")
    records = read_json(manifest)
    for name in ENCODER_FILES:
        if sha256_file(RESOURCES / name) != records[name]["sha256"]:
            raise ValueError(f"released asset changed: {name}")

    compass, info, links = (str(RESOURCES / name) for name in ENCODER_FILES)
    print("Building full-axis GRC order from external gene priors only", flush=True)
    embedding, coverage = build_gene_embedding_matrix(
        full_axis, *load_genecompass(compass), fill_missing=True, random_seed=SEED
    )
    adjacency, edges = load_string_ppi_adjacency(full_axis, info, links, min_score=700)
    labels, _ = formula1_grc_clustering(
        embedding,
        adjacency,
        n_clusters=32,
        n_refinement_steps=3,
        sinkhorn_reg=0.1,
        semantic_weight=1.0,
        ppi_weight=1.0,
        random_state=SEED,
    )
    genes, _, order = reorder_genes_by_cluster(full_axis, labels, return_order=True)
    mask = np.asarray([gene in kept for gene in genes], dtype=bool)
    if list(np.asarray(genes)[np.argsort(order)]) != full_axis or int(mask.sum()) != len(visible):
        raise ValueError("GRC did not preserve the canonical C2 axis")
    index = {gene: i for i, gene in enumerate(visible)}
    visible_indices = [index[gene] for gene in genes if gene in kept]

    def pack(matrix):
        packed = np.zeros((len(matrix), len(genes)), dtype=np.float32)
        packed[:, mask] = matrix[:, visible_indices]
        return packed

    train, val, controls = [pack(matrix) for matrix in matrices]
    encoder = initialize_encoder(list(genes), train, Path(compass))
    keep = jnp.asarray(mask)
    released_apply = encoder.state.apply_fn

    def apply_masked(*args, **kwargs):
        (latent, prediction), changed = released_apply(*args, **kwargs)
        return (latent, jnp.where(keep, prediction, 0.0)), changed

    state = encoder.state.replace(apply_fn=apply_masked)
    train_centered = train - np.asarray(encoder.X_mean)
    mask_loss = jax.jit(train_step_bio, static_argnames=["training"])
    rng, permutation_rng = jax.random.split(jax.random.PRNGKey(0))
    permutation = np.asarray(jax.random.permutation(permutation_rng, len(train)))
    if len(permutation) != BATCH_CELLS * UPDATES:
        raise ValueError("bounded full-axis C2 training rows changed")
    losses = []
    fit_started = time.monotonic()
    print(
        "Starting 10 released GCAE updates with reconstruction masked at all 128 held slots",
        flush=True,
    )
    for positions in permutation.reshape(UPDATES, BATCH_CELLS):
        rng, step = jax.random.split(rng)
        state, loss, _, _, _, projected = mask_loss(
            state,
            jnp.asarray(train_centered[positions]),
            step,
            encoder.ppi_mask,
            encoder.scgpt_pe,
            training=True,
        )
        losses.append(float(loss))
        if np.any(np.asarray(projected)[:, ~mask]):
            raise ValueError("excluded output contributed to the released loss")
    fit_seconds = time.monotonic() - fit_started
    if int(state.step) != UPDATES or not np.isfinite(losses).all():
        raise ValueError("unexpected GCAE update count or non-finite loss")

    checkpoint = output / "encoder.msgpack"
    checkpoint.write_bytes(serialization.to_bytes(state))
    encoder.state = serialization.from_bytes(encoder.state, checkpoint.read_bytes())
    val_recon = np.asarray(encoder.inverse_transform(encoder.transform(val)))
    visible_mse = float(np.mean((val_recon[:, mask] - val[:, mask]) ** 2))
    sample_started = time.monotonic()
    result = np.asarray(encoder.inverse_transform(encoder.transform(controls)))
    held_predictions = result[:, ~mask]
    if (
        result.shape != (300, len(full_axis))
        or not np.isfinite(result).all()
        or not np.isfinite(visible_mse)
        or np.any(np.std(held_predictions, axis=0, dtype=np.float64) == 0)
    ):
        raise ValueError("held output slots did not yield finite varying model predictions")
    inputs = output / "encoder_inputs.json"
    atomic_json(
        inputs,
        {
            **hashes,
            "canonical_gene_order_sha256": sha256_json(full_axis),
            "grc_gene_order": list(genes),
            "grc_indices": order.tolist(),
            "held_gene_order_sha256": sha256_json(held),
            "genecompass_coverage": coverage,
            "string_edges": int(edges),
        },
    )
    atomic_json(
        output / "result.json",
        {
            "status": "full_axis_gcae_smoke_complete",
            "scenario": "C2",
            "formal_eligible": False,
            "test_conditions_accessed": 0,
            "source_control_reconstruction_shape": list(result[:, np.argsort(order)].shape),
            "fit_masked_input_columns": len(held),
            "fit_masked_loss_columns": len(held),
            "varying_held_prediction_columns": len(held),
            "upstream_commit": COMMIT,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "fit_updates": int(encoder.state.step),
            "loss_normalization_width": len(full_axis),
            "first_fit_loss": losses[0],
            "last_fit_loss": losses[-1],
            "visible_source_validation_mse": visible_mse,
            "fit_seconds": fit_seconds,
            "source_control_inference_seconds": time.monotonic() - sample_started,
            "elapsed_seconds": time.monotonic() - started,
            "inputs_sha256": sha256_file(inputs),
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != STUDY or any(
        os.getenv(name) != value
        for name, value in {
            "CUDA_VISIBLE_DEVICES": "1",
            "JAX_PLATFORMS": "cuda",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }.items()
    ):
        raise ValueError("the full-axis smoke requires the sealed GPU 1 environment")
    git = ["git", "-C", str(UPSTREAM)]
    if (
        subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != COMMIT
        or subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
    ):
        raise ValueError("the official upstream checkout changed")
    with lock(SERVER / "runtime/v2-gpu-1.lock"):
        admit_selected_idle(snapshot(os.getpid()), 1, args.gpu_uuid)
        output.mkdir()
        atomic_json(
            output / "worker.json",
            {
                "stage": "full_axis_masked_gcae",
                "owner": process_info(os.getpid()),
                "physical_gpu": 1,
                "gpu_uuid": args.gpu_uuid,
            },
        )
        sys.path.insert(0, str(UPSTREAM))
        fit_full_axis(output)


if __name__ == "__main__":
    main()
