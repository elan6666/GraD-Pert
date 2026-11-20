"""Restore the scBIG C2 encoder and exercise the released latent-flow path."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from benchmarks.common.study_fit_partition import crosscell_fit_rows
from benchmarks.crosscell.scbig_c2_encoder_smoke import (
    BATCH_CELLS,
    COMMIT,
    RESOURCES,
    SEED,
    SUPPORT_SHA,
    UPDATES,
    UPSTREAM,
    source_inputs,
)
from benchmarks.crosscell.study_data import cohort_inputs
from benchmarks.crosscell.study_paths import SERVER, STUDY
from benchmarks.expression_holdout.data import read_allowed_rows
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot

ENCODER_PROBE = STUDY / "scbig-crosscell-c2-official-encoder-smoke-a2"
ENCODER_RECEIPT = "33c1cebef51bd88692d8b796961a893b7a8be922ad92572099597444a7468295"
ASSETS_RECEIPT = "44e014d412ca61a70944f645ed33533ebc114fbfc9edc9ba42ef7d1a7a98e44b"
FULL_AXIS_PROBE = STUDY / "scbig-crosscell-c2-masked-full-axis-smoke-a1"
FULL_AXIS_RECEIPT = "ce0724430526f8959de5973599196f74f1c2225df8de2a20403ee0d4a72bd682"
ACCUMULATION = 20
LINE_NAMES = ("RPE1", "hepg2", "jurkat")


def initialize_encoder(panel, train, compass):
    """Construct the official wrapper and its training state for a fixed GRC order."""
    import jax
    import jax.numpy as jnp
    from scbig.models._gcae_impl import (
        BioInductiveAutoEncoder,
        create_bio_train_state,
        get_encoder_for_ablation,
    )

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
    encoder.set_priors(panel, genecompass_path=str(compass))
    encoder.input_dim = len(panel)
    encoder.X_mean = jnp.asarray(train.mean(axis=0, keepdims=True), dtype=jnp.float32)
    names = (
        "latent_dim",
        "embed_dim",
        "num_layers",
        "num_heads",
        "mlp_dim",
        "dropout_rate",
        "chunk_size",
        "pe_type",
        "attention_type",
        "window_size",
        "num_inducing",
        "n_pathway_queries",
        "use_pathway_pooling",
        "pooling_type",
    )
    encoder.model = BioInductiveAutoEncoder(
        output_dim=len(panel), **{name: getattr(encoder, name) for name in names}
    )
    encoder.state = create_bio_train_state(
        encoder.model,
        jax.random.PRNGKey(0),
        (BATCH_CELLS, len(panel)),
        1e-4,
        encoder.ppi_mask,
        encoder.scgpt_pe,
        encoder.pathway_matrix,
    )
    return encoder


def restore_encoder():
    from flax import serialization

    receipt_path = ENCODER_PROBE / "result.json"
    inputs_path = ENCODER_PROBE / "encoder_inputs.json"
    weights = ENCODER_PROBE / "encoder.msgpack"
    if sha256_file(receipt_path) != ENCODER_RECEIPT:
        raise ValueError("encoder receipt changed")
    receipt = read_json(receipt_path)
    if (
        receipt["status"] != "encoder_smoke_complete"
        or receipt["test_conditions_accessed"]
        or sha256_file(inputs_path) != receipt["inputs_sha256"]
        or sha256_file(weights) != receipt["checkpoint_sha256"]
    ):
        raise ValueError("encoder artifacts disagree with the successful source-only receipt")
    inputs = read_json(inputs_path)
    panel, (train, _, controls), row_hashes = source_inputs()
    order = np.asarray(inputs["grc_indices"])
    sorted_panel = [panel[index] for index in order]
    if (
        any(inputs[key] != value for key, value in row_hashes.items())
        or sorted_panel != inputs["ordered_genes"]
        or sorted(order.tolist()) != list(range(len(panel)))
    ):
        raise ValueError("could not reproduce the encoder fit input")
    train = np.asarray(train[:, order], dtype=np.float32)
    compass = RESOURCES / "genecompass_gene_embeddings_full.pkl"
    if sha256_file(compass) != receipt["asset_sha256"][compass.name]:
        raise ValueError("GeneCompass prior differs from the encoder receipt")
    encoder = initialize_encoder(sorted_panel, train, compass)
    encoder.state = serialization.from_bytes(encoder.state, weights.read_bytes())
    restored = np.asarray(encoder.inverse_transform(encoder.transform(controls[:, order])))
    error = float(np.mean((restored[:, np.argsort(order)] - controls) ** 2))
    if int(encoder.state.step) != UPDATES or not np.isclose(
        error, receipt["source_reconstruction_mse"], rtol=1e-6
    ):
        raise ValueError("restored encoder disagrees with the original reconstruction")
    return encoder, sorted_panel, order, error


def restore_full_axis_encoder():
    from flax import serialization

    receipt_path = FULL_AXIS_PROBE / "result.json"
    inputs_path = FULL_AXIS_PROBE / "encoder_inputs.json"
    weights = FULL_AXIS_PROBE / "encoder.msgpack"
    if sha256_file(receipt_path) != FULL_AXIS_RECEIPT:
        raise ValueError("full-axis encoder receipt changed")
    receipt = read_json(receipt_path)
    if (
        receipt["status"] != "full_axis_gcae_smoke_complete"
        or receipt["test_conditions_accessed"]
        or receipt["fit_masked_input_columns"] != 128
        or receipt["fit_masked_loss_columns"] != 128
        or sha256_file(inputs_path) != receipt["inputs_sha256"]
        or sha256_file(weights) != receipt["checkpoint_sha256"]
    ):
        raise ValueError("full-axis encoder artifacts disagree with the source-only receipt")
    inputs = read_json(inputs_path)
    visible, (train, val, _), row_hashes = source_inputs()
    panel = inputs["grc_gene_order"]
    order = np.asarray(inputs["grc_indices"])
    if len(panel) != 3352 or any(inputs[key] != value for key, value in row_hashes.items()):
        raise ValueError("full-axis encoder positions or source input rows changed")
    positions = {gene: i for i, gene in enumerate(visible)}
    keep = np.asarray([gene in positions for gene in panel])
    selected = [positions[gene] for gene in panel if gene in positions]

    def pack(values):
        result = np.zeros((len(values), len(panel)), dtype=np.float32)
        result[:, keep] = values[:, selected]
        return result

    manifest = RESOURCES / "release-assets-a1.json"
    compass = RESOURCES / "genecompass_gene_embeddings_full.pkl"
    if sha256_file(manifest) != ASSETS_RECEIPT or (
        sha256_file(compass) != read_json(manifest)[compass.name]["sha256"]
    ):
        raise ValueError("released prior changed")
    encoder = initialize_encoder(panel, pack(train), compass)
    encoder.state = serialization.from_bytes(encoder.state, weights.read_bytes())
    validation = pack(val)
    reconstruction = np.asarray(encoder.inverse_transform(encoder.transform(validation)))
    mse = float(np.mean((reconstruction[:, keep] - validation[:, keep]) ** 2))
    if int(encoder.state.step) != UPDATES or not np.isclose(
        mse, receipt["visible_source_validation_mse"], rtol=1e-6
    ):
        raise ValueError("full-axis encoder reload failed the original validation check")
    return encoder, panel, order, mse


def flow_inputs(panel, bank):
    source, _, cohort, frozen = cohort_inputs()
    support = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
    if sha256_file(support) != SUPPORT_SHA:
        raise ValueError("C2 shared fit support changed")
    rng = np.random.default_rng(SEED)
    adata = anndata.read_h5ad(source, backed="r")
    try:
        lines = adata.obs["cell_line"].astype(str).to_numpy()
        labels = adata.obs["condition"].astype(str).to_numpy()
        roles = crosscell_fit_rows(frozen, labels, cohort, compatibility=read_json(support))
        valid = np.asarray(
            [
                c != "ctrl"
                and all(g in bank for g in c.split("+") if g != "ctrl")
                and len(c.split("+")) <= 2
                for c in labels
            ],
            dtype=bool,
        )
        train_ids = []
        val_controls = []
        val_ids = []
        for line in LINE_NAMES:
            train = roles["train"][(lines[roles["train"]] == line) & valid[roles["train"]]]
            controls = roles["source_control"][lines[roles["source_control"]] == line]
            validation = roles["val"][(lines[roles["val"]] == line) & valid[roles["val"]]]
            train_ids.extend(rng.choice(train, 1024, replace=False).tolist())
            sampled_controls = rng.choice(controls, 600, replace=False)
            train_ids.extend(sampled_controls[:300].tolist())
            val_controls.append(np.sort(sampled_controls[300:]))
            val_ids.append(int(rng.choice(validation)))
        training = np.sort(train_ids)
        val_control = val_controls[0]
        used = np.r_[training, val_control, val_ids]
        if np.intersect1d(used, np.r_[frozen["test"], frozen["target_control"]]).size:
            raise ValueError("target-line rows entered the source-only flow probe")
        held = set(cohort["masked_genes_in_expression_order"])
        keep = np.asarray([gene not in held for gene in panel])
        columns = adata.var_names.get_indexer(np.asarray(panel)[keep]).tolist()
        if min(columns) < 0:
            raise ValueError("encoder gene order is missing from the frozen expression axis")

        def read(ids):
            values = read_allowed_rows(adata, ids, columns, adata.n_vars).toarray()
            if keep.all():
                return values
            packed = np.zeros((len(ids), len(panel)), dtype=np.float32)
            packed[:, keep] = values
            return packed

        def obs(ids):
            slots = [c.split("+") for c in labels[ids]]
            return pd.DataFrame(
                {
                    "is_control": labels[ids] == "ctrl",
                    "pert1": [parts[0] for parts in slots],
                    "pert2": [parts[1] if len(parts) == 2 else "ctrl" for parts in slots],
                    "cell_line": lines[ids],
                    "condition": labels[ids],
                },
                index=adata.obs_names[ids],
            )

        return (
            read(training),
            obs(training),
            read(val_control),
            obs(val_control),
            obs(np.asarray(val_ids[:1])),
            {
                "train": sha256_json(training.tolist()),
                "source_validation_control": sha256_json(val_control.tolist()),
                "source_validation_condition_row": sha256_json(val_ids[:1]),
            },
        )
    finally:
        adata.file.close()


def execute(output, full_axis=False):
    import jax
    import optax
    import torch
    from cellflow.data._dataloader import TrainSampler
    from cellflow.model import CellFlow
    from flax import serialization

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("the released flow did not receive one CUDA device")
    started = time.monotonic()
    restore = restore_full_axis_encoder if full_axis else restore_encoder
    encoder, panel, order, reconstruction_mse = restore()
    print("Verified the original encoder reconstruction. Encoder updates this run: 0.", flush=True)
    assets_path = RESOURCES / "release-assets-a1.json"
    if sha256_file(assets_path) != ASSETS_RECEIPT:
        raise ValueError("scBIG released asset manifest changed")
    bank_path = RESOURCES / "ESM2_pert_features.pt"
    asset_sha = read_json(assets_path)[bank_path.name]["sha256"]
    if sha256_file(bank_path) != asset_sha:
        raise ValueError("scBIG released ESM2 embedding bank changed")
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    x, train_obs, controls, control_obs, val_obs, hashes = flow_inputs(panel, bank)
    conditions = set(pd.concat([train_obs, val_obs])[["pert1", "pert2"]].stack()) - {"ctrl"}
    perturbations = {
        key: np.asarray(bank[key], dtype=np.float32).reshape(-1) for key in sorted(conditions)
    }
    perturbations["ctrl"] = np.zeros_like(next(iter(perturbations.values())))
    cell_lines = dict(zip(LINE_NAMES, np.eye(len(LINE_NAMES), dtype=np.float32), strict=True))

    def latent_data(values, obs):
        latent = np.asarray(encoder.transform(values, batch_size=256), dtype=np.float32)
        adata = anndata.AnnData(X=latent, obs=obs)
        adata.obsm["X_pca"] = latent
        adata.uns["pert_embeddings"] = perturbations
        adata.uns["cell_line_embeddings"] = cell_lines
        return adata

    train_data = latent_data(x, train_obs)
    validation_controls = latent_data(controls, control_obs)
    flow = CellFlow(train_data, solver="otfm")
    flow.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"pert": ["pert1", "pert2"]},
        perturbation_covariate_reps={"pert": "pert_embeddings"},
        sample_covariates=["cell_line"],
        sample_covariate_reps={"cell_line": "cell_line_embeddings"},
        split_covariates=["cell_line"],
        max_combination_length=2,
        null_value=0.0,
    )
    optimizer = optax.MultiSteps(
        optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam(), optax.scale(-5e-5)),
        ACCUMULATION,
    )
    flow.prepare_model(
        condition_mode="deterministic",
        regularization=0.0,
        pooling="attention_seed",
        condition_embedding_dim=256,
        time_encoder_dims=(1024, 1024),
        hidden_dims=(1024, 1024),
        decoder_dims=(1024, 1024),
        hidden_dropout=0.1,
        decoder_dropout=0.1,
        optimizer=optimizer,
        probability_path={"constant_noise": 0.5},
        condition_encoder_kwargs={},
    )
    sampler = TrainSampler(data=flow.train_data, batch_size=1024)
    rng = jax.random.PRNGKey(SEED)
    host_rng = np.random.default_rng(SEED)
    fit_started = time.monotonic()
    print("Starting 20 official OT flow microsteps on C2 source rows", flush=True)
    losses = []
    for _ in range(ACCUMULATION):
        rng, step = jax.random.split(rng)
        losses.append(flow.solver.step_fn(step, sampler.sample(host_rng)))
    fit_seconds = time.monotonic() - fit_started
    vf_state = flow.solver.vf_state
    effective_updates = int(vf_state.opt_state.gradient_step)
    if (
        int(vf_state.step) != ACCUMULATION
        or effective_updates != 1
        or not np.isfinite(losses).all()
    ):
        raise ValueError("unexpected released flow update count or non-finite loss")
    flow.solver.is_trained = True
    checkpoint = output / "flow.msgpack"
    checkpoint.write_bytes(serialization.to_bytes(flow.solver.vf_state_inference))
    flow.solver.vf_state_inference = serialization.from_bytes(
        flow.solver.vf_state_inference, checkpoint.read_bytes()
    )
    infer_started = time.monotonic()
    print("Running released frozen flow inference and decoder on 300 source controls", flush=True)
    predictions = flow.predict(validation_controls, val_obs, condition_id_key="condition")
    (latent,) = predictions.values()
    decoded = np.asarray(encoder.inverse_transform(latent))[:, np.argsort(order)]
    if decoded.shape != (300, len(panel)) or not np.isfinite(decoded).all():
        raise ValueError("scBIG did not decode 300 source controls on the sealed gene axis")
    if full_axis:
        _, _, cohort, _ = cohort_inputs()
        held = np.isin(
            np.asarray(panel)[np.argsort(order)], cohort["masked_genes_in_expression_order"]
        )
        if held.sum() != 128 or (np.std(decoded[:, held], axis=0, dtype=np.float64) == 0).any():
            raise ValueError("the full-axis flow did not produce varying held outputs")
    atomic_json(
        output / "result.json",
        {
            "status": "full_axis_flow_smoke_complete"
            if full_axis
            else "visible_axis_flow_smoke_complete",
            "scenario": "C2",
            "formal_eligible": False,
            "full_axis_source_prediction_verified": full_axis,
            "full_scoring_axis_verified": False,
            "upstream_commit": COMMIT,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "encoder_receipt_sha256": FULL_AXIS_RECEIPT if full_axis else ENCODER_RECEIPT,
            "encoder_reload_reconstruction_mse": reconstruction_mse,
            "encoder_updates": 0,
            "flow_microsteps": ACCUMULATION,
            "flow_optimizer_updates": effective_updates,
            "flow_checkpoint_sha256": sha256_file(checkpoint),
            "stages_omitted": ["gene_space_joint_loss", "augmented_ot"],
            "test_conditions_accessed": 0,
            "source_validation_condition": str(val_obs["condition"].iat[0]),
            "source_prediction_shape": list(decoded.shape),
            "fit_seconds": fit_seconds,
            "inference_seconds": time.monotonic() - infer_started,
            "total_seconds": time.monotonic() - started,
            "train_flow_loss": [float(loss) for loss in losses],
            "row_sha256": hashes,
            "esm2_sha256": asset_sha,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--full-axis", action="store_true")
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
        raise ValueError("the flow probe requires the sealed GPU 1 environment")
    git = ["git", "-C", str(UPSTREAM)]
    if (
        subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != COMMIT
        or subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
    ):
        raise ValueError("the pinned scBIG upstream checkout changed")
    with lock(SERVER / "runtime/v2-gpu-1.lock"):
        admit_selected_idle(snapshot(os.getpid()), 1, args.gpu_uuid)
        output.mkdir()
        atomic_json(
            output / "worker.json",
            {
                "stage": "full_axis_flow_smoke" if args.full_axis else "visible_axis_flow_smoke",
                "owner": process_info(os.getpid()),
                "physical_gpu": 1,
                "gpu_uuid": args.gpu_uuid,
            },
        )
        sys.path.insert(0, str(UPSTREAM))
        execute(output, full_axis=args.full_axis)


if __name__ == "__main__":
    main()
