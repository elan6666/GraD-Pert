"""Fit the released scBIG OT flow until the first scheduled gene-loss step."""

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
from benchmarks.crosscell.scbig_c2_flow_probe import (
    ASSETS_RECEIPT,
    COMMIT,
    LINE_NAMES,
    RESOURCES,
    SEED,
    SUPPORT_SHA,
    UPSTREAM,
    initialize_encoder,
    initialize_flow,
)
from benchmarks.crosscell.study_data import cohort_inputs
from benchmarks.crosscell.study_paths import SERVER, STUDY
from benchmarks.expression_holdout.data import read_allowed_rows
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot

STAGE1 = STUDY / "scbig-crosscell-c2-budget3h-gcae-stage1-a1"
STAGE1_RECEIPT_SHA = "43e4f24d0f64bbbb1180b423dadcdbee3424539810e7fa4b79b2f87d86350b94"
HGNC = SERVER.parent / "DinoGenePT/data/hgnc/2026-08-28/hgnc_complete_set.txt"
HGNC_SHA = "0615a070f1628e6727953f67ad9248dd0f0ddbb16d41a7b40e06aa852fc3f448"
COVERAGE = STUDY / "scbig-c2-esm2-coverage-a2.json"
COVERAGE_SHA = "0ae9df3193f284c1d545454fccbb9206a2a8b84894cb3fdcaf4a7f68ec1e1b62"
GENE_LOSS_PERIOD = 1000


def verify_assets():
    manifest = RESOURCES / "release-assets-a1.json"
    if sha256_file(manifest) != ASSETS_RECEIPT:
        raise ValueError("the official asset receipt changed")
    manifest = read_json(manifest)
    names = ("genecompass_gene_embeddings_full.pkl", "ESM2_pert_features.pt")
    files = tuple(RESOURCES / name for name in names)
    if any(sha256_file(path) != manifest[path.name]["sha256"] for path in files):
        raise ValueError("the official GeneCompass or ESM2 asset changed")
    return files


def restore_stage1(compass):
    from flax import serialization

    receipt = STAGE1 / "result.json"
    if sha256_file(receipt) != STAGE1_RECEIPT_SHA:
        raise ValueError("stage 1 receipt changed")
    fit = read_json(receipt)
    if (
        fit["status"] != "budgeted_gcae_stage1_complete"
        or fit["fit_stop_reason"] != "epoch_complete"
        or fit["test_conditions_accessed"]
    ):
        raise ValueError("the expected source-only stage 1 did not complete")
    parts = {
        "encoder_inputs.json": "inputs_sha256",
        "encoder.msgpack": "checkpoint_sha256",
        "encoder_input_center.npy": "encoder_input_center_sha256",
    }
    if any(sha256_file(STAGE1 / name) != fit[identity] for name, identity in parts.items()):
        raise ValueError("stage 1 artifacts disagree with the receipt")
    inputs = read_json(STAGE1 / "encoder_inputs.json")
    panel = inputs["grc_gene_order"]
    center = np.load(STAGE1 / "encoder_input_center.npy", allow_pickle=False)
    if len(panel) != 3352 or center.shape != (1, len(panel)) or not np.isfinite(center).all():
        raise ValueError("stage 1 checkpoint has an unexpected input center")
    encoder = initialize_encoder(panel, center, compass)
    encoder.state = serialization.from_bytes(
        encoder.state, (STAGE1 / "encoder.msgpack").read_bytes()
    )
    if int(encoder.state.step) != fit["fit_updates"]:
        raise ValueError("stage 1 CUDA checkpoint has an unexpected update count")
    return encoder, inputs


def audited_prior(esm, genes):
    import torch

    bank = torch.load(esm, map_location="cpu", weights_only=True)
    if sha256_file(HGNC) != HGNC_SHA or sha256_file(COVERAGE) != COVERAGE_SHA:
        raise ValueError("HGNC or ESM2 C2 coverage audit inputs changed")
    audit = read_json(COVERAGE)
    if len(bank) != audit["embeddings"] or audit["fit_unresolved_after_alias"]:
        raise ValueError("the released ESM2 bank does not match the C2 audit")
    records = pd.read_csv(
        HGNC,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["symbol", "prev_symbol", "alias_symbol"],
    )
    missing = set(genes) - bank.keys()
    if missing != set(genes).intersection(audit["fit_missing"]):
        raise ValueError("exact ESM2 support differs from the C2 fit audit")
    choices = {gene: set() for gene in missing}
    for approved, previous, synonyms in records.itertuples(index=False, name=None):
        if approved not in bank:
            continue
        for alias in missing.intersection((*previous.split("|"), *synonyms.split("|"))):
            choices[alias].add(approved)
    if any(len(matches) != 1 for matches in choices.values()):
        raise ValueError(f"non-unique or missing official ESM2 mappings: {choices}")
    aliases = {key: next(iter(choices[key])) for key in sorted(choices)}
    result = {
        gene: np.asarray(bank[aliases.get(gene, gene)], dtype=np.float32).reshape(-1)
        for gene in sorted(genes)
    }
    result["ctrl"] = np.zeros_like(next(iter(result.values())))
    return result, aliases


def prepare_latents(encoder, inputs, esm):
    source, _, cohort, frozen = cohort_inputs()
    support = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
    if sha256_file(support) != SUPPORT_SHA:
        raise ValueError("C2 source support changed")
    adata = anndata.read_h5ad(source, backed="r")
    try:
        conditions = adata.obs["condition"].astype(str).to_numpy()
        lines = adata.obs["cell_line"].astype(str).to_numpy()
        roles = crosscell_fit_rows(frozen, conditions, cohort, compatibility=read_json(support))
        rows = np.sort(np.r_[roles["train"], roles["source_control"]])
        if (
            set(lines[rows]) != set(LINE_NAMES)
            or sha256_json(rows.tolist()) != inputs["training_row_sha256"]
            or np.intersect1d(rows, np.r_[frozen["test"], frozen["target_control"]]).size
        ):
            raise ValueError("flow fit does not match the guarded encoder source split")
        panel = inputs["grc_gene_order"]
        held = set(cohort["masked_genes_in_expression_order"])
        canonical = list(map(str, adata.var_names))
        order = np.asarray(inputs["grc_indices"])
        keep = np.asarray([gene not in held for gene in panel])
        if list(np.asarray(canonical)[order]) != panel or np.count_nonzero(~keep) != 128:
            raise ValueError("flow expression differ from stage 1 full-axis identities")
        genes = adata.var_names.get_indexer(np.asarray(panel)[keep]).tolist()
        if min(genes) < 0 or np.any(np.asarray(encoder.X_mean)[0, ~keep]):
            raise ValueError("held expression leaked into the saved encoder centering")
        target = set(conditions[rows]) - {"ctrl"}
        if any(
            not condition.endswith("+ctrl") or condition.count("+") != 1 for condition in target
        ):
            raise ValueError("the released single-target flow path would drop a fitting condition")
        prior, aliases = audited_prior(
            esm, {condition.removesuffix("+ctrl") for condition in target}
        )
        visible = read_allowed_rows(adata, rows, genes, len(canonical)).toarray()
        packed = np.zeros((len(rows), len(panel)), dtype=np.float32)
        packed[:, keep] = visible
        del visible
        print(
            f"Encoding {len(rows)} source rows using the completed stage 1 checkpoint", flush=True
        )
        latent = np.asarray(encoder.transform(packed, batch_size=256), dtype=np.float32)
        del packed
        labels = conditions[rows]
        obs = pd.DataFrame(
            {
                "is_control": labels == "ctrl",
                "pert1": [condition.removesuffix("+ctrl") for condition in labels],
                "pert2": "ctrl",
                "condition": labels,
                "cell_line": lines[rows],
            },
            index=pd.Index(rows.astype(str)),
        )
    finally:
        adata.file.close()
    if latent.shape != (len(rows), 50) or not np.isfinite(latent).all():
        raise ValueError("nonfinite or misaligned sealed source embeddings")
    flow_data = anndata.AnnData(X=latent, obs=obs)
    flow_data.obsm["X_pca"] = latent
    flow_data.uns["pert_embeddings"] = prior
    flow_data.uns["cell_line_embeddings"] = dict(
        zip(LINE_NAMES, np.eye(len(LINE_NAMES), dtype=np.float32), strict=True)
    )
    return flow_data, aliases


def run(output):
    import jax
    from cellflow.data._dataloader import TrainSampler
    from flax import serialization

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("flow fitting requires one CUDA device")
    started = time.monotonic()
    compass, esm = verify_assets()
    encoder, inputs = restore_stage1(compass)
    source, aliases = prepare_latents(encoder, inputs, esm)
    flow = initialize_flow(source)
    sampler = TrainSampler(data=flow.train_data, batch_size=1024)
    treatment = source.obs.loc[~source.obs["is_control"]]
    expected_contexts = len(treatment[["condition", "cell_line"]].drop_duplicates())
    if sampler.n_source_dists != len(LINE_NAMES) or sampler.n_target_dists != expected_contexts:
        raise ValueError("official sampler lost a sealed source line or treatment context")
    rng = jax.random.PRNGKey(SEED)
    host_rng = np.random.default_rng(SEED)
    planned = GENE_LOSS_PERIOD - 1
    fit_started = time.monotonic()
    print(
        f"Starting {planned} released flow microsteps before the first gene-space update",
        flush=True,
    )
    for update in range(1, planned + 1):
        if time.monotonic() - started > 1800:
            raise RuntimeError("scBIG flow exceeded its first-stage resource allocation")
        rng, key = jax.random.split(rng)
        loss = float(flow.solver.step_fn(key, sampler.sample(host_rng)))
        if not np.isfinite(loss):
            raise ValueError("nonfinite official OT loss")
        if update == 1 or update % 25 == 0 or update == planned:
            print(f"flow_microstep={update}/{planned} loss={loss:.8g}", flush=True)
            atomic_json(
                output / "training_progress.json",
                {"microsteps": update, "planned": planned, "loss": loss},
            )
    state = flow.solver.vf_state
    effective = int(state.opt_state.gradient_step)
    if (
        int(state.step) != planned
        or effective != planned // 20
        or int(state.opt_state.mini_step) != planned % 20
    ):
        raise ValueError("the released flow accumulators have an unexpected update position")
    weights = output / "flow_train.msgpack"
    ema = output / "flow_inference.msgpack"
    rng_path = output / "flow_rng.json"
    weights.write_bytes(serialization.to_bytes(state))
    ema.write_bytes(serialization.to_bytes(flow.solver.vf_state_inference))
    atomic_json(
        rng_path,
        {"jax_key": np.asarray(rng).tolist(), "sampler_rng": host_rng.bit_generator.state},
    )
    atomic_json(
        output / "result.json",
        {
            "status": "official_ot_before_first_gene_update",
            "scenario": "C2",
            "formal_eligible": False,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "upstream_commit": COMMIT,
            "encoder_receipt_sha256": STAGE1_RECEIPT_SHA,
            "train_row_sha256": inputs["training_row_sha256"],
            "source_rows": int(source.n_obs),
            "source_treatment_contexts": expected_contexts,
            "source_lines": list(LINE_NAMES),
            "esm2_resolved_aliases": aliases,
            "flow_microsteps": planned,
            "flow_optimizer_updates": effective,
            "flow_accumulated_microsteps": planned % 20,
            "gene_space_updates": 0,
            "next_gene_space_update_at": GENE_LOSS_PERIOD,
            "test_conditions_accessed": 0,
            "flow_train_sha256": sha256_file(weights),
            "flow_inference_sha256": sha256_file(ema),
            "flow_rng_sha256": sha256_file(rng_path),
            "fit_seconds": time.monotonic() - fit_started,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != STUDY or any(
        os.getenv(key) != value
        for key, value in {
            "CUDA_VISIBLE_DEVICES": "1",
            "JAX_PLATFORMS": "cuda",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }.items()
    ):
        raise ValueError("the stage 2 runner requires the sealed GPU 1 environment")
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
                "stage": "ot_flow_before_gene_space",
                "owner": process_info(os.getpid()),
                "physical_gpu": 1,
                "gpu_uuid": args.gpu_uuid,
            },
        )
        sys.path.insert(0, str(UPSTREAM))
        run(output)


if __name__ == "__main__":
    main()
