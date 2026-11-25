"""Run the released scBIG encoder and OT/gene-loss flow on Jurkat J2."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch

from benchmarks.crosscell import scbig_c2 as released
from benchmarks.crosscell.remaining_c2_readiness import hgnc_links, reconcile
from benchmarks.crosscell.replay_common_c2 import require_upstream
from benchmarks.crosscell.scbig_c2_encoder_smoke import ENCODER_FILES
from benchmarks.crosscell.scbig_c2_flow_fit import verify_assets
from benchmarks.crosscell.scbig_c2_flow_probe import initialize_encoder, initialize_flow
from benchmarks.crosscell.study_paths import COHORT_SHA, REPO, SERVER, STUDY
from benchmarks.jurkat_j2.study import (
    context,
    evaluate_segment,
    fit_data,
    frozen_receipt,
    score_axis,
)
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot

SEED = 42
FLOW_STEPS = 4800
CELL_BATCH = 256


def priors(esm, targets, fit_targets):
    bank = torch.load(esm, map_location="cpu", weights_only=True)
    support = reconcile(targets, set(bank), hgnc_links(), verified_aliases={})
    missing = support["unresolved"]
    if set(missing) != {"QARS"} or support["collisions"] or not set(missing) <= fit_targets:
        raise ValueError("unexpected Jurkat ESM2 coverage")
    mapping = support["aliases"]
    vectors = {
        n: np.asarray(bank[mapping.get(n, n)], dtype=np.float32).reshape(-1)
        for n in sorted(targets - set(missing))
    }
    donor = np.stack([vectors[n] for n in sorted(fit_targets & vectors.keys())])
    vectors.update({name: donor.mean(0, dtype=np.float64).astype(np.float32) for name in missing})
    vectors["ctrl"] = np.zeros_like(next(iter(vectors.values())))
    return vectors, {"aliases": mapping, "train_only_mean_prior_for": missing}


def fit_encoder(encoder, matrix, keep, output, *, smoke):
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from scbig.models._gcae_impl import train_step_bio

    apply = encoder.state.apply_fn

    def masked(*args, **kwargs):
        (latent, prediction), change = apply(*args, **kwargs)
        return (latent, jnp.where(keep, prediction, 0.0)), change

    state = encoder.state.replace(apply_fn=masked)
    step = jax.jit(train_step_bio, static_argnames=["training"])
    permutation = np.random.default_rng(SEED).permutation(len(matrix))
    rng = jax.random.PRNGKey(SEED)
    center = np.asarray(encoder.X_mean)
    count = 0
    for start in range(0, len(permutation), CELL_BATCH):
        if smoke and count == 10:
            break
        idx = permutation[start : start + CELL_BATCH]
        rng, key = jax.random.split(rng)
        state, loss, _, _, _, prediction = step(
            state,
            jnp.asarray(matrix[idx] - center),
            key,
            encoder.ppi_mask,
            encoder.scgpt_pe,
            training=True,
        )
        if not np.isfinite(float(loss)) or bool(jnp.any(prediction[:, ~keep])):
            raise ValueError("GCAE loss or held output does not respect J2")
        count += 1
        if count == 1 or count % 25 == 0:
            atomic_json(output / "training_progress.json", {"stage": "encoder", "updates": count})
    checkpoint = output / "encoder.msgpack"
    checkpoint.write_bytes(serialization.to_bytes(state))
    encoder.state = serialization.from_bytes(encoder.state, checkpoint.read_bytes())
    np.save(output / "encoder-center.npy", center, allow_pickle=False)
    return {"encoder_optimizer_updates": count, "encoder_sha256": sha256_file(checkpoint)}


def run(output, identity, *, smoke, resume):
    import jax
    from flax import serialization

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("scBIG J2 needs exactly one visible CUDA GPU")
    start = time.monotonic()
    checkpoint = output / "flow_inference.msgpack"
    if resume:
        receipt = frozen_receipt(output, identity, checkpoint.name)
    compass, esm = verify_assets()
    assets = read_json(released.RESOURCES / "release-assets-a1.json")
    for name in ENCODER_FILES:
        if sha256_file(released.RESOURCES / name) != assets[name]["sha256"]:
            raise ValueError(f"official GCAE asset changed: {name}")
    prepared = fit_data(SEED)
    with context() as data:
        test = tuple(data.split.test_conditions)
    genes, labels, order, coverage, edges = released.full_axis_grc(
        list(prepared.gene_ids),
        str(compass),
        *(str(released.RESOURCES / f) for f in ENCODER_FILES[1:]),
    )
    order = np.asarray(order)
    if list(np.asarray(prepared.gene_ids)[order]) != list(genes):
        raise ValueError("the GRC layout is not a permutation of J2 genes")
    keep = np.isin(order, np.asarray(prepared.allowed_indices))
    axis = score_axis(prepared.gene_ids)
    known = {str(row): i for i, row in enumerate(prepared.adata.obs_names)}
    rows = np.sort([known[row] for row in (*prepared.train_row_ids, *prepared.control_row_ids)])
    matrix = prepared.adata.X[rows][:, order].toarray()
    if matrix[:, ~keep].any():
        raise ValueError("J2 masked expression entered the GCAE")
    controls = np.asarray([known[row] for row in prepared.control_row_ids])
    encoder = initialize_encoder(list(genes), matrix, compass)
    inputs = {
        "gene_order": list(genes),
        "permutation": order.tolist(),
        "genecompass_coverage": coverage,
        "string_edges": int(edges),
    }
    if resume:
        encoder_checkpoint = output / "encoder.msgpack"
        if (
            sha256_file(encoder_checkpoint) != receipt["encoder_sha256"]
            or read_json(output / "encoder-inputs.json") != inputs
            or not np.array_equal(
                np.asarray(encoder.X_mean),
                np.load(output / "encoder-center.npy", allow_pickle=False),
            )
        ):
            raise ValueError("the frozen scBIG encoder or its inputs changed")
        encoder.state = serialization.from_bytes(encoder.state, encoder_checkpoint.read_bytes())
    else:
        fit_info = fit_encoder(encoder, matrix, keep, output, smoke=smoke)
        atomic_json(output / "encoder-inputs.json", inputs)
    train_names = {n.removesuffix("+ctrl") for n in prepared.train_conditions}
    val_names = {n.removesuffix("+ctrl") for n in prepared.val_conditions}
    test_names = {n.removesuffix("+ctrl") for n in test}
    if (train_names | val_names) & test_names:
        raise ValueError("held identities entered J2 scBIG training")
    bank, prior_receipt = priors(esm, train_names | val_names | test_names, train_names)
    if smoke:
        random = np.random.default_rng(SEED)
        source_controls = random.choice(controls, 300, replace=False)
        source_treatment = rows[prepared.adata.obs["condition"].iloc[rows].to_numpy() != "ctrl"][
            :1024
        ]
        subset = np.sort(np.r_[source_controls, source_treatment])
        # The smoke scores a training condition after an OT update.
        readout = str(prepared.adata.obs["condition"].iloc[source_treatment[0]])
    else:
        subset = rows
    obs_names = prepared.adata.obs["condition"].iloc[subset].astype(str).to_numpy()
    if np.array_equal(subset, rows):
        encoded = np.asarray(encoder.transform(matrix, batch_size=CELL_BATCH), dtype=np.float32)
    else:
        encoded = np.asarray(
            encoder.transform(prepared.adata.X[subset][:, order].toarray(), batch_size=CELL_BATCH),
            dtype=np.float32,
        )
    del matrix
    obs = pd.DataFrame(
        {
            "is_control": obs_names == "ctrl",
            "pert1": [n.removesuffix("+ctrl") for n in obs_names],
            "pert2": "ctrl",
            "condition": obs_names,
            "cell_line": "jurkat",
        },
        index=subset.astype(str),
    )
    flows = anndata.AnnData(encoded, obs=obs)
    flows.obsm["X_pca"] = encoded
    flows.uns["pert_embeddings"] = {g: bank[g] for g in set(obs["pert1"])}
    flows.uns["cell_line_embeddings"] = {"jurkat": np.ones(1, dtype=np.float32)}
    flow = initialize_flow(flows)
    if resume:
        if any(receipt.get(k) != v for k, v in prior_receipt.items()):
            raise ValueError("the frozen scBIG prior changed")
        state = checkpoint.read_bytes()
        flow.solver.vf_state_inference = serialization.from_bytes(
            flow.solver.vf_state_inference, state
        )
        saved, saved_tree = jax.tree_util.tree_flatten(serialization.msgpack_restore(state))
        current, current_tree = jax.tree_util.tree_flatten(
            serialization.to_state_dict(flow.solver.vf_state_inference)
        )
        if saved_tree != current_tree or any(
            x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes()
            for x, y in zip(map(np.asarray, saved), map(np.asarray, current), strict=True)
        ):
            raise ValueError("the restored scBIG flow differs from the saved inference state")
        flow.solver.is_trained = True
    else:
        labels, pathways = released.loss_priors(list(genes), order, keep, compass)
        flow_info = released.fit(
            output,
            flow,
            encoder,
            flows,
            None,
            genes,
            keep,
            labels,
            pathways,
            start + 4200,
            prepared=prepared,
            step_cap=20 if smoke else FLOW_STEPS,
        )
        receipt = {**identity, **fit_info, **flow_info, **prior_receipt, "test_updates": 0}
        atomic_json(output / "training_result.json", receipt)
    apply = released.frozen_predictor(flow, encoder, order, bank, "jurkat")
    if smoke:
        begin = time.monotonic()
        example = apply(readout, prepared.adata.X[source_controls].toarray())[:, axis]
        if example.shape != (300, len(axis)) or not np.isfinite(example).all():
            raise ValueError("scBIG source smoke did not yield 300 finite predictions")
        atomic_json(
            output / "result.json",
            {
                **receipt,
                "status": "trained_validation_only",
                "test_conditions_accessed": 0,
                "validation_prediction_shape": list(example.shape),
                "seconds_per_condition": time.monotonic() - begin,
            },
        )
        return
    result = evaluate_segment(
        lambda name, controls, _ids, columns: apply(name, controls)[:, columns],
        output,
        receipt,
        deadline=start + 10600,
    )
    if result is None:
        return
    if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("the frozen scBIG flow changed")
    atomic_json(
        output / "result.json",
        {
            **receipt,
            **result,
            "status": "evaluated",
            "evaluation_source_commit": identity["source_commit"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publication", required=True, type=Path)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--gpu", required=True, type=int, choices=(0, 1))
    parser.add_argument("--gpu-uuid", required=True)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--smoke-run", type=Path)
    phase.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.resolve().parent != STUDY or os.environ.get("CUDA_VISIBLE_DEVICES") != str(
        args.gpu
    ):
        raise ValueError("invalid J2 output or GPU")
    source = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    require_upstream(released.UPSTREAM, released.COMMIT)
    sys.path.insert(0, str(released.UPSTREAM))
    identity = {
        "model": "scbig",
        "scenario": "J2",
        "run_id": args.output.name,
        "gpu": args.gpu,
        "source_commit": source.commit,
        "official_commit": released.COMMIT,
        "cohort_sha256": COHORT_SHA,
        "formal_eligible": source.formal_eligible,
        "config_sha256": sha256_json(
            {"flow_microsteps": FLOW_STEPS, "seed": SEED, "encoder_epochs": 1}
        ),
    }
    if args.smoke_run:
        smoke = read_json(args.smoke_run / "result.json")
        if (
            any(smoke[k] != v for k, v in identity.items() if k not in {"run_id", "gpu"})
            or smoke["status"] != "trained_validation_only"
            or smoke["test_conditions_accessed"]
            or smoke["validation_prediction_shape"] != [300, 4486]
        ):
            raise ValueError("the official scBIG J2 smoke did not pass")
    with lock(SERVER / "runtime" / f"v2-gpu-{args.gpu}.lock"):
        admit_selected_idle(snapshot(os.getpid()), args.gpu, args.gpu_uuid)
        if not args.resume:
            args.output.mkdir()
        atomic_json(
            args.output / "worker.json", {"owner": process_info(os.getpid()), "gpu": args.gpu}
        )
        run(args.output, identity, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    main()
