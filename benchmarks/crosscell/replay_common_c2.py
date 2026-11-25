"""Re-score frozen C2 checkpoints on the full or historical common gene axis."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from benchmarks.common import official_module_session
from benchmarks.crosscell.common_score_panel import PANEL, load_panel
from benchmarks.crosscell.official_fit_probe import SUPPORT_SHA
from benchmarks.crosscell.study_data import cohort_inputs, score
from benchmarks.crosscell.study_paths import (
    COHORT_SHA,
    REPO,
    SERVER,
    STUDY,
    TXPERT_CHECKOUT,
    TXPERT_COMMIT,
)
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file
from scripts.v2.shared_gpu import process_info, snapshot

SUPPORT = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
ORIGINAL = {
    "txpert": (
        "txpert-crosscell-c2-released-scratch-full-a2",
        "best.ckpt",
        "44a8643cf9e95797699228ece3bbb0949225b76d97046a8402adcabb5d2a71d9",
    ),
    "perturbdiff": (
        "perturbdiff-crosscell-c2-budget3h-formal-a1",
        "checkpoint.pt",
        "6736b3944c65d6dab21cd040f755561d6c761a4c44e3e24e8ed0f8ddbbffee28",
    ),
    "scbig": (
        "scbig-crosscell-c2-budget3h-formal-a1",
        "flow_inference.msgpack",
        "4f2839770b0c60a545c2653f816e9764ab5b5f3f935ea469be7c6647fedab474",
    ),
}


def require_upstream(checkout, expected):
    git = ["git", "-C", str(checkout)]
    if (
        subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != expected
        or subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
    ):
        raise ValueError(f"released checkout changed: {checkout.name}")


def perturbdiff_predictor(checkpoint, receipt, source, cohort, output, *, latent_readout=False):
    import anndata
    import torch
    import yaml

    from benchmarks.crosscell import perturbdiff_c2 as released

    require_upstream(released.UPSTREAM, receipt["official_commit"])
    if sha256_file(released.CONFIG) != receipt["config_sha256"]:
        raise ValueError("PerturbDiff frozen configuration changed")
    sys.path.insert(0, str(released.UPSTREAM))
    cfg = yaml.safe_load(released.CONFIG.read_text())
    vocab = read_json(checkpoint.parent / "vocabulary.json")
    identities = {x.removesuffix("+ctrl") for x in cohort["condition_ids"]}
    fitted = set(vocab["pert"]) - identities - {"non-targeting"}
    data = anndata.read_h5ad(source, backed="r")
    try:
        batches = data.obs["batch"].astype(str).to_numpy()
        num_genes = data.n_vars
    finally:
        data.file.close()
    with tempfile.TemporaryDirectory(dir=output, prefix="official-prior-") as tmp:
        prior = Path(tmp) / "replogle.pkl"
        if (
            released.load_prior(
                vocab["pert"], fitted, identities, prior, cfg["allowed_missing_fit_target"]
            )
            != receipt["prior_sha256"]
        ):
            raise ValueError("PerturbDiff frozen vocabulary or gene prior changed")
        model, _, _, _ = released.build_perturbdiff(
            released.UPSTREAM,
            prior,
            vocabulary=vocab,
            genes=num_genes,
            cell_prior=released.ASSETS / "celltype.pkl",
            max_steps=cfg["max_updates"],
        )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)

    @torch.inference_mode()
    def inference(condition, controls, control_rows):
        if latent_readout:
            covariates = {
                "cov_pert": vocab["pert"][condition.removesuffix("+ctrl")],
                "cov_celltype": vocab["cell_line"]["k562"],
                "cov_batch": vocab["batch"][released.UNSEEN_BATCH],
            }
            encoded = model._encode_covariates(
                {
                    field: torch.full((1, 1), index, device="cuda")
                    for field, index in covariates.items()
                }
            )
            return encoded.detach().cpu().numpy().reshape(1, -1)
        torch.manual_seed(cfg["evaluation_seed"] + cohort["condition_ids"].index(condition))
        return released.predict(
            model,
            vocab,
            cfg,
            condition.removesuffix("+ctrl"),
            "k562",
            controls,
            batches[control_rows],
            unseen_batch=True,
        )

    return inference


def scbig_predictor(checkpoint, receipt, cross, cohort, *, latent_readout=False):
    import jax
    import numpy as np
    from flax import serialization

    from benchmarks.crosscell import scbig_c2 as released

    require_upstream(released.UPSTREAM, receipt["official_commit"])
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("scBIG requires its pinned CUDA environment")
    sys.path.insert(0, str(released.UPSTREAM))
    prefix = released.verify_prefix()
    compass, esm = released.verify_assets()
    encoder, inputs = released.restore_stage1(compass)
    training, aliases = released.prepare_source_latents(encoder, inputs, esm)
    if aliases != prefix["esm2_resolved_aliases"] or training.n_obs != prefix["source_rows"]:
        raise ValueError("the frozen scBIG encoder/context differs from the original")
    flow = released.initialize_flow(training)
    state = checkpoint.read_bytes()
    flow.solver.vf_state_inference = serialization.from_bytes(flow.solver.vf_state_inference, state)
    # Flax stores parameter mappings in insertion order. A fresh official
    # template can deserialize identical leaves under a different key order.
    state_dict = serialization.msgpack_restore(state)
    restored = serialization.to_state_dict(flow.solver.vf_state_inference)
    lhs, lhs_tree = jax.tree_util.tree_flatten(state_dict)
    rhs, rhs_tree = jax.tree_util.tree_flatten(restored)
    if lhs_tree != rhs_tree:
        raise ValueError("scBIG inference state tree changed during checkpoint restore")
    for index, (saved, loaded) in enumerate(zip(lhs, rhs, strict=True)):
        x, y = np.asarray(saved), np.asarray(loaded)
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            raise ValueError(f"scBIG state leaf {index} changed during checkpoint restore")
    flow.solver.is_trained = True
    prior = released.test_prior(esm, cohort["condition_ids"])
    return released.frozen_predictor(
        flow,
        encoder,
        inputs["grc_indices"],
        prior,
        cross["target_cell_line"],
        latent_readout=latent_readout,
    )


def run(args, source_identity, *, shared=None):
    model = args.model
    output = args.output
    origin, filename, digest = ORIGINAL[model]
    checkpoint = STUDY / origin / filename
    manifest = checkpoint.parent / "result.json"
    if sha256_file(manifest) != digest or sha256_file(SUPPORT) != SUPPORT_SHA:
        raise ValueError("the accepted C2 fit or manifest changed")
    receipt = read_json(manifest)
    required = dict(
        status="evaluated",
        formal_eligible=True,
        scenario="C2",
        conditions=128,
        test_controls_per_condition=300,
        test_updates=0,
        cohort_sha256=COHORT_SHA,
        shared_fit_support_file_sha256=SUPPORT_SHA,
        checkpoint_sha256=sha256_file(checkpoint),
    )
    if any(receipt.get(k) != value for k, value in required.items()):
        raise ValueError("original result does not match its frozen fit")
    started = time.monotonic()
    if model == "txpert":
        from benchmarks.txpert.crosscell_c2 import fit
        from benchmarks.txpert.runtime import inspect_cuda_runtime, load_runtime_contract

        _, contract = load_runtime_contract(repository_root=REPO, checkout_root=TXPERT_CHECKOUT)
        inspect_cuda_runtime(contract=contract, device="cuda:0")
        args.fit_compatibility = SUPPORT
        args.fit_compatibility_sha = SUPPORT_SHA
        scored = fit(args, output, source_identity, replay_checkpoint=checkpoint)
    else:
        source, cross, cohort, rows = cohort_inputs()
        panel = None if args.full_axis else load_panel(source, cross, cohort)
        if model == "perturbdiff":
            inference = perturbdiff_predictor(checkpoint, receipt, source, cohort, output)
        else:
            predict_scbig = scbig_predictor(checkpoint, receipt, cross, cohort)
            inference = predict_scbig
            if shared is not None:

                def inference(condition, controls):
                    shared.check()
                    started_condition = time.monotonic()
                    prediction = predict_scbig(condition, controls)
                    shared.finish_batch(started_condition)
                    return prediction

        with official_module_session(
            checkout_root=TXPERT_CHECKOUT,
            expected_commit=TXPERT_COMMIT,
            module_names=("gspp.metrics",),
        ) as (modules, _):
            scored = score(
                inference,
                modules["gspp.metrics"],
                source,
                cross,
                cohort,
                rows,
                read_json(SUPPORT),
                output,
                journal_key=receipt["checkpoint_sha256"],
                score_genes=panel["score_genes"] if panel is not None else None,
                full_axis=args.full_axis,
                include_control_row_ids=model == "perturbdiff",
                stop_starting_conditions_at=started + 10800,
            )
    if (
        scored.get("conditions") != required["conditions"]
        or scored.get("test_panel_genes") != (3352 if args.full_axis else 3194)
        or scored["paired_rows_sha256"] != receipt["paired_rows_sha256"]
        or sha256_file(checkpoint) != receipt["checkpoint_sha256"]
        or sha256_file(manifest) != digest
    ):
        raise ValueError("frozen replay did not complete with the original checkpoint and cohort")
    atomic_json(
        output / "result.json",
        {
            **scored,
            **required,
            "run_id": output.name,
            "model": model,
            "source_commit": receipt["source_commit"],
            "evaluation_source_commit": source_identity.commit,
            "original_run_id": origin,
            "original_receipt_sha256": digest,
            **({"score_panel_sha256": sha256_file(PANEL)} if not args.full_axis else {}),
            "replay_optimizer_updates": 0,
            "evaluation_seconds": time.monotonic() - started,
            **({"shared_gpu_evidence": shared.evidence()} if shared is not None else {}),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=ORIGINAL)
    parser.add_argument("--full-axis", action="store_true")
    parser.add_argument("--shared-gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if args.shared_gpu and (args.model != "scbig" or not args.full_axis):
        parser.error("shared CUDA admission applies only to full-gene scBIG replay")
    args.output = args.output.resolve()
    if (args.output / "result.json").exists():
        raise FileExistsError("the frozen C2 replay already completed")
    if args.output.parent != STUDY or any(
        os.getenv(k) != v
        for k, v in {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }.items()
    ):
        raise ValueError("invalid provenance or CUDA admission settings")
    identity = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        verification_method="git_ls_remote_live",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    with lock(SERVER / "runtime" / f"v2-gpu-{args.gpu}.lock"):
        shared = None
        if args.shared_gpu:
            from benchmarks.crosscell.prescribe_gpu import scbig_shared_gpu

            shared = scbig_shared_gpu(args.gpu, args.gpu_uuid)
            shared.admit()
            shared.configure_jax_allocator()
        else:
            admit_selected_idle(snapshot(os.getpid()), args.gpu, args.gpu_uuid)
        args.output.mkdir(exist_ok=args.full_axis)
        atomic_json(
            args.output / "worker.json", {"owner": process_info(os.getpid()), "gpu": args.gpu}
        )
        run(args, identity, shared=shared)


if __name__ == "__main__":
    main()
