"""Read the five frozen C2 baselines' declared internal states, without fitting."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import anndata
import numpy as np

from benchmarks.common.study_fit_partition import crosscell_fit_rows
from benchmarks.crosscell.common_score_panel import PANEL, load_panel
from benchmarks.crosscell.replay_common_c2 import (
    ORIGINAL,
    SUPPORT,
    SUPPORT_SHA,
    perturbdiff_predictor,
    require_upstream,
    scbig_predictor,
)
from benchmarks.crosscell.study_data import cohort_inputs
from benchmarks.crosscell.study_paths import COHORT_SHA, REPO, SERVER, STUDY
from gradpert.data._io import atomic_json, read_json
from gradpert.evaluation.crosscell import read_rows
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot

ANCHORS = 12
RUNS = {
    **ORIGINAL,
    "prescribe": (
        "prescribe-crosscell-c2-budget3h-shared-formal-a2",
        "best.ckpt",
        "38eae7ea819b573bde185257a19a376395c35459131b081ee82ce27bd2128164",
    ),
    "scfoundation": ("scfoundation-crosscell-c2-budget3h-formal-a1", "checkpoint.pt", None),
}
LAYERS = {
    "txpert": "TxPert decoder input after intrinsic and learned perturbation composition",
    "perturbdiff": "PerturbDiff covariate encoder output for the unseen K562 context",
    "scbig": "scBIG post-flow latent before inverse expression projection",
    "prescribe": "PRESCRIBE conditioning encoder before the posterior",
    "scfoundation": "scFoundation/GEARS cross_gene_state before the final expression decoder",
}


def anchors_for(source, cross, cohort, rows, output, *, full_context=False):
    draw_path = Path(cohort["shared_frozen_draws_path"])
    if sha256_file(draw_path) != cross["shared_frozen_draws_sha256"]:
        raise ValueError("draw manifest changed")
    raw = read_json(draw_path)["draws"]
    draws = {c: raw[c] for c in cohort["condition_ids"]}
    if sha256_json(draws) != cohort["per_condition_pairing_sha256"]:
        raise ValueError("frozen pairing changed")
    panel = load_panel(source, cross, cohort)
    selection, data = {}, {}
    control_ids = rows["target_control"]
    adata = anndata.read_h5ad(source, backed="r")
    try:
        cols = adata.var_names.get_indexer(panel["score_genes"])
        batch = adata.obs["batch"].astype(str).to_numpy()
        for name, draw in draws.items():
            controls, truth = map(np.asarray, (draw["input_control_rows"], draw["truth_rows"]))
            if len(controls) != 300 or not np.isin(controls, control_ids).all():
                raise ValueError("diagnostic controls must follow the frozen 300-cell draw")
            matching = control_ids[np.isin(batch[control_ids], batch[truth])]
            ctrl_mean = np.asarray(adata.X[matching].mean(axis=0)).ravel()[cols]
            observed = read_rows(adata, truth)[:, cols].mean(0, dtype=np.float64) - ctrl_mean
            selection[name] = controls[:ANCHORS].tolist()
            model_rows = controls if full_context else controls[:ANCHORS]
            data[name] = (read_rows(adata, model_rows), model_rows, observed)
    finally:
        adata.file.close()
    atomic_json(output / "anchor_rows.json", selection)
    return data, sha256_json(selection), cols


def pool_output(predict, module, *, before=False):
    """Collect emitted states only. The hook does not alter released inference."""
    values = []
    handle = (
        module.register_forward_pre_hook(lambda _module, inputs: values.append(inputs[0].detach()))
        if before
        else module.register_forward_hook(lambda _module, _inputs, out: values.append(out.detach()))
    )
    try:
        predict()
    finally:
        handle.remove()
    if not values or values[0].ndim != 2:
        raise ValueError("the declared layer did not emit cell-by-feature states")
    return values[0].cpu().numpy()[:ANCHORS]


def prescribe_readout(checkpoint, details, source, cohort, rows, output, device):
    import torch

    from benchmarks.crosscell import prescribe_c2 as released

    require_upstream(released.UPSTREAM, details["official_commit"])
    reference = checkpoint.parent / "fit_projection.npz"
    if (
        sha256_file(reference) != details["fit_projection_sha256"]
        or sha256_file(released.CONFIG) != details["config_sha256"]
        or sha256_file(released.AUDIT) != released.AUDIT_SHA
    ):
        raise ValueError("PRESCRIBE released config, aliases, or projection changed")
    with np.load(reference, allow_pickle=False) as projection:
        cols = projection["visible_indices"]
    data = anndata.read_h5ad(source, backed="r")
    try:
        names = data.obs["condition"].astype(str).to_numpy()
        roles = crosscell_fit_rows(rows, names, cohort, compatibility=read_json(SUPPORT))
        train, val = [sorted(set(names[roles[role]])) for role in ("train", "val")]
    finally:
        data.file.close()
    aliases = details["target_aliases"]
    bank, supported = released.build_embedding_bank(train, val, cohort["condition_ids"], aliases)
    if sha256_json(supported) != details["prior_contributors_sha256"]:
        raise ValueError("PRESCRIBE fitted embedding contributors changed")
    sys.path[:0] = [str(released.UPSTREAM), str(released.ASSETS / "python-pyblaze-2.3.2")]
    import yaml
    from src.nn.encoder import Encoder

    cfg = yaml.safe_load(released.CONFIG.read_text())
    with tempfile.TemporaryDirectory(dir=output) as directory:
        staging = Path(directory)
        where = staging / "scLLM_weights/scGPT"
        where.mkdir(parents=True)
        with (where / "embedding.pkl").open("xb") as stream:
            pickle.dump(bank, stream)
        try:
            os.chdir(staging)
            encoder = Encoder(latent_dim=cfg["latent_dim"], backbone="scGPT", pert_num=len(cols))
        finally:
            os.chdir(REPO)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")["state_dict"]
    prefix = "model.encoder."
    encoder.load_state_dict(
        {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}, strict=True
    )
    encoder.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(name, controls):
        batch = released.prepare_batch(controls[:, cols], [name] * len(controls), aliases)
        return encoder(batch.x.to(device), batch.pert).cpu().numpy()

    return predict


def scfoundation_readout(checkpoint, source, cross, cohort, rows, controls_projection, output):
    import torch

    from benchmarks.crosscell import scfoundation_c2 as released

    panel = load_panel(source, cross, cohort)
    gears, positions, targets, _, _ = released.initialize(
        output, source, rows, cohort, read_json(SUPPORT), panel, 5, 2
    )
    model = gears.model.cuda().eval().requires_grad_(False)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )

    @torch.inference_mode()
    def predict(name, controls):
        values = controls[:, controls_projection]
        states = []
        # Match the official six-cell inference microbatch.
        for i in range(0, len(values), released.BATCH):
            chunk = values[i : i + released.BATCH]
            batch = released.Batch(
                chunk, positions, [targets[name]] * len(chunk), [name] * len(chunk)
            )
            states.append(
                pool_output(lambda batch=batch: model(batch.to("cuda")), model.cross_gene_state)
            )
        return np.concatenate(states)

    return predict


def collect(
    input_rows: dict,
    predictor: Callable,
    output: Path,
    *,
    expression_context: bool = True,
) -> np.ndarray:
    features = []
    for i, (name, (control, rows, _)) in enumerate(input_rows.items(), start=1):
        latent = np.asarray(predictor(name, control, rows), dtype=np.float64)
        if latent.ndim != 2 or latent.shape[0] != (ANCHORS if expression_context else 1):
            raise ValueError(f"unexpected representation shape for {name}: {latent.shape}")
        if not np.isfinite(latent).all():
            raise ValueError(f"nonfinite representation: {name}")
        features.append(latent.mean(0))
        if i == 1 or i % 8 == 0:
            atomic_json(
                output / "representation_progress.json", {"done": i, "total": len(input_rows)}
            )
    return np.stack(features)


def run(args, identity):
    model, output = args.model, args.output
    origin, checkpoint_name, expected = RUNS[model]
    origin_result = STUDY / origin / "result.json"
    checkpoint = origin_result.parent / checkpoint_name
    expected = expected or args.baseline_receipt_sha
    if (
        not expected
        or sha256_file(origin_result) != expected
        or sha256_file(SUPPORT) != SUPPORT_SHA
    ):
        raise ValueError("the accepted baseline receipt or support changed")
    receipt = read_json(origin_result)
    checkpoint_sha = sha256_file(checkpoint)
    scope = {
        "scenario": "C2",
        "cohort_sha256": COHORT_SHA,
        "shared_fit_support_file_sha256": SUPPORT_SHA,
        "status": "evaluated",
        "checkpoint_sha256": checkpoint_sha,
        "test_updates": 0,
    }
    if any(receipt.get(k) != v for k, v in scope.items()):
        raise ValueError("baseline checkpoint provenance did not verify")
    attempt = "a2" if model == "scbig" else "a1"
    common = STUDY / f"{model}-crosscell-c2-common3194-frozen-{attempt}/result.json"
    common = common if model in ORIGINAL else origin_result
    score_result = read_json(common)
    if (
        score_result["checkpoint_sha256"] != checkpoint_sha
        or score_result["conditions"] != 128
        or score_result["test_panel_genes"] != 3194
        or score_result["status"] != "evaluated"
        or score_result["score_panel_sha256"] != sha256_file(PANEL)
    ):
        raise ValueError("no matching common-panel performance receipt")
    source, cross, cohort, rows = cohort_inputs()
    anchors, anchor_sha, columns = anchors_for(
        source, cross, cohort, rows, output, full_context=model == "txpert"
    )

    if model == "txpert":
        from benchmarks.txpert.crosscell_c2 import fit

        args.fit_compatibility, args.fit_compatibility_sha = SUPPORT, SUPPORT_SHA

        def read_representation(trained, exact_predict):
            return collect(
                anchors,
                lambda condition, controls, _rows: pool_output(
                    lambda: exact_predict(condition, controls), trained.model.decoder, before=True
                ),
                output,
            )

        features = fit(
            args,
            output,
            identity,
            replay_checkpoint=checkpoint,
            read_representation=read_representation,
        )
    else:
        if model == "perturbdiff":
            apply = perturbdiff_predictor(
                checkpoint, receipt, source, cohort, output, latent_readout=True
            )
        else:
            if model == "scbig":
                forward = scbig_predictor(checkpoint, receipt, cross, cohort, latent_readout=True)
            elif model == "prescribe":
                forward = prescribe_readout(
                    checkpoint, receipt, source, cohort, rows, output, "cpu" if args.cpu else "cuda"
                )
            else:
                forward = scfoundation_readout(
                    checkpoint, source, cross, cohort, rows, columns, output
                )

            def apply(condition, controls, _rows):
                return forward(condition, controls[:ANCHORS])

        features = collect(anchors, apply, output, expression_context=model != "perturbdiff")
    if sha256_file(checkpoint) != checkpoint_sha or sha256_file(origin_result) != expected:
        raise ValueError("accepted frozen checkpoint or provenance changed")
    destination = output / "condition-representations.npz"
    np.savez_compressed(
        destination,
        condition_ids=np.asarray(list(anchors)),
        representations=features,
        observed_delta=np.stack([x[2] for x in anchors.values()]),
    )
    atomic_json(
        output / "result.json",
        {
            **scope,
            "status": "representations_complete",
            "run_id": output.name,
            "model": model,
            "source_commit": identity.commit,
            "source_checkpoint_receipt_sha256": expected,
            "performance_receipt": str(common),
            "performance_receipt_sha256": sha256_file(common),
            "paired_rows_sha256": score_result["paired_rows_sha256"],
            "score_panel_sha256": score_result["score_panel_sha256"],
            "shapes": list(features.shape),
            "expression_context": model != "perturbdiff",
            "anchor_rows_sha256": anchor_sha,
            "analysis_layer": LAYERS[model],
            "representation_sha256": sha256_file(destination),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=RUNS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--baseline-receipt-sha")
    runtime = parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument("--gpu", type=int, choices=(0, 1))
    runtime.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.parent != STUDY:
        raise ValueError("representation outputs must stay in the agreed study")
    if args.cpu:
        if args.model != "prescribe" or args.gpu_uuid:
            parser.error("only the PRESCRIBE encoder has a CPU diagnostic route")
    elif not args.gpu_uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("representation inference requires the agreed CUDA card")
    identity = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        verification_method="git_ls_remote_live",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    permit = nullcontext() if args.cpu else lock(SERVER / "runtime" / f"v2-gpu-{args.gpu}.lock")
    with permit:
        if not args.cpu:
            admit_selected_idle(snapshot(os.getpid()), args.gpu, args.gpu_uuid)
        args.output.mkdir()
        atomic_json(
            args.output / "worker.json", {"owner": process_info(os.getpid()), "gpu": args.gpu}
        )
        run(args, identity)


if __name__ == "__main__":
    main()
