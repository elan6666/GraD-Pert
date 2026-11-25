"""Fit the released PerturbDiff recipe on the frozen Jurkat J2 split."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from benchmarks.common.full_gene_evaluation import replay_fields, verify_replay
from benchmarks.crosscell.official_fit_probe import COMMITS, PRIOR_SHA
from benchmarks.crosscell.perturbdiff_c2 import (
    ASSETS,
    CONFIG,
    GENEPT,
    GENEPT_SHA,
    UPSTREAM,
    build_perturbdiff,
    fit,
    groups_for,
    predict,
)
from benchmarks.crosscell.remaining_c2_readiness import hgnc_links, reconcile
from benchmarks.crosscell.replay_common_c2 import require_upstream
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


def target(condition):
    return condition.removesuffix("+ctrl")


def j2_prior(fitted, test, path):
    released = ASSETS / "replogle-prior.pkl"
    if sha256_file(released) != PRIOR_SHA["perturbdiff"] or sha256_file(GENEPT) != GENEPT_SHA:
        raise ValueError("released GenePT prior changed")
    with GENEPT.open("rb") as stream:
        external = pickle.load(stream)
    with released.open("rb") as stream:
        primary = pickle.load(stream)
    if any(not np.array_equal(primary[g], external[g]) for g in primary.keys() & external.keys()):
        raise ValueError("GenePT assets disagree on a shared symbol")
    merged = {**external, **primary}
    support = reconcile(fitted | test, set(merged), hgnc_links(), verified_aliases={})
    missing = set(support["unresolved"])
    if support["collisions"] or missing != {"C1orf109", "CCDC130", "DDN", "FRG2"} or missing & test:
        raise ValueError("unexpected Jurkat J2 GenePT coverage")
    aliases = support["aliases"]
    bank = {g: merged[aliases.get(g, g)] for g in sorted((fitted | test) - missing)}
    # The released covariate encoder zero-initializes absent GenePT keys.
    with path.open("xb") as stream:
        pickle.dump(bank, stream)
    return sorted(missing & fitted), sorted(missing & test)


def run(output, cfg, identity, smoke, fit_seconds, *, resume, replay=None, colab_prefix=None):
    started = time.monotonic()
    parent = replay or output
    checkpoint = parent / "checkpoint.pt"
    if resume:
        receipt = frozen_receipt(
            parent, {**identity, "run_id": parent.name}, checkpoint.name, replay=replay is not None
        )
    prepared = fit_data(cfg["seed"])
    adata = prepared.adata
    axis = score_axis(prepared.gene_ids)
    train_names = {target(n) for n in prepared.train_conditions}
    val_names = {target(n) for n in prepared.val_conditions}
    with context() as data:
        test_names = {target(n) for n in data.split.test_conditions}
        frozen_control_batches = dict(zip(data.row_ids, data.context_ids, strict=True))
        draw_order = [draw.condition_id for draw in data.control_manifest.draws]
        paired_rows_sha256 = data.control_manifest_file_sha256
    if (train_names | val_names) & test_names:
        raise ValueError("J2 condition identities leaked into fit")
    batches = np.asarray(prepared.row_context_ids)
    lines = np.full(len(batches), "jurkat")
    conditions = adata.obs["condition"].astype(str).to_numpy()
    treatments = np.asarray([f"{target(n)}+ctrl" if n != "ctrl" else n for n in conditions])
    index = {str(r): i for i, r in enumerate(adata.obs_names)}
    control_rows = np.asarray([index[r] for r in prepared.control_row_ids])
    controls = defaultdict(list)
    for row in control_rows:
        controls[("jurkat", batches[row])].append(row)
    rng = np.random.default_rng(cfg["seed"])
    fits = [
        groups_for(
            [index[r] for r in rows],
            treatments,
            lines,
            batches,
            rng,
            cfg["cell_set_size"],
        )
        for rows in (prepared.train_row_ids, prepared.val_row_ids)
    ]
    vocab = {
        "pert": {
            g: i
            for i, g in enumerate(sorted(train_names | val_names | test_names | {"non-targeting"}))
        },
        "cell_line": {"jurkat": 0},
        "batch": {
            g: i for i, g in enumerate(sorted(controls_name for _, controls_name in controls))
        },
    }
    with tempfile.TemporaryDirectory(dir=output, prefix="official-prior-") as directory:
        prior = Path(directory) / "replogle.pkl"
        missing_fit, missing_test = j2_prior(train_names | val_names, test_names, prior)
        model, _, _, _ = build_perturbdiff(
            UPSTREAM,
            prior,
            vocabulary=vocab,
            genes=adata.n_vars,
            cell_prior=ASSETS / "celltype.pkl",
            max_steps=cfg["max_updates"],
        )
        if not resume:
            n, exposed, loss = fit(
                adata,
                *fits,
                controls,
                list(prepared.allowed_indices),
                model,
                vocab,
                cfg,
                output,
                cfg["smoke_updates"] if smoke else cfg["max_updates"],
                started + fit_seconds,
            )
    if resume:
        if (missing_fit, missing_test) != (
            receipt["missing_fit_targets"],
            receipt["missing_test_targets"],
        ):
            raise ValueError("the frozen PerturbDiff prior coverage changed")
    else:
        torch.save(model.state_dict(), checkpoint)
        receipt = {
            **identity,
            "optimizer_updates": n,
            "fit_rows_exposed": exposed,
            "validation_loss": loss,
            "checkpoint_sha256": sha256_file(checkpoint),
            "missing_fit_targets": missing_fit,
            "missing_test_targets": missing_test,
            "missing_genept_policy": "released_zero_embedding",
            "fit_seconds": time.monotonic() - started,
            "test_updates": 0,
        }
        atomic_json(output / "training_result.json", receipt)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    if smoke:
        rng = np.random.default_rng(cfg["seed"] + 1)
        sample = rng.choice(control_rows, 300, replace=False)
        begin = time.monotonic()
        values = predict(
            model,
            vocab,
            cfg,
            fits[1][0][0],
            "jurkat",
            adata.X[sample].toarray(),
            batches[sample],
        )
        if not np.isfinite(values).all() or values.shape != (300, adata.n_vars):
            raise ValueError("source-only smoke did not return 300 finite predictions")
        atomic_json(
            output / "result.json",
            {
                **receipt,
                "status": "trained_validation_only",
                "test_conditions_accessed": 0,
                "validation_prediction_shape": [300, len(axis)],
                "seconds_per_condition": time.monotonic() - begin,
            },
        )
        return

    means, cloud_receipt = {}, {}
    prediction_identity = None
    if colab_prefix is not None:
        from benchmarks.jurkat_j2.colab_perturbdiff import load_condition_means

        input_manifest = colab_prefix / "manifest.json"
        if read_json(input_manifest)["paired_rows_sha256"] != paired_rows_sha256:
            raise ValueError("Colab and the server use different frozen control manifests")
        means, cloud_receipt = load_condition_means(
            input_manifest, colab_prefix, replay, draw_order
        )
        prediction_identity = sha256_json(
            [receipt["checkpoint_sha256"], cloud_receipt["predictions_sha256"]]
        )

    def apply(name, control, ids, columns):
        if name in means:
            return np.broadcast_to(means[name][columns], control.shape)
        torch.manual_seed(int(sha256_json([cfg["evaluation_seed"], name])[:8], 16))
        control_contexts = np.asarray([frozen_control_batches[r] for r in ids])
        return predict(model, vocab, cfg, target(name), "jurkat", control, control_contexts)[
            :, columns
        ]

    result = evaluate_segment(
        apply,
        output,
        receipt,
        deadline=started + cfg["max_wall_seconds"] - 60,
        full_axis=replay is not None,
        prediction_identity=prediction_identity,
    )
    if result is None:
        return
    if sha256_file(checkpoint) != receipt["checkpoint_sha256"] or list(output.rglob("*.pkl")):
        raise ValueError("checkpoint or metrics-only output changed")
    atomic_json(
        output / "result.json",
        {
            **receipt,
            **result,
            "status": "evaluated",
            "evaluation_source_commit": identity["source_commit"],
            **(
                {
                    "inference_provider": "Google Colab and campus GPU",
                    "colab_inference": cloud_receipt,
                    "campus_inference_conditions": 592 - len(means),
                }
                if cloud_receipt
                else {}
            ),
            **(replay_fields(replay, identity) if replay is not None else {}),
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
    phase.add_argument("--replay-full", type=Path)
    parser.add_argument("--colab-prefix", type=Path)
    args = parser.parse_args()
    if args.output.resolve().parent != STUDY or os.environ.get("CUDA_VISIBLE_DEVICES") != str(
        args.gpu
    ):
        raise ValueError("invalid J2 study path or GPU")
    source = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    commit = COMMITS["perturbdiff"]
    require_upstream(UPSTREAM, commit)
    sys.path.insert(0, str(UPSTREAM))
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg.update(expression_genes=5000, max_wall_seconds=10800)
    identity = {
        "model": "perturbdiff",
        "scenario": "J2",
        "run_id": args.output.name,
        "gpu": args.gpu,
        "source_commit": source.commit,
        "official_commit": commit,
        "cohort_sha256": COHORT_SHA,
        "config_sha256": sha256_json(cfg),
        "formal_eligible": source.formal_eligible,
    }
    fit_seconds = cfg["max_fit_seconds"]
    if args.smoke_run:
        smoke = read_json(args.smoke_run / "result.json")
        if (
            any(smoke.get(k) != v for k, v in identity.items() if k not in {"run_id", "gpu"})
            or smoke.get("status") != "trained_validation_only"
            or smoke.get("test_conditions_accessed") != 0
        ):
            raise ValueError("the official J2 smoke did not pass")
        fit_seconds = max(
            180,
            min(
                fit_seconds,
                cfg["max_wall_seconds"] - 592 * smoke["seconds_per_condition"] * 1.10 - 180,
            ),
        )
    if args.colab_prefix is not None and (
        args.replay_full is None or args.colab_prefix.resolve().parent != STUDY
    ):
        raise ValueError("a Colab prefix must belong to a full replay in the sealed study")
    if args.replay_full is not None:
        args.replay_full = args.replay_full.resolve()
        if args.replay_full.parent != STUDY or args.replay_full == args.output.resolve():
            raise ValueError("frozen replay requires a distinct original study run")
        verify_replay(args.replay_full, identity, "checkpoint.pt", conditions=592)
        if (args.output / "result.json").exists():
            raise FileExistsError("the full-gene replay already completed")
    with lock(SERVER / "runtime" / f"v2-gpu-{args.gpu}.lock"):
        admit_selected_idle(snapshot(os.getpid()), args.gpu, args.gpu_uuid)
        torch.manual_seed(cfg["seed"])
        torch.cuda.set_per_process_memory_fraction(0.90)
        if not args.resume:
            args.output.mkdir(exist_ok=args.replay_full is not None)
        atomic_json(
            args.output / "worker.json", {"owner": process_info(os.getpid()), "gpu": args.gpu}
        )
        run(
            args.output,
            cfg,
            identity,
            args.smoke,
            fit_seconds,
            resume=args.resume or args.replay_full is not None,
            replay=args.replay_full,
            colab_prefix=args.colab_prefix,
        )


if __name__ == "__main__":
    main()
