"""Released PRESCRIBE training and frozen evaluation on Jurkat J2."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from benchmarks.crosscell.official_fit_probe import COMMITS
from benchmarks.crosscell.prescribe_c2 import (
    ASSETS,
    CONFIG,
    PRIOR_SHA,
    UPSTREAM,
    build_embedding_bank,
    fit_projection,
    run,
    wait_for_gpu,
)
from benchmarks.crosscell.remaining_c2_readiness import hgnc_links, reconcile
from benchmarks.crosscell.replay_common_c2 import require_upstream
from benchmarks.crosscell.study_paths import COHORT_SHA, REPO, STUDY
from benchmarks.jurkat_j2.study import context, fit_data, score_axis
from gradpert.data._io import read_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json


def fit_inputs():
    with context() as data:
        genes = data.expression_gene_ids
        test = tuple(data.split.test_conditions)
        panel = tuple(genes[i] for i in score_axis(genes))

    def load(cfg):
        import pickle

        prepared = fit_data(cfg["seed"])
        cols = np.asarray(prepared.allowed_indices)
        matrix = prepared.adata.X[:, cols]
        visible = pd.Index(np.asarray(genes)[cols])
        index = {str(row): i for i, row in enumerate(prepared.adata.obs_names)}
        roles = {
            key: np.asarray([index[row] for row in values])
            for key, values in (
                ("train", prepared.train_row_ids),
                ("val", prepared.val_row_ids),
                ("source_control", prepared.control_row_ids),
            )
        }
        names = prepared.adata.obs["condition"].astype(str).to_numpy()
        batches = np.asarray(prepared.row_context_ids)
        lines = np.full(len(names), "jurkat")
        train, val = prepared.train_conditions, prepared.val_conditions
        targets = {g for c in (*train, *val, *test) for g in c.split("+") if g != "ctrl"}
        path = ASSETS / "scgpt-primary.pkl"
        if sha256_file(path) != PRIOR_SHA["prescribe"]:
            raise ValueError("the released scGPT prior changed")
        with path.open("rb") as stream:
            support = reconcile(
                targets, set(pickle.load(stream)), hgnc_links(), verified_aliases={}
            )
        missing = tuple(support["unresolved"])
        if support["collisions"] or set(missing) != {"QARS", "TBCE"}:
            raise ValueError("unexpected Jurkat scGPT target coverage")
        aliases = support["aliases"]
        bank, contributors = build_embedding_bank(train, val, test, aliases, missing=missing)
        count = {}
        for role in ("train", "val"):
            count[role] = sum(names[i].split("+")[0] in missing for i in roles[role])
        return fit_projection(
            matrix,
            cols,
            visible,
            roles,
            names,
            lines,
            batches,
            train,
            aliases,
            bank,
            contributors,
            count,
            cfg,
        )

    return panel, load


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
    phase.add_argument("--replay-full", type=Path)
    args = parser.parse_args()
    if args.replay_full is not None:
        args.replay_full = args.replay_full.resolve()
        if args.replay_full.parent != STUDY or args.replay_full == args.output.resolve():
            raise ValueError("frozen replay requires a distinct original study run")
        if (args.output / "result.json").exists():
            raise FileExistsError("the full-gene replay already completed")
    if args.output.resolve().parent != STUDY or os.environ.get("CUDA_VISIBLE_DEVICES") != str(
        args.gpu
    ):
        raise ValueError("invalid J2 study path or GPU")
    provenance = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    commit = COMMITS["prescribe"]
    require_upstream(UPSTREAM, commit)
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["fit_seconds"] = 5400
    identity = {
        "model": "prescribe",
        "scenario": "J2",
        "run_id": args.output.name,
        "gpu": args.gpu,
        "source_commit": provenance.commit,
        "official_commit": commit,
        "formal_eligible": provenance.formal_eligible,
        "cohort_sha256": COHORT_SHA,
        "config_sha256": sha256_json(cfg),
    }
    if args.smoke_run:
        smoke = read_json(args.smoke_run / "result.json")
        if (
            any(smoke.get(k) != v for k, v in identity.items() if k != "run_id")
            or smoke.get("status") != "trained_validation_only"
            or smoke.get("test_conditions_accessed") != 0
            or smoke.get("validation_prediction_shape") != [300, 4486]
        ):
            raise ValueError("the official J2 smoke did not pass")
    args.output.mkdir(exist_ok=args.replay_full is not None)
    with wait_for_gpu(args.output, args.gpu, args.gpu_uuid):
        if torch.cuda.device_count() != 1:
            raise RuntimeError("PRESCRIBE J2 requires one CUDA device")
        run(
            args.output,
            identity,
            cfg,
            args.smoke,
            False,
            time.monotonic(),
            j2=True,
            replay=args.replay_full,
        )


if __name__ == "__main__":
    main()
