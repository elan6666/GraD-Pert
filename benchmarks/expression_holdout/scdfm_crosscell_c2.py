"""Run upstream scDFM on the shared, frozen Cross-cell C2 partition."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from benchmarks.common import official_module_session
from benchmarks.common.study_fit_partition import open_crosscell_fit_partition
from benchmarks.crosscell.study_data import cohort_inputs, score
from benchmarks.crosscell.study_paths import COHORT_SHA, REPO, SERVER, STUDY
from benchmarks.crosscell.study_paths import TXPERT_CHECKOUT as UPSTREAM
from benchmarks.crosscell.study_paths import TXPERT_COMMIT as UPSTREAM_COMMIT
from benchmarks.expression_holdout import scdfm
from benchmarks.expression_holdout.runner import restore_last
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file
from scripts.v2.shared_gpu import process_info, snapshot

CHECKOUT = SERVER / "upstreams/scdfm-8de47e7"
CONFIG = REPO / "configs/studies/scdfm-crosscell-c2/scdfm/crosscell_k562.yaml"
SMOKE_UPDATES = 10
GPU_UUID = "GPU-23bb1466-b82b-982d-8eaf-9a0002c6a4e7"


def execute(args, identity):
    source, cross, cohort, rows = cohort_inputs()
    fit_population = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
    if sha256_file(fit_population) != args.fit_support_sha256:
        raise ValueError("shared C2 fit support differs from the queue plan")
    support = read_json(fit_population)
    if support["source_cohort_sha256"] != COHORT_SHA:
        raise ValueError("shared C2 fit support belongs to another cohort")
    config = yaml.safe_load(CONFIG.read_text())
    if any(config["training"][key] != value for key, value in scdfm.OFFICIAL_RECIPE.items()):
        raise ValueError("full-run recipe differs from upstream run.sh")
    run_identity = {
        "run_id": args.output.name,
        "scenario": "C2",
        "formal_eligible": identity.formal_eligible,
        "source_commit": identity.commit,
        "config_sha256": sha256_file(CONFIG),
        "official_commit": scdfm.OFFICIAL_SHA,
        "shared_fit_support_file_sha256": args.fit_support_sha256,
        "cohort_sha256": COHORT_SHA,
        "initialization": "from_scratch",
        "torch_version": str(torch.__version__),
    }
    if args.smoke_run:
        smoke_result = read_json(args.smoke_run / "result.json")
        if (
            smoke_result.get("status") != "trained_validation_only"
            or smoke_result.get("completed_updates") != SMOKE_UPDATES
            or any(smoke_result.get(k) != v for k, v in run_identity.items() if k != "run_id")
        ):
            raise ValueError("formal scDFM run requires the accepted smoke on identical inputs")
    prepared = open_crosscell_fit_partition(
        source=source,
        verified_rows=rows,
        cohort=cohort,
        mask_expression=True,
        fit_compatibility=support,
    )
    atomic_json(args.output / "double_holdout_fit.json", prepared.receipt)
    runtime = {"official_checkout": str(CHECKOUT), "device": "cuda:0", "seed": config["seed"]}
    fitted = scdfm.fit(
        prepared,
        config,
        runtime,
        args.output / "fit",
        max_updates=SMOKE_UPDATES if args.smoke else None,
    )
    atomic_json(args.output / "fit_receipt.json", fitted.receipt)
    checkpoint = args.output / "fit/last.pt"
    restore_last(fitted, "scdfm", checkpoint)
    result = {
        **run_identity,
        "checkpoint_role": "last",
        "checkpoint_sha256": sha256_file(checkpoint),
        "planned_updates": fitted.receipt["total_optimizer_updates"],
        "completed_updates": fitted.receipt["completed_updates"],
        "update_elapsed_seconds": fitted.receipt["updates_elapsed_seconds"],
        "steps_per_equivalent_epoch": fitted.receipt["steps_per_epoch"],
    }
    options = {
        "cell_batch": config["evaluation"]["cell_batch"],
        "query_count": config["evaluation"]["query_count"],
        "seed": config["seed"],
    }

    def predict(condition, controls):
        targets = tuple(part for part in condition.split("+") if part != "ctrl")
        return scdfm.predict(fitted, controls, targets, prepared.gene_ids, **options)

    if args.smoke:
        # Exercise checkpoint reload and the full output axis with source-validation
        # controls only. The held-out K562 expression is never opened by a smoke.
        val_id = prepared.val_row_ids[0]
        pool = prepared.val_control_indices[val_id]
        control_count = config["evaluation"]["control_cells"]
        positions = np.random.default_rng(config["seed"]).choice(pool, size=control_count)
        controls = prepared.adata.X[positions].toarray()
        condition = str(prepared.adata.obs.loc[val_id, "condition"])
        started = time.monotonic()
        predicted = predict(condition, controls)
        return {
            **result,
            "status": "trained_validation_only",
            "validation_control_rows": control_count,
            "validation_prediction_shape": list(predicted.shape),
            "validation_inference_seconds": time.monotonic() - started,
            "test_conditions_accessed": 0,
        }

    with official_module_session(
        checkout_root=UPSTREAM,
        expected_commit=UPSTREAM_COMMIT,
        module_names=("gspp.metrics",),
    ) as (modules, _):
        metrics = score(
            predict, modules["gspp.metrics"], source, cross, cohort, rows, support, args.output
        )
    return {**result, **metrics, "status": "evaluated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-support-sha256", required=True)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--smoke", action="store_true")
    stage.add_argument("--smoke-run", type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(STUDY) or any(
        os.getenv(key) != value
        for key, value in {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }.items()
    ):
        raise ValueError("scDFM C2 requires the study server and sealed CUDA envelope")
    identity = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    with lock(SERVER / "runtime" / "v2-gpu-0.lock"):
        admit_selected_idle(snapshot(os.getpid()), 0, GPU_UUID)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        args.output.mkdir(exist_ok=False)
        atomic_json(
            args.output / "owner.json", {"owner": process_info(os.getpid()), "physical_gpu": 0}
        )
        try:
            result = execute(args, identity)
        except Exception as error:
            atomic_json(
                args.output / "failure.json", {"type": type(error).__name__, "message": str(error)}
            )
            raise
        atomic_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
