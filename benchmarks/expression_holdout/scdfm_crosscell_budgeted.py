"""Queue entry for the time-capped scDFM C2 fit and bounded frozen evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from benchmarks.common import official_module_session
from benchmarks.common.study_fit_partition import open_crosscell_fit_partition
from benchmarks.crosscell.study_data import cohort_inputs, score
from benchmarks.crosscell.study_paths import COHORT_SHA, REPO, SERVER, STUDY
from benchmarks.crosscell.study_paths import TXPERT_CHECKOUT as UPSTREAM
from benchmarks.crosscell.study_paths import TXPERT_COMMIT as UPSTREAM_COMMIT
from benchmarks.expression_holdout import scdfm
from benchmarks.expression_holdout.contract import FittedAdapter
from benchmarks.expression_holdout.runner import restore_last
from benchmarks.expression_holdout.scdfm_crosscell_c2 import CHECKOUT, GPU_UUID
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot

CONFIG = REPO / "configs/studies/scdfm-crosscell-c2-3h/scdfm/crosscell_k562.yaml"
SMOKE_RESULT_SHA = "a2d20b68c7f5ebeb31dbb34595e0b767508220306c29511dbc14ec9c42be1779"


def verify_smoke(smoke, fit_support):
    receipt_path = smoke / "result.json"
    if sha256_file(receipt_path) != SMOKE_RESULT_SHA:
        raise ValueError("sealed scDFM C2 smoke changed")
    receipt = read_json(receipt_path)
    if (
        not receipt["formal_eligible"]
        or receipt["validation_prediction_shape"] != [300, 3352]
        or receipt["cohort_sha256"] != COHORT_SHA
        or receipt["shared_fit_support_file_sha256"] != fit_support
        or receipt["official_commit"] != scdfm.OFFICIAL_SHA
        or receipt["torch_version"] != str(torch.__version__)
    ):
        raise ValueError("sealed scDFM smoke differs from the capped run inputs")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", receipt["source_commit"], "HEAD"],
        cwd=REPO,
        check=True,
    )


def fit_capped(output, inputs, support, config, started, identity):
    source, _, cohort, rows = inputs
    if (output / "fit").exists():
        raise FileExistsError("capped fit must use a new fit directory")
    prepared = open_crosscell_fit_partition(
        source=source,
        verified_rows=rows,
        cohort=cohort,
        mask_expression=True,
        fit_compatibility=support,
    )
    atomic_json(output / "double_holdout_fit.json", prepared.receipt)
    axes_file = output / "axes.json"
    atomic_json(
        axes_file,
        {"expression_gene_ids": prepared.gene_ids, "graph_gene_ids": prepared.graph_gene_ids},
    )
    budget = config["budget"]
    runtime = {"official_checkout": str(CHECKOUT), "device": "cuda:0", "seed": config["seed"]}
    fitted = scdfm.fit(
        prepared,
        config,
        runtime,
        output / "fit",
        stop_starting_updates_at=(
            started + budget["training_wall_seconds"] - budget["checkpoint_reserve_seconds"]
        ),
    )
    fit_receipt = output / "fit_receipt.json"
    atomic_json(fit_receipt, fitted.receipt)
    atomic_json(
        output / "training_result.json",
        {
            **identity,
            "status": "trained_budgeted",
            "checkpoint_role": "last",
            "checkpoint_sha256": sha256_file(output / "fit/last.pt"),
            "coexpression_mask_sha256": sha256_file(output / "fit/coexpression-mask.pt"),
            "gene_axes_sha256": sha256_file(axes_file),
            "fit_receipt_sha256": sha256_file(fit_receipt),
            "planned_updates": fitted.receipt["total_optimizer_updates"],
            "completed_updates": fitted.receipt["completed_updates"],
            "budget_stop_reason": fitted.receipt["budget_stop_reason"],
            "training_wall_seconds": time.monotonic() - started,
        },
    )


def score_bounded(output, inputs, support, config, started, identity):
    source, cross, cohort, rows = inputs
    training = read_json(output / "training_result.json")
    if training["status"] != "trained_budgeted" or any(
        training.get(k) != v for k, v in identity.items()
    ):
        raise ValueError("capped frozen evaluation differs from the fit identity")
    files = {
        "checkpoint_sha256": "fit/last.pt",
        "coexpression_mask_sha256": "fit/coexpression-mask.pt",
        "gene_axes_sha256": "axes.json",
        "fit_receipt_sha256": "fit_receipt.json",
    }
    for key, relative in files.items():
        if sha256_file(output / relative) != training[key]:
            raise ValueError(f"sealed scDFM state changed: {relative}")
    fit_receipt = read_json(output / "fit_receipt.json")
    axes = read_json(output / "axes.json")
    genes, graph = tuple(axes["expression_gene_ids"]), axes["graph_gene_ids"]
    if sha256_json(list(genes)) != cross["expression_gene_order_sha256"] or graph[
        : len(genes)
    ] != list(genes):
        raise ValueError("scDFM restored gene axes do not match the frozen panel")
    official = scdfm.load_official(CHECKOUT)
    device = torch.device("cuda:0")
    with scdfm.official_namespace(CHECKOUT, official.modules):
        vocab = official.tokenizer.GeneVocab(
            graph, specials=["<pad>", "<cls>", "<mask>", "control"]
        )
        shape = fit_receipt["actual_model"]
        if len(vocab) != shape["ntoken"] or shape["nlayers"] != 4:
            raise ValueError("scDFM restored vocabulary does not match the fit")
        model = official.origin.model(
            ntoken=len(vocab),
            nlayers=shape["nlayers"],
            fusion_method="differential_perceiver",
            perturbation_function="crisper",
            mask_path=str(output / "fit/coexpression-mask.pt"),
        ).to(device)
        fitted = FittedAdapter(
            scdfm.ScDFMState(
                model, official, CHECKOUT, vocab, genes, device, fit_receipt["resolved_training"]
            ),
            {},
        )
        restore_last(fitted, "scdfm", output / "fit/last.pt")
    model.eval().requires_grad_(False)

    def predict(condition, controls):
        targets = tuple(part for part in condition.split("+") if part != "ctrl")
        with torch.no_grad():
            return scdfm.predict(
                fitted,
                controls,
                targets,
                genes,
                cell_batch=config["evaluation"]["cell_batch"],
                query_count=config["evaluation"]["query_count"],
                seed=config["seed"],
            )

    key = sha256_json(
        {
            "checkpoint": training["checkpoint_sha256"],
            "config": training["config_sha256"],
            "pairs": cohort["per_condition_pairing_sha256"],
            "source": training["source_commit"],
        }
    )
    budget = config["budget"]
    with official_module_session(
        checkout_root=UPSTREAM,
        expected_commit=UPSTREAM_COMMIT,
        module_names=("gspp.metrics",),
    ) as (modules, _):
        result = score(
            predict,
            modules["gspp.metrics"],
            source,
            cross,
            cohort,
            rows,
            support,
            output,
            journal_key=key,
            stop_starting_conditions_at=(
                started
                + budget["evaluation_task_wall_seconds"]
                - budget["evaluation_condition_reserve_seconds"]
            ),
        )
    if "conditions" in result:
        atomic_json(output / "result.json", {**training, **result, "status": "evaluated"})


def coordinate(output, budget):
    """Run the fit, then release the GPU between individual evaluation tasks."""
    output.mkdir(exist_ok=False)
    argv = [
        sys.executable,
        "-u",
        "-m",
        "benchmarks.expression_holdout.scdfm_crosscell_budgeted",
        *sys.argv[1:],
    ]
    subprocess.run([*argv, "--stage", "fit"], check=True, timeout=budget["training_wall_seconds"])
    if not (output / "training_result.json").is_file():
        raise RuntimeError("scDFM capped training has no accepted checkpoint receipt")
    journal = output / "condition-receipts"
    while not (output / "result.json").is_file():
        previous_count = len(list(journal.glob("*.json")))
        try:
            subprocess.run(
                [*argv, "--stage", "evaluate"],
                check=True,
                timeout=budget["evaluation_task_wall_seconds"],
            )
        except subprocess.TimeoutExpired:
            # The subprocess is reaped here; atomic per-condition receipts survive.
            if len(list(journal.glob("*.json"))) <= previous_count:
                raise
        if (
            not (output / "result.json").is_file()
            and len(list(journal.glob("*.json"))) <= previous_count
        ):
            raise RuntimeError("scDFM bounded evaluation made no progress")


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-support-sha256", required=True)
    parser.add_argument("--smoke-run", type=Path, required=True)
    parser.add_argument("--stage", choices=("fit", "evaluate"))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(STUDY) or any(
        os.getenv(key) != value
        for key, value in {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }.items()
    ):
        raise ValueError("budgeted scDFM C2 requires the sealed server CUDA envelope")
    source = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    config = yaml.safe_load(CONFIG.read_text())
    if any(
        config["training"][key] != value
        for key, value in scdfm.OFFICIAL_RECIPE.items()
        if key != "steps"
    ):
        raise ValueError("capped scDFM changed a training setting other than the step budget")
    if args.stage is None:
        verify_smoke(args.smoke_run, args.fit_support_sha256)
        coordinate(args.output, config["budget"])
        return
    support_path = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
    if sha256_file(support_path) != args.fit_support_sha256:
        raise ValueError("shared C2 fit support differs from the queue plan")
    support = read_json(support_path)
    if support["source_cohort_sha256"] != COHORT_SHA:
        raise ValueError("shared C2 fit support is for another cohort")
    identity = {
        "run_id": args.output.name,
        "scenario": "C2",
        "formal_eligible": source.formal_eligible,
        "source_commit": source.commit,
        "config_sha256": sha256_file(CONFIG),
        "official_commit": scdfm.OFFICIAL_SHA,
        "shared_fit_support_file_sha256": args.fit_support_sha256,
        "cohort_sha256": COHORT_SHA,
        "initialization": "from_scratch",
        "training_budget": "project_capped_3h",
    }
    with lock(SERVER / "runtime" / "v2-gpu-0.lock"):
        admit_selected_idle(snapshot(os.getpid()), 0, GPU_UUID)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        atomic_json(
            args.output / "worker.json",
            {"stage": args.stage, "owner": process_info(os.getpid()), "physical_gpu": 0},
        )
        inputs = cohort_inputs()
        if args.stage == "fit":
            fit_capped(args.output, inputs, support, config, started, identity)
        else:
            score_bounded(args.output, inputs, support, config, started, identity)


if __name__ == "__main__":
    main()
