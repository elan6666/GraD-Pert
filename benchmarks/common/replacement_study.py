"""Train and freeze the GEARS and Scouter/GenePT-Seed dual-holdout baselines."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import yaml

from benchmarks.common import official_module_session
from benchmarks.common.study_fit_partition import open_crosscell_fit_partition
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
from benchmarks.expression_holdout import scouter
from benchmarks.jurkat_j2 import study as j2
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot

SUPPORT = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
CONFIGS = REPO / "configs/v2/official_expression_20261013"
PRIOR = Path(
    "/data/yilangliu/DinoGenePT/data/embeddings/seed-go-protein-pathway-master-aligned.npz"
)
PRIOR_SHA = "34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318"


def targets(condition):
    return [gene for gene in condition.split("+") if gene != "ctrl"]


def run(args, output, identity, shared=None):
    started = time.monotonic()
    if args.scenario == "J2":
        prepared = j2.fit_data(seed=1)
        seal = j2.cohort()
        condition_ids = read_json(Path(seal["shared_split_path"]))["test_conditions"]
        if sha256_json(condition_ids) != seal["condition_ids_sha256"]["test"]:
            raise ValueError("frozen J2 conditions changed")
    else:
        if sha256_file(SUPPORT) != SUPPORT_SHA:
            raise ValueError("C2 shared support hash changed")
        support = read_json(SUPPORT)
        source, cross, cohort, rows = cohort_inputs()
        condition_ids = cohort["condition_ids"]
        prepared = open_crosscell_fit_partition(
            source=source,
            verified_rows=rows,
            cohort=cohort,
            mask_expression=True,
            fit_compatibility=support,
        )
    if set(condition_ids) & set((*prepared.train_conditions, *prepared.val_conditions)):
        raise ValueError("fit roles contain a frozen test condition")
    required = sorted({gene for name in condition_ids for gene in targets(name)})
    genes = prepared.gene_ids
    if len(genes) != (5000 if args.scenario == "J2" else 3352):
        raise ValueError("canonical full axis changed")
    if set(genes[i] for i in prepared.excluded_indices) != set(genes) & set(required):
        raise ValueError("fit expression holdout differs from the frozen test targets")
    atomic_json(output / "fit_partition.json", prepared.receipt)

    capture_start = time.monotonic()
    next_epoch_deadline = started + args.fit_minutes * 60
    config = yaml.safe_load(
        (
            CONFIGS
            / ("scouter_project_jurkat.yaml" if args.model == "scouter" else "gears_jurkat.yaml")
        ).read_text()
    )
    epochs = args.epochs or (40 if args.model == "scouter" else 20)

    def on_epoch(epoch, value):
        if shared is not None:
            shared.check()
        elapsed = time.monotonic() - capture_start
        atomic_json(
            output / "training_progress.json",
            {
                "epoch": epoch,
                "epoch_limit": epochs,
                "validation_loss": float(value),
                "fit_seconds": elapsed,
            },
        )
        # Finish completed epochs. Avoid starting another estimated to exceed the fit window.
        if epoch < epochs and time.monotonic() + elapsed / epoch >= next_epoch_deadline:
            raise scouter.FitBudgetReached

    if args.model == "scouter":
        config["training"]["n_epochs"] = epochs
        config["training"]["patience"] = 5
        runtime = {
            "official_checkout": SERVER / "upstreams/scouter-0cfddd0",
            "prior_kind": "project_genept_seed",
            "prior_path": PRIOR,
            "prior_sha256": PRIOR_SHA,
            "prior_dimension": 2048,
            "prior_embedding_key": "vectors",
            "required_target_genes": required,
            "device": "cuda:0",
            "seed": 1,
        }
        model_module = scouter
    else:
        from benchmarks.gears import replacement as model_module

        config["training"]["epochs"] = epochs
        runtime = {
            "official_checkout": SERVER / "upstreams/gears",
            "required_target_genes": required,
            "device": "cuda:0",
            "seed": 1,
        }
    atomic_json(
        output / "effective_config.json", {"model": config["model"], "training": config["training"]}
    )
    module_name = "scouter" if args.model == "scouter" else "gears"
    with official_module_session(
        checkout_root=runtime["official_checkout"],
        expected_commit=model_module.OFFICIAL_SHA,
        module_names=(module_name,),
    ) as (_, upstream):
        options = {"fit_role_masks": True} if args.model == "scouter" else {}
        fitted = model_module.fit(prepared, config, runtime, output, on_epoch=on_epoch, **options)
        del prepared
        receipt = {
            **identity,
            **fitted.receipt,
            "upstream": upstream.payload(),
            "checkpoint_sha256": sha256_file(output / "last.pt"),
            "fit_partition_sha256": sha256_file(output / "fit_partition.json"),
            "effective_config_sha256": sha256_file(output / "effective_config.json"),
            "status": "trained_validation_only",
            "test_updates": 0,
            "test_conditions_accessed_during_fit": 0,
            "fit_seconds": time.monotonic() - capture_start,
        }
        atomic_json(output / "training_result.json", receipt)
        if args.validation_only:
            atomic_json(output / "result.json", {**receipt, "formal_eligible": False})
            return

        last_gpu_check = time.monotonic()

        def predict(name, controls):
            nonlocal last_gpu_check
            if shared is not None and time.monotonic() - last_gpu_check >= 10:
                shared.check()
                last_gpu_check = time.monotonic()
            return model_module.predict(
                fitted, controls, targets(name), genes, cell_batch=128, query_count=300, seed=1
            )

        deadline = started + args.wall_minutes * 60
        if args.scenario == "J2":

            def predict_j2(name, controls, ids, axis):
                if len(ids) != 300 or len(axis) != len(genes):
                    raise ValueError(
                        "the common J2 scorer did not pass the canonical controls and axis"
                    )
                return predict(name, controls)

            metrics = j2.evaluate_segment(
                predict_j2, output, receipt, deadline=deadline, full_axis=True
            )
            if metrics is None:
                return
        else:
            with official_module_session(
                checkout_root=TXPERT_CHECKOUT,
                expected_commit=TXPERT_COMMIT,
                module_names=("gspp.metrics",),
            ) as (modules, _):
                metrics = score(
                    predict,
                    modules["gspp.metrics"],
                    source,
                    cross,
                    cohort,
                    rows,
                    support,
                    output,
                    journal_key=receipt["checkpoint_sha256"],
                    stop_starting_conditions_at=deadline,
                    full_axis=True,
                )
            if "conditions_completed" in metrics:
                atomic_json(
                    output / "segment.json",
                    {
                        **metrics,
                        "status": "evaluation_segment_complete",
                        "checkpoint_sha256": receipt["checkpoint_sha256"],
                        "test_updates": 0,
                    },
                )
                return
        if metrics["conditions"] != len(condition_ids):
            raise ValueError("frozen evaluation omitted test conditions")
        if (
            metrics["native_prediction_genes"] != len(genes)
            or metrics["imputed_prediction_genes"] != 0
            or metrics["systema_reference_scope"] != "train_only"
        ):
            raise ValueError("replacement baseline changed the all-native frozen scoring contract")
        gpu_evidence = shared.evidence() if shared is not None else {}
        atomic_json(
            output / "result.json",
            {**receipt, **metrics, "gpu_evidence": gpu_evidence, "status": "evaluated"},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gears", "scouter"), required=True)
    parser.add_argument("--scenario", choices=("J2", "C2"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publication", required=True, type=Path)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--gpu", choices=(0, 1), type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--fit-minutes", type=int, default=135)
    parser.add_argument("--wall-minutes", type=int, default=180)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--smoke-run", type=Path)
    parser.add_argument("--share-gpu", action="store_true")
    args = parser.parse_args()
    if not 0 < args.fit_minutes < args.wall_minutes or (
        args.epochs is not None and args.epochs < 1
    ):
        parser.error("expected a positive fit budget below total and positive epoch limit")
    if args.model == "gears":
        if args.validation_only and args.epochs != 1:
            parser.error("GEARS smoke requires exactly one validation-only epoch")
        if not args.validation_only and args.smoke_run is None:
            parser.error("GEARS formal execution requires the completed dataset smoke")
    output = args.output.resolve()
    if output.parent != STUDY or os.getenv("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("expected the study output root and one selected physical CUDA device")
    source = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        verification_method="git_ls_remote_live",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    identity = {
        "run_id": output.name,
        "scenario": args.scenario,
        "model": args.model,
        "source_commit": source.commit,
        "formal_eligible": source.formal_eligible,
        "cohort_sha256": COHORT_SHA,
        "gpu": args.gpu,
        "selected_eval_checkpoint": "last",
    }
    if args.smoke_run is not None:
        smoke = args.smoke_run.resolve()
        smoke_receipt = read_json(smoke / "result.json")
        if (
            smoke.parent != STUDY
            or smoke_receipt["model"] != args.model
            or smoke_receipt["scenario"] != args.scenario
            or smoke_receipt["source_commit"] != source.commit
            or smoke_receipt["cohort_sha256"] != COHORT_SHA
            or smoke_receipt["status"] != "trained_validation_only"
            or smoke_receipt["epochs_completed"] != 1
            or smoke_receipt["test_conditions_accessed_during_fit"] != 0
            or smoke_receipt["checkpoint_sha256"] != sha256_file(smoke / "last.pt")
            or smoke_receipt["output_validation_shape"]
            != [300, 5000 if args.scenario == "J2" else 3352]
        ):
            raise ValueError("GEARS smoke receipt is incompatible with this formal fit")
        identity["smoke_receipt_sha256"] = sha256_file(smoke / "result.json")
    lock_mode = "replacement-shared" if args.share_gpu else "v2"
    with lock(SERVER / f"runtime/{lock_mode}-gpu-{args.gpu}.lock"):
        shared = None
        if args.share_gpu:
            from benchmarks.crosscell.prescribe_gpu import SharedGpu

            # Preserve resident processes and enforce a separate allocator cap.
            shared = SharedGpu(
                args.gpu,
                args.gpu_uuid,
                {
                    "allocator_mib": 12288,
                    "process_mib": 14336,
                    "reserve_mib": 4096,
                    "foreign_limit_mib": 13312,
                    "pause_ratio": 0.0,
                },
                allow_same_account_resident=True,
            )
            shared.admit()
            shared.configure_allocator()
        else:
            admit_selected_idle(snapshot(os.getpid()), args.gpu, args.gpu_uuid)
        output.mkdir(exist_ok=False)
        atomic_json(output / "worker.json", {**identity, "owner": process_info(os.getpid())})
        torch.manual_seed(1)
        run(args, output, identity, shared=shared)


if __name__ == "__main__":
    main()
