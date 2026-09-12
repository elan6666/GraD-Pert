"""Official Scouter + GenePT-Seed on canonical data; separate process only."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from pathlib import Path

import numpy as np

from benchmarks.common import build_training_validation_adata, official_module_session
from benchmarks.common.full_gate import require_completed_smoke
from benchmarks.common.r50_gate import require_r50_smoke, seal_external_step, seal_r50_smoke
from benchmarks.scouter.official_api import fit_official, predict_exact_controls, prepare_data
from gradpert.artifacts import PredictionConditionArrays
from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, atomic_text
from gradpert.evaluation import CanonicalEvaluationData
from gradpert.execution.artifact_run import seal_evaluated_run
from gradpert.execution.identity import inspect_environment, inspect_source_identity
from gradpert.features.text_prior import verify_text_prior_npz
from gradpert.hashing import sha256_file
from gradpert.training.data import CanonicalTrainingData, write_training_data_receipt


def run(args):
    config = load_experiment_config(args.config)
    if config.model_id != "scouter_genept_seed" or config.training.formal_run_policy not in {
        "external_full_100",
        "external_fixed_50",
    }:
        raise ValueError("Scouter requires its explicit external-full config")
    r50 = config.training.formal_run_policy == "external_fixed_50"
    step_smoke = getattr(args, "step_smoke", False)
    if step_smoke and (not r50 or args.smoke):
        raise ValueError("--step-smoke requires R50 and excludes --smoke")
    if (config.training.monitor, config.training.monitor_mode, config.training.min_delta) != (
        "val/scouter_loss",
        "min",
        0.001,
    ):
        raise ValueError("Scouter requires the frozen official validation-loss selection")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator contract missing")
    source = inspect_source_identity(
        args.repository_root,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=args.publication_receipt,
        expected_publication_receipt_sha256=args.publication_receipt_sha256,
    )
    environment = inspect_environment(
        args.repository_root, device_name=args.device, lock_file=args.environment_lock
    )
    destination = args.run_root.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    small_root = destination / "small_results"
    atomic_json(small_root / "source_identity.json", source.payload())
    atomic_json(small_root / "environment.json", environment.payload())
    atomic_text(small_root / "config.resolved.yaml", args.config.read_text())
    with CanonicalTrainingData(
        dataset_id=config.dataset_id,
        protocol_id=config.data.protocol_id,
        data_root=args.data_root,
        run_seed=1,
    ) as training:
        training.require_experiment_data_contract(
            registry_version=config.data.registry_version, split_policy=config.data.split_policy
        )
        write_training_data_receipt(training, small_root / "training_data.json")
        if not args.smoke and not step_smoke:
            atomic_json(
                small_root / "smoke_gate.json",
                (require_r50_smoke if r50 else require_completed_smoke)(
                    args.smoke_run_root,
                    config=config,
                    config_sha256=sha256_file(args.config),
                    training_data=training,
                    source_commit=source.commit,
                    **({"environment_sha256": environment.payload_sha256} if r50 else {}),
                ),
            )
        targets = tuple(
            sorted(
                {
                    gene
                    for condition in [
                        *training.split.train_conditions,
                        *training.split.val_conditions,
                        *training.split.test_conditions,
                    ]
                    for gene in condition.split("+")
                    if gene != "ctrl"
                }
            )
        )
        prior = verify_text_prior_npz(
            args.genept_seed,
            expected_sha256=config.model.parameters["genept_seed_sha256"].value,
            expected_gene_ids=targets,
            perturbation_target_gene_ids=targets,
        )
        atomic_json(
            small_root / "prior.json",
            {
                "label": "Scouter + GenePT-Seed Protein+Reactome+SIGNOR (modified prior)",
                "source_sha256": prior.source_sha256,
                "width": prior.embedding_width,
                "target_count": len(targets),
                "target_order_sha256": prior.gene_order_sha256,
                "selected_matrix_sha256": prior.selected_matrix_sha256,
                "control_embedding": "zero",
                "frozen": True,
            },
        )
        adapted = build_training_validation_adata(training, axis="expression")
        atomic_json(
            small_root / "official_data_adapter.json",
            {
                **adapted.receipt,
                "test_truth_during_fit": False,
                "DE_rankings": "not_computed_unused_by_official_loss",
                "nonzero_masks": "official_helper_train_validation_only",
            },
        )
        with official_module_session(
            checkout_root=args.official_checkout,
            expected_commit=config.source_code.commit,
            module_names=("scouter",),
        ) as (modules, checkout):
            torch = importlib.import_module("torch")
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            data = prepare_data(modules["scouter"], adapted, prior)
            model = modules["scouter"].Scouter(data, device=args.device)
            receipt = fit_official(
                model,
                config,
                epochs=1 if args.smoke else int(config.training.max_epochs.value),
                progress_path=small_root / "stage_progress.json",
                r50=r50,
                last_checkpoint_path=destination / "checkpoints/last.pt"
                if r50 and not step_smoke
                else None,
                **(
                    {"step_checkpoint_path": destination / "checkpoints/step.pt"}
                    if step_smoke
                    else {}
                ),
            )
            if step_smoke:
                atomic_json(small_root / "official_checkout.json", checkout.payload())
                return seal_external_step(
                    destination,
                    config=config,
                    config_sha256=sha256_file(args.config),
                    training_data=training,
                    source=source,
                    environment_sha256=environment.payload_sha256,
                    checkpoint=destination / "checkpoints/step.pt",
                    update=receipt,
                )
            checkpoint = destination / "checkpoints" / "best.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            torch.save(model.network.state_dict(), checkpoint)
            checkpoint_sha256 = sha256_file(checkpoint)
            atomic_json(
                small_root / "training_receipt.json",
                {
                    **receipt,
                    "checkpoint_sha256": checkpoint_sha256,
                    "phase": "smoke" if args.smoke else "full",
                },
            )
            atomic_json(small_root / "official_checkout.json", checkout.payload())
            if r50 and args.smoke:
                seal_r50_smoke(
                    destination,
                    config=config,
                    config_sha256=sha256_file(args.config),
                    training_data=training,
                    source=source,
                    environment_sha256=environment.payload_sha256,
                    best_checkpoint=checkpoint,
                    last_checkpoint=destination / "checkpoints/last.pt",
                    validation_value=receipt["best_val_loss"],
                )
                return {
                    "run_id": args.run_id,
                    "status": "trained_validation_only",
                    "phase": "smoke",
                    "epochs_completed": 1,
                    "scientific_completion": False,
                }
            roles = ("best", "last") if r50 else ("best",)
            evaluations = {}
            for role in roles:
                role_destination = destination / "evaluations" / role if r50 else destination
                if r50:
                    role_destination.mkdir(parents=True, exist_ok=False)
                    role_checkpoint = destination / "checkpoints" / f"{role}.pt"
                    model.network.load_state_dict(
                        torch.load(role_checkpoint, map_location=args.device, weights_only=True)
                    )
                    checkpoint_sha256 = sha256_file(role_checkpoint)
                    random.seed(1)
                    np.random.seed(1)
                    torch.manual_seed(1)
                    torch.cuda.manual_seed_all(1)
                # Instantiate the canonical test reader only after fitting and best sealing.
                with CanonicalEvaluationData(
                    dataset_id=config.dataset_id,
                    protocol_id=config.data.protocol_id,
                    split_name="test",
                    data_root=args.data_root,
                ) as test:
                    predictions = []
                    for draw in test.control_manifest.draws:
                        controls = test.load_control_rows(tuple(draw.ordered_row_ids))
                        predictions.append(
                            PredictionConditionArrays(
                                condition_id=draw.condition_id,
                                prediction=predict_exact_controls(
                                    model,
                                    torch,
                                    draw.condition_id,
                                    controls.expression,
                                    int(config.training.eval_batch_size.value),
                                ),
                                input_control=controls.expression,
                                input_control_row_ids=controls.ordered_row_ids,
                            )
                        )
                    sealed = seal_evaluated_run(
                        destination=role_destination,
                        config=config,
                        config_sha256=sha256_file(args.config),
                        run_id=f"{args.run_id}-{role}" if r50 else args.run_id,
                        run_seed=1,
                        source=source,
                        environment=environment,
                        training_data=training,
                        test_data=test,
                        predictions=predictions,
                        checkpoint_sha256=checkpoint_sha256,
                    )
                if r50:
                    evaluation_record = {
                        "checkpoint_role": role,
                        "checkpoint_epoch": receipt["best_epoch"]
                        if role == "best"
                        else receipt["last_epoch"],
                        "checkpoint_sha256": checkpoint_sha256,
                        "training_git_sha": source.commit,
                        "evaluation_git_sha": source.commit,
                        "upstream_commit": config.source_code.commit,
                        "source_dirty": source.dirty,
                        "config_sha256": sha256_file(args.config),
                        "environment_sha256": environment.payload_sha256,
                        "run_manifest_sha256": sha256_file(
                            role_destination / "small_results/run_manifest.json"
                        ),
                        "prediction_manifest_sha256": sha256_file(sealed.prediction_manifest_path),
                        "evaluation_manifest_sha256": sha256_file(sealed.evaluation_manifest_path),
                    }
                    atomic_json(
                        role_destination / "small_results/checkpoint_role.json", evaluation_record
                    )
                    evaluations[role] = evaluation_record
            if r50:
                atomic_json(small_root / "best_last_tests.json", evaluations)
            atomic_json(small_root / "official_checkout.json", checkout.payload())
    if list(destination.rglob("*.pkl")):
        raise RuntimeError("persistent PKL postcondition failed")
    return {
        "run_id": args.run_id,
        "status": sealed.run_manifest.status,
        "phase": "smoke" if args.smoke else "full",
        "epochs_completed": receipt["epochs_completed"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "official-checkout",
        "data-root",
        "run-root",
        "repository-root",
        "genept-seed",
        "environment-lock",
        "publication-receipt",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication-receipt-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--step-smoke", action="store_true")
    parser.add_argument("--smoke-run-root", type=Path)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
