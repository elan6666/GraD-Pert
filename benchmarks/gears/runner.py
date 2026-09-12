"""End-to-end one-epoch runner for the frozen official GEARS package."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse  # type: ignore[import-untyped]

from benchmarks.common import (
    build_training_validation_adata,
    official_module_session,
    write_adapter_receipt,
    write_pickle,
)
from benchmarks.common.full_gate import require_completed_smoke
from benchmarks.common.r50_gate import require_r50_smoke, seal_r50_smoke
from benchmarks.gears.official_api import (
    GearsModelParameters,
    GearsOfficialModules,
    OfficialGearsAPI,
)
from gradpert.artifacts import PredictionConditionArrays
from gradpert.config import ExperimentConfig, load_experiment_config
from gradpert.data._io import atomic_json, atomic_text
from gradpert.data.registry import load_dataset_registry
from gradpert.evaluation import CanonicalEvaluationData
from gradpert.execution.artifact_run import seal_evaluated_run
from gradpert.execution.identity import inspect_environment, inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.data import CanonicalTrainingData, write_training_data_receipt

PROJECT_REPOSITORY = "https://github.com/elan6666/GraD-Pert.git"


def _ensure_official_sparse_expression(adata: Any) -> dict[str, object]:
    """Preserve values while satisfying the frozen GEARS sparse-X contract."""

    observed_sparse = bool(sparse.issparse(adata.X))
    if observed_sparse:
        adata.X = adata.X.tocsr()
    else:
        adata.X = sparse.csr_matrix(np.asarray(adata.X))
    return {
        "input_expression_storage": "sparse" if observed_sparse else "dense",
        "official_expression_storage": "scipy_csr_matrix",
    }


def _parameter(config: ExperimentConfig, name: str, expected: type) -> Any:
    value = config.model.parameters[name].value
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise ValueError(f"GEARS parameter {name} has the wrong type")
    return value


def _training(config: ExperimentConfig, name: str, expected: type) -> Any:
    value = getattr(config.training, name).value
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise ValueError(f"GEARS training parameter {name} has the wrong type")
    return value


def preflight(config_path: Path, checkout_root: Path) -> dict[str, object]:
    config = load_experiment_config(config_path)
    if config.model_id != "gears" or config.training.formal_run_policy not in {
        "smoke_only",
        "external_full_100",
        "external_fixed_50",
    }:
        raise ValueError("GEARS runner requires a gears smoke-only experiment config")
    with official_module_session(
        checkout_root=checkout_root,
        expected_commit=config.source_code.commit,
        module_names=("gears", "gears.utils", "gears.data_utils"),
    ) as (modules, receipt):
        modules["torch"] = importlib.import_module("torch")
        modules["torch_geometric.loader"] = importlib.import_module("torch_geometric.loader")
        OfficialGearsAPI(GearsOfficialModules.from_mapping(modules))
    return {
        "schema_version": "official-runner-preflight-v1",
        "model_id": config.model_id,
        "dataset_id": config.dataset_id,
        "config_path": str(config_path.resolve()),
        "checkout": receipt.payload(),
        "smoke_epochs": config.training.smoke_epochs.value,
        "formal_run_policy": config.training.formal_run_policy,
        "official_symbols": [
            "gears.PertData",
            "gears.GEARS.model_initialize",
            "gears.GEARS.train",
            "gears.GEARS.save_model",
            "gears.utils.create_cell_graph_for_prediction",
        ],
    }


def run_one_epoch(
    *,
    config_path: Path,
    checkout_root: Path,
    data_root: Path,
    official_data_root: Path,
    run_root: Path,
    run_id: str,
    device: str,
    repository_root: Path,
    formal: bool,
    development_commit: str | None,
    smoke: bool = False,
    smoke_run_root: Path | None = None,
    source_publication_receipt: Path | None = None,
    source_publication_receipt_sha256: str | None = None,
) -> dict[str, object]:
    config_file = config_path.resolve(strict=True)
    config = load_experiment_config(config_file)
    requested_epochs = 1 if smoke else int(config.training.max_epochs.value)
    r50 = config.training.formal_run_policy == "external_fixed_50"
    if (
        r50
        and device.startswith("cuda")
        and os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
    ):
        raise ValueError("R50 CUDA allocator contract missing")
    if config.model_id != "gears" or config.training.max_epochs.value not in {1, 50, 100}:
        raise ValueError("GEARS execution requires a one-epoch GEARS config")
    destination = run_root.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("run root must be new and empty")
    destination.mkdir(parents=True, exist_ok=True)
    dataset_entry = load_dataset_registry(
        repository_root / "registry" / "datasets" / f"{config.dataset_id}.yaml"
    )
    condition_policy = dataset_entry.benchmark_condition_policy
    if config.source_code.commit != condition_policy.official_commit:
        raise ValueError("GEARS config and condition-policy commits differ")
    source = inspect_source_identity(
        repository_root,
        formal=formal,
        expected_repository=PROJECT_REPOSITORY,
        development_commit=development_commit,
        publication_receipt=source_publication_receipt,
        expected_publication_receipt_sha256=source_publication_receipt_sha256,
    )
    environment = inspect_environment(
        repository_root,
        device_name=device,
        lock_file=repository_root / "benchmarks" / "environments" / "gears.uv.lock",
    )
    if r50 and (source.dirty or not source.formal_eligible):
        raise ValueError("R50 requires clean published source before fitting")
    config_sha256 = sha256_file(config_file)
    small_root = destination / "small_results"
    atomic_text(small_root / "config.resolved.yaml", config_file.read_text(encoding="utf-8"))
    atomic_json(small_root / "source_identity.json", source.payload())
    atomic_json(small_root / "environment.json", environment.payload())
    random.seed(1)
    np.random.seed(1)
    official_data_root.mkdir(parents=True, exist_ok=True)

    with CanonicalTrainingData(
        dataset_id=config.dataset_id,
        protocol_id=config.data.protocol_id,
        data_root=data_root,
        run_seed=1,
    ) as training_data:
        training_data.require_experiment_data_contract(
            registry_version=config.data.registry_version,
            split_policy=config.data.split_policy,
        )
        write_training_data_receipt(training_data, small_root / "training_data.json")
        if requested_epochs == 100 or (r50 and not smoke):
            atomic_json(
                small_root / "smoke_gate.json",
                (require_r50_smoke if r50 else require_completed_smoke)(
                    smoke_run_root,
                    config=config,
                    config_sha256=config_sha256,
                    training_data=training_data,
                    source_commit=source.commit,
                    **({"environment_sha256": environment.payload_sha256} if r50 else {}),
                ),
            )
        adapted = build_training_validation_adata(training_data, axis="graph")
        storage_receipt = _ensure_official_sparse_expression(adapted.adata)
        split_path = destination / "official_adapter" / "custom_split.pkl"
        split_sha256 = write_pickle(
            split_path,
            {
                "train": list(adapted.train_conditions),
                "val": list(adapted.val_conditions),
                "test": [],
            },
        )
        adapter_receipt = {
            **adapted.receipt,
            "official_model": "gears",
            "official_test_loader_policy": "empty_then_removed_before_fit",
            **storage_receipt,
            "split_pickle_sha256": split_sha256,
            "de_ranking_scope": "official_formula_train_validation_with_singleton_fallback",
            "nonzero_metadata": "official_formula_with_DE_ranking",
        }
        write_adapter_receipt(small_root / "official_data_adapter.json", adapter_receipt)
        with official_module_session(
            checkout_root=checkout_root,
            expected_commit=config.source_code.commit,
            module_names=("gears", "gears.utils", "gears.data_utils"),
        ) as (modules, checkout_receipt):
            modules["torch"] = importlib.import_module("torch")
            modules["torch_geometric.loader"] = importlib.import_module("torch_geometric.loader")
            api = OfficialGearsAPI(GearsOfficialModules.from_mapping(modules))
            pert_data = api.prepare_training_data(
                data_root=official_data_root,
                dataset_id=f"gradpert_{config.dataset_id}",
                training_validation_adata=adapted.adata,
                split_pickle_path=split_path,
                train_batch_size=_training(config, "train_batch_size", int),
                eval_batch_size=_training(config, "eval_batch_size", int),
            )
            atomic_json(
                small_root / "official_de_ranking.json",
                pert_data.gradpert_de_ranking_receipt,
            )
            observed_graph_resources = {
                "gene2go_all.pkl": sha256_file(official_data_root / "gene2go_all.pkl"),
                "essential_all_data_pert_genes.pkl": sha256_file(
                    official_data_root / "essential_all_data_pert_genes.pkl"
                ),
            }
            expected_graph_resources = {
                "gene2go_all.pkl": condition_policy.gene2go_resource_sha256,
                "essential_all_data_pert_genes.pkl": (
                    condition_policy.essential_gene_resource_sha256
                ),
            }
            if observed_graph_resources != expected_graph_resources:
                raise ValueError("official GEARS graph resources differ from condition policy")
            covered_targets = api.require_perturbation_coverage(
                pert_data,
                [
                    *training_data.split.train_conditions,
                    *training_data.split.val_conditions,
                    *training_data.split.test_conditions,
                ],
            )
            torch = modules["torch"]
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            model = api.fit_one_epoch(
                pert_data=pert_data,
                parameters=GearsModelParameters(
                    hidden_size=_parameter(config, "hidden_size", int),
                    num_go_gnn_layers=_parameter(config, "num_go_gnn_layers", int),
                    num_gene_gnn_layers=_parameter(config, "num_gene_gnn_layers", int),
                    decoder_hidden_size=_parameter(config, "decoder_hidden_size", int),
                    num_similar_genes_go_graph=_parameter(
                        config, "num_similar_genes_go_graph", int
                    ),
                    num_similar_genes_co_express_graph=_parameter(
                        config, "num_similar_genes_co_express_graph", int
                    ),
                    coexpress_threshold=_parameter(config, "coexpress_threshold", float),
                    uncertainty=_parameter(config, "uncertainty", bool),
                    uncertainty_reg=float(_parameter(config, "uncertainty_reg", int)),
                    direction_lambda=_parameter(config, "direction_lambda", float),
                    no_perturb=_parameter(config, "no_perturb", bool),
                ),
                learning_rate=_training(config, "learning_rate", float),
                weight_decay=_training(config, "weight_decay", float),
                checkpoint_dir=destination / "checkpoints" / "best",
                device=device,
                experiment_name=run_id,
                **(
                    {
                        "epochs": requested_epochs,
                        "patience": 10,
                        "progress_path": small_root / "validation_progress.json",
                    }
                    if r50
                    or (config.training.formal_run_policy == "external_full_100" and not smoke)
                    else {}
                ),
                **(
                    {"r50": True, "last_checkpoint_path": destination / "checkpoints/last.pt"}
                    if r50
                    else {}
                ),
            )
            checkpoint_path = destination / "checkpoints" / "best" / "model.pt"
            checkpoint_sha256 = sha256_file(checkpoint_path)
            framework_pickles = (
                destination / "checkpoints" / "best" / "config.pkl",
                split_path,
            )
            missing_framework_pickles = [
                str(path) for path in framework_pickles if not path.is_file()
            ]
            if missing_framework_pickles:
                raise FileNotFoundError(
                    "frozen GEARS did not materialize expected temporary metadata: "
                    + ", ".join(missing_framework_pickles)
                )
            removed_framework_pickle_sha256 = {
                str(path.relative_to(destination)): sha256_file(path) for path in framework_pickles
            }
            for path in framework_pickles:
                path.unlink()
            atomic_json(
                small_root / "training_receipt.json",
                {
                    "schema_version": "official-training-receipt-v1",
                    "model_id": config.model_id,
                    "dataset_id": config.dataset_id,
                    "epochs_requested": requested_epochs,
                    "phase": "smoke" if requested_epochs == 1 else "full",
                    "epochs_completed": len(getattr(model, "gradpert_validation_history", [None])),
                    "validation_history": getattr(model, "gradpert_validation_history", []),
                    "official_training_api": "gears.GEARS.train",
                    "train_batch_size": _training(config, "train_batch_size", int),
                    "eval_batch_size": _training(config, "eval_batch_size", int),
                    "optimizer": _training(config, "optimizer", str),
                    "learning_rate": _training(config, "learning_rate", float),
                    "weight_decay": _training(config, "weight_decay", float),
                    "scheduler": _training(config, "scheduler", str),
                    "canonical_test_loader_present_during_fit": False,
                    "canonical_test_truth_present_during_fit": False,
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_format": "official_state_dict_pt",
                    "checkpoint_configuration_reconstruction": (
                        "self_contained_experiment_config_plus_frozen_official_graph_preparation"
                    ),
                },
            )
            atomic_json(
                small_root / "checkpoint_retention.json",
                {
                    "schema_version": "checkpoint-retention-v1",
                    "policy": "best_and_last" if r50 else "model_pt_only_after_official_fit",
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "removed_framework_pickle_sha256": removed_framework_pickle_sha256,
                    "persistent_pkl_count": 0,
                },
            )
            if r50 and smoke:
                atomic_json(small_root / "official_checkout.json", checkout_receipt.payload())
                seal_r50_smoke(
                    destination,
                    config=config,
                    config_sha256=config_sha256,
                    training_data=training_data,
                    source=source,
                    environment_sha256=environment.payload_sha256,
                    best_checkpoint=checkpoint_path,
                    last_checkpoint=destination / "checkpoints/last.pt",
                    validation_value=model.gradpert_validation_history[0]["val_mse_de"],
                )
                return {
                    "run_id": run_id,
                    "run_root": str(destination),
                    "status": "trained_validation_only",
                    "scientific_completion": False,
                    "epochs_completed": 1,
                    "formal_eligible": source.formal_eligible,
                }
            evaluations = {}
            for role in ("best", "last") if r50 else ("best",):
                role_destination = destination / "evaluations" / role if r50 else destination
                if r50:
                    role_destination.mkdir(parents=True, exist_ok=False)
                    role_checkpoint = (
                        checkpoint_path if role == "best" else destination / "checkpoints/last.pt"
                    )
                    # The official inference adapter forwards best_model, not model.
                    model.best_model.load_state_dict(
                        torch.load(role_checkpoint, map_location=device, weights_only=True)
                    )
                    checkpoint_sha256 = sha256_file(role_checkpoint)
                    random.seed(1)
                    np.random.seed(1)
                    torch.manual_seed(1)
                    torch.cuda.manual_seed_all(1)
                # Construct the canonical test reader only after fit and checkpoint
                # sealing. It does not exist anywhere in the training lifecycle.
                with CanonicalEvaluationData(
                    dataset_id=config.dataset_id,
                    protocol_id=config.data.protocol_id,
                    split_name="test",
                    data_root=data_root,
                ) as test_data:
                    predictions: list[PredictionConditionArrays] = []
                    for draw in test_data.control_manifest.draws:
                        controls = test_data.load_control_rows(tuple(draw.ordered_row_ids))
                        targets = tuple(
                            part for part in draw.condition_id.split("+") if part != "ctrl"
                        )
                        graph_prediction = api.predict_exact_controls(
                            trained_model=model,
                            perturbation_genes=targets,
                            input_controls=np.pad(
                                controls.expression,
                                (
                                    (0, 0),
                                    (
                                        0,
                                        training_data.manifest.n_graph_genes
                                        - training_data.manifest.n_expression_genes,
                                    ),
                                ),
                            ),
                            batch_size=_training(config, "eval_batch_size", int),
                        )
                        predictions.append(
                            PredictionConditionArrays(
                                condition_id=draw.condition_id,
                                prediction=graph_prediction[:, : adapted.expression_gene_count],
                                input_control=controls.expression,
                                input_control_row_ids=controls.ordered_row_ids,
                            )
                        )
                    sealed = seal_evaluated_run(
                        destination=role_destination,
                        config=config,
                        config_sha256=config_sha256,
                        run_id=f"{run_id}-{role}" if r50 else run_id,
                        run_seed=1,
                        source=source,
                        environment=environment,
                        training_data=training_data,
                        test_data=test_data,
                        predictions=predictions,
                        checkpoint_sha256=checkpoint_sha256,
                    )
                if r50:
                    best_epoch = [
                        row["epoch"] for row in model.gradpert_validation_history if row["best"]
                    ][-1]
                    record = {
                        "checkpoint_role": role,
                        "checkpoint_epoch": best_epoch
                        if role == "best"
                        else model.gradpert_last_epoch,
                        "checkpoint_sha256": checkpoint_sha256,
                        "training_git_sha": source.commit,
                        "evaluation_git_sha": source.commit,
                        "source_dirty": source.dirty,
                        "upstream_commit": config.source_code.commit,
                        "config_sha256": config_sha256,
                        "environment_sha256": environment.payload_sha256,
                        "run_manifest_sha256": sha256_file(
                            role_destination / "small_results/run_manifest.json"
                        ),
                        "prediction_manifest_sha256": sha256_file(sealed.prediction_manifest_path),
                        "evaluation_manifest_sha256": sha256_file(sealed.evaluation_manifest_path),
                    }
                    atomic_json(role_destination / "small_results/checkpoint_role.json", record)
                    evaluations[role] = record
            if r50:
                if any(
                    p.suffix.lower() in {".pkl", ".pickle"} or p.name.startswith(".result-work-")
                    for p in destination.rglob("*")
                ):
                    raise RuntimeError("external R50 whole-root zero-PKL/work postcondition failed")
                atomic_json(small_root / "best_last_tests.json", evaluations)
        atomic_json(
            small_root / "official_checkout.json",
            {
                **checkout_receipt.payload(),
                "condition_policy_id": condition_policy.policy_id,
                "condition_policy_sha256": sha256_json(condition_policy.model_dump(mode="json")),
                "excluded_condition_count": len(condition_policy.excluded_conditions),
                "excluded_conditions_sha256": sha256_json(condition_policy.excluded_conditions),
                "official_graph_resources": observed_graph_resources,
                "covered_target_count": len(covered_targets),
                "covered_targets_sha256": sha256_json(list(covered_targets)),
            },
        )
    return {
        "run_id": run_id,
        "run_root": str(destination),
        "status": sealed.run_manifest.status,
        "formal_eligible": sealed.run_manifest.formal_eligible,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-run-root", type=Path)
    parser.add_argument("--source-publication-receipt", type=Path)
    parser.add_argument("--source-publication-receipt-sha256")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--official-data-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--development-commit")
    args = parser.parse_args(argv)
    if args.preflight_only:
        payload = preflight(args.config, args.official_checkout)
    else:
        required = {
            "data_root": args.data_root,
            "official_data_root": args.official_data_root,
            "run_root": args.run_root,
            "run_id": args.run_id,
            "repository_root": args.repository_root,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"execution requires: {', '.join(missing)}")
        if args.formal == (args.development_commit is not None):
            parser.error("choose exactly one of --formal or --development-commit")
        payload = run_one_epoch(
            config_path=args.config,
            checkout_root=args.official_checkout,
            data_root=args.data_root,
            official_data_root=args.official_data_root,
            run_root=args.run_root,
            run_id=args.run_id,
            device=args.device,
            repository_root=args.repository_root,
            formal=args.formal,
            development_commit=args.development_commit,
            smoke=args.smoke,
            smoke_run_root=args.smoke_run_root,
            source_publication_receipt=args.source_publication_receipt,
            source_publication_receipt_sha256=args.source_publication_receipt_sha256,
        )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
