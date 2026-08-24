"""End-to-end one-epoch runner for the frozen public TxPert package."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from benchmarks.common import (
    build_training_validation_adata,
    official_module_session,
    write_adapter_receipt,
    write_pickle,
)
from benchmarks.txpert.official_api import OfficialPublicAPI, OfficialPublicModules
from gradpert.artifacts import PredictionConditionArrays
from gradpert.config import ExperimentConfig, load_experiment_config
from gradpert.data._io import atomic_json, atomic_text
from gradpert.evaluation import CanonicalEvaluationData
from gradpert.execution.artifact_run import seal_evaluated_run
from gradpert.execution.identity import inspect_environment, inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.data import CanonicalTrainingData, write_training_data_receipt

PROJECT_REPOSITORY = "https://github.com/elan6666/GraD-Pert.git"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _official_config(
    config: ExperimentConfig,
    checkout_root: Path,
) -> tuple[Path, dict[str, Any]]:
    config_file_value = config.model.parameters["official_config_file"].value
    config_sha_value = config.model.parameters["official_config_sha256"].value
    if not isinstance(config_file_value, str) or not isinstance(config_sha_value, str):
        raise ValueError("TxPert official config path/hash must be strings")
    official_config_path = (checkout_root.resolve() / config_file_value).resolve(strict=True)
    if not official_config_path.is_relative_to(checkout_root.resolve()):
        raise ValueError("TxPert official config resolves outside official checkout")
    observed_config_sha256 = _sha256_file(official_config_path)
    if observed_config_sha256 != config_sha_value:
        raise ValueError("TxPert official config SHA-256 mismatch")
    parsed = yaml.safe_load(official_config_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or not all(
        isinstance(parsed.get(key), dict) for key in ("model", "datamodule", "graph")
    ):
        raise ValueError("TxPert official YAML lacks model/datamodule/graph mappings")
    datamodule = parsed["datamodule"]
    if (
        datamodule.get("batch_size") != config.training.train_batch_size.value
        or datamodule.get("match_cntr") is not True
        or datamodule.get("avg_cntr") is not True
        or datamodule.get("obsm_key") != "raw"
    ):
        raise ValueError("TxPert experiment config differs from frozen official datamodule values")
    return official_config_path, parsed


def preflight(config_path: Path, checkout_root: Path) -> dict[str, object]:
    config = load_experiment_config(config_path)
    if config.model_id != "txpert_public" or config.training.formal_run_policy != "smoke_only":
        raise ValueError("TxPert runner requires a txpert_public smoke-only experiment config")
    official_config_path, _ = _official_config(config, checkout_root)
    with official_module_session(
        checkout_root=checkout_root,
        expected_commit=config.source_code.commit,
        module_names=(
            "gspp.predictor",
            "gspp.data.datamodule",
            "gspp.data.graphmodule",
        ),
    ) as (modules, receipt):
        modules["torch"] = importlib.import_module("torch")
        modules["lightning"] = importlib.import_module("lightning")
        OfficialPublicAPI(OfficialPublicModules.from_mapping(modules))
    return {
        "schema_version": "official-runner-preflight-v1",
        "model_id": config.model_id,
        "dataset_id": config.dataset_id,
        "config_path": str(config_path.resolve()),
        "checkout": receipt.payload(),
        "smoke_epochs": config.training.smoke_epochs.value,
        "formal_run_policy": config.training.formal_run_policy,
        "official_config_path": str(official_config_path),
        "official_config_sha256": _sha256_file(official_config_path),
        "official_symbols": [
            "gspp.predictor.PertPredictor",
            "gspp.data.datamodule.PertDataModule",
            "gspp.data.graphmodule.GSPGraph",
            "lightning.Trainer",
        ],
    }


def _write_official_cache(
    *,
    cache_root: Path,
    adapted: Any,
) -> dict[str, str]:
    cache_root.mkdir(parents=True, exist_ok=True)
    adata_path = cache_root / "de_adata_test.h5ad"
    adapted.adata.write_h5ad(adata_path)
    split_root = cache_root / "splits"
    split_sha256 = write_pickle(
        split_root / "train_test_split.pkl",
        {
            "train": list(adapted.train_conditions),
            "val": list(adapted.val_conditions),
            "test": list(adapted.val_conditions),
        },
    )
    subgroup_sha256 = write_pickle(
        split_root / "subgroup.pkl",
        {
            "adapter_policy": "canonical_val_duplicated_as_required_official_test",
            "canonical_test_truth_present": False,
        },
    )
    return {
        "adapted_h5ad_sha256": sha256_file(adata_path),
        "split_pickle_sha256": split_sha256,
        "subgroup_pickle_sha256": subgroup_sha256,
    }


def run_one_epoch(
    *,
    config_path: Path,
    checkout_root: Path,
    data_root: Path,
    run_root: Path,
    run_id: str,
    device: str,
    repository_root: Path,
    formal: bool,
    development_commit: str | None,
) -> dict[str, object]:
    config_file = config_path.resolve(strict=True)
    config = load_experiment_config(config_file)
    if config.model_id != "txpert_public" or config.training.max_epochs.value != 1:
        raise ValueError("TxPert execution requires a one-epoch TxPert config")
    official_config_path, official_config = _official_config(config, checkout_root)
    destination = run_root.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("run root must be new and empty")
    destination.mkdir(parents=True, exist_ok=True)
    source = inspect_source_identity(
        repository_root,
        formal=formal,
        expected_repository=PROJECT_REPOSITORY,
        development_commit=development_commit,
    )
    environment = inspect_environment(
        repository_root,
        device_name=device,
        lock_file=checkout_root / "uv.lock",
    )
    config_sha256 = sha256_file(config_file)
    small_root = destination / "small_results"
    atomic_text(small_root / "config.resolved.yaml", config_file.read_text(encoding="utf-8"))
    atomic_json(small_root / "source_identity.json", source.payload())
    atomic_json(small_root / "environment.json", environment.payload())
    random.seed(1)
    np.random.seed(1)

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
        adapted = build_training_validation_adata(training_data, axis="expression")
        cell_types = tuple(sorted(adapted.adata.obs["cell_type"].astype(str).unique()))
        if len(cell_types) != 1:
            raise ValueError("TxPert within-cell adapter requires exactly one cell type")
        cache_root = destination / "official_adapter" / "cache"
        cache_receipts = _write_official_cache(cache_root=cache_root, adapted=adapted)
        write_adapter_receipt(
            small_root / "official_data_adapter.json",
            {
                **adapted.receipt,
                **cache_receipts,
                "official_model": "txpert_public",
                "official_test_partition": "canonical_val_duplicate",
                "canonical_test_truth_present": False,
                "cell_type": cell_types[0],
            },
        )
        with official_module_session(
            checkout_root=checkout_root,
            expected_commit=config.source_code.commit,
            module_names=(
                "gspp.predictor",
                "gspp.data.datamodule",
                "gspp.data.graphmodule",
            ),
        ) as (modules, checkout_receipt):
            modules["torch"] = importlib.import_module("torch")
            modules["lightning"] = importlib.import_module("lightning")
            api = OfficialPublicAPI(OfficialPublicModules.from_mapping(modules))
            data_module = api.prepare_training_data_module(
                cache_path=cache_root,
                batch_size=int(config.training.train_batch_size.value),
                cell_type=cell_types[0],
            )
            covered_targets = api.require_perturbation_coverage(
                data_module,
                [
                    *training_data.split.train_conditions,
                    *training_data.split.val_conditions,
                    *training_data.split.test_conditions,
                ],
            )
            torch = modules["torch"]
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            model = api.build_model(
                data_module=data_module,
                model_args=official_config["model"],
                graph_args=official_config["graph"],
                learning_rate=float(config.training.learning_rate.value),
                weight_decay=float(config.training.weight_decay.value),
                device=device,
                match_control_for_eval=True,
            )
            checkpoint_path = destination / "checkpoints" / "epoch-001.ckpt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer = api.fit_one_epoch(
                model=model,
                training_only_data_module=data_module,
                checkpoint_path=checkpoint_path,
                accelerator="gpu" if device.startswith("cuda") else "cpu",
            )
            completed_epochs = int(trainer.current_epoch)
            optimizer_steps = int(trainer.global_step)
            if completed_epochs != 1 or optimizer_steps <= 0:
                raise RuntimeError(
                    "official TxPert smoke did not complete exactly one optimizer epoch"
                )
            checkpoint_sha256 = sha256_file(checkpoint_path)
            atomic_json(
                small_root / "training_receipt.json",
                {
                    "schema_version": "official-training-receipt-v1",
                    "model_id": config.model_id,
                    "dataset_id": config.dataset_id,
                    "epochs_requested": 1,
                    "epochs_completed": completed_epochs,
                    "optimizer_steps": optimizer_steps,
                    "official_training_api": (
                        "lightning.Trainer.fit(gspp.predictor.PertPredictor)"
                    ),
                    "train_batch_size": int(config.training.train_batch_size.value),
                    "eval_batch_size": int(config.training.eval_batch_size.value),
                    "optimizer": str(config.training.optimizer.value),
                    "learning_rate": float(config.training.learning_rate.value),
                    "weight_decay": float(config.training.weight_decay.value),
                    "scheduler": str(config.training.scheduler.value),
                    "validation_batches_during_fit": 0,
                    "canonical_test_truth_present_during_fit": False,
                    "checkpoint_sha256": checkpoint_sha256,
                },
            )
            # Canonical test Truth is not opened until official fit and
            # checkpoint sealing have both completed.
            with CanonicalEvaluationData(
                dataset_id=config.dataset_id,
                protocol_id=config.data.protocol_id,
                split_name="test",
                data_root=data_root,
            ) as test_data:
                predictions: list[PredictionConditionArrays] = []
                for draw in test_data.control_manifest.draws:
                    controls = test_data.load_control_rows(tuple(draw.ordered_row_ids))
                    targets = tuple(part for part in draw.condition_id.split("+") if part != "ctrl")
                    predicted = api.predict_exact_controls(
                        trained_model=model,
                        perturbation_genes=targets,
                        perturbation_to_id=data_module.pert2id,
                        input_controls=controls.expression,
                        batch_size=int(config.training.eval_batch_size.value),
                    )
                    predictions.append(
                        PredictionConditionArrays(
                            condition_id=draw.condition_id,
                            prediction=predicted,
                            input_control=controls.expression,
                            input_control_row_ids=controls.ordered_row_ids,
                        )
                    )
                sealed = seal_evaluated_run(
                    destination=destination,
                    config=config,
                    config_sha256=config_sha256,
                    run_id=run_id,
                    run_seed=1,
                    source=source,
                    environment=environment,
                    training_data=training_data,
                    test_data=test_data,
                    predictions=predictions,
                    checkpoint_sha256=checkpoint_sha256,
                )
        atomic_json(
            small_root / "official_checkout.json",
            {
                **checkout_receipt.payload(),
                "official_config_path": str(official_config_path),
                "official_config_sha256": _sha256_file(official_config_path),
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
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--data-root", type=Path)
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
            run_root=args.run_root,
            run_id=args.run_id,
            device=args.device,
            repository_root=args.repository_root,
            formal=args.formal,
            development_commit=args.development_commit,
        )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
