"""Fit the frozen public TxPert cross-cell GAT on the sealed C2 partition."""

from __future__ import annotations

import argparse
import gc
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from benchmarks.common import official_module_session
from benchmarks.common.study_fit_partition import open_crosscell_fit_partition
from benchmarks.crosscell.common_score_panel import load_panel
from benchmarks.crosscell.study_data import cohort_inputs, score
from benchmarks.crosscell.study_paths import COHORT_SHA, REPO, SERVER, STUDY
from benchmarks.crosscell.study_paths import TXPERT_CHECKOUT as UPSTREAM
from benchmarks.crosscell.study_paths import TXPERT_COMMIT as UPSTREAM_COMMIT
from benchmarks.txpert.official_api import OfficialPublicAPI, OfficialPublicModules
from benchmarks.txpert.released_recipe import CROSS_CELL_GAT, ZENODO_RECORD
from benchmarks.txpert.runner import _materialize_official_cell_context
from benchmarks.txpert.runtime import inspect_cuda_runtime, load_runtime_contract
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file
from scripts.v2.shared_gpu import process_info, snapshot

OFFICIAL_CONFIG_SHA = "36c11ed5e188301c699ba2ef853493808683368f9a94f8005c857ac3e4396048"
RECIPE = CROSS_CELL_GAT


def bind_official_datasets(official, config, prepared, cross, cohort):
    data_args = dict(config.datamodule)
    data_args["mode"] = "baseline"
    data = official.PertDataModule(**data_args)
    if (
        set(data.train_cell_types) != set(cross["source_cell_lines"])
        or data.val_cell_type is not None
        or data.test_cell_type != cross["target_cell_line"]
    ):
        raise ValueError("public TxPert dataset context differs from the frozen fold")
    _materialize_official_cell_context(prepared.adata)
    data.adata = prepared.adata
    data._split_data()
    controls = set(prepared.control_row_ids)
    if tuple(data.treatment_data.obs_names) != tuple(
        row for row in prepared.adata.obs_names if row not in controls
    ):
        raise ValueError("official TxPert dropped a frozen treatment row")
    if tuple(data.control_data_train.obs_names) != prepared.control_row_ids:
        raise ValueError("official TxPert dropped or changed source controls")
    data.condition_group = {
        "train": list(prepared.train_conditions),
        "val": list(prepared.val_conditions),
        "test": list(cohort["condition_ids"]),
    }
    for role, row_ids in (("train", prepared.train_row_ids), ("val", prepared.val_row_ids)):
        exact = data.treatment_data[list(row_ids)]
        dataset = official.XcelltypePerturbDataset(
            exact,
            data.control_data_train,
            data.conditions.loc[list(row_ids)],
            data.condition_names.loc[list(row_ids)],
            data.device,
            train_cond_names=data.control_train_condition_names if role == "train" else None,
            match_cntr=data.match_cntr,
            avg_cntr=data.avg_cntr,
            obsm_key=data.obsm_key,
        )
        setattr(data, f"{role}_data", dataset)
    data.data_prepared = True
    return data


def fit(args, output, identity, *, replay_checkpoint=None):
    model_panel = None
    source, cross, cohort, rows = cohort_inputs()
    if sha256_file(args.fit_compatibility) != args.fit_compatibility_sha:
        raise ValueError("C2 shared fit support manifest changed")
    compatibility = read_json(args.fit_compatibility)
    if (
        compatibility["source_cohort_sha256"] != COHORT_SHA
        or compatibility["upstream_commit"] != UPSTREAM_COMMIT
        or compatibility["vocabulary_sha256"] != sha256_file(UPSTREAM / "data/gears_gene_set.csv")
    ):
        raise ValueError("C2 shared fit support is not tied to the official vocabulary and cohort")
    prepared = open_crosscell_fit_partition(
        source=source,
        verified_rows=rows,
        cohort=cohort,
        mask_expression=True,
        fit_compatibility=compatibility,
    )
    atomic_json(output / "double_holdout_fit.json", prepared.receipt)
    config_path = UPSTREAM / RECIPE.config_file
    if sha256_file(config_path) != OFFICIAL_CONFIG_SHA:
        raise ValueError("frozen TxPert public configuration changed")
    config = OmegaConf.to_object(OmegaConf.load(config_path))
    receipt = {
        "run_id": output.name,
        "scenario": "C2",
        "formal_eligible": identity.formal_eligible,
        "source_commit": identity.commit,
        "cohort_sha256": COHORT_SHA,
        "configuration_sha256": OFFICIAL_CONFIG_SHA,
        "shared_fit_support_file_sha256": args.fit_compatibility_sha,
        "fit_sha256": sha256_file(output / "double_holdout_fit.json"),
        "initialization": "from_scratch",
        "recipe": {**asdict(RECIPE), "source": ZENODO_RECORD},
    }
    if replay_checkpoint is None and args.smoke_run:
        smoke = read_json(args.smoke_run / "result.json")
        for field in (
            "scenario",
            "formal_eligible",
            "source_commit",
            "cohort_sha256",
            "configuration_sha256",
            "fit_sha256",
            "shared_fit_support_file_sha256",
            "initialization",
            "recipe",
        ):
            if smoke[field] != receipt[field]:
                raise ValueError(f"C2 smoke receipt disagrees on {field}")
        if smoke["status"] != "trained_validation_only" or smoke["epochs_completed"] != 1:
            raise ValueError("C2 must pass the one-epoch smoke on the same runtime")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    with official_module_session(
        checkout_root=UPSTREAM,
        expected_commit=UPSTREAM_COMMIT,
        module_names=(
            "gspp.predictor",
            "gspp.metrics",
            "gspp.data.datamodule",
            "gspp.data.graphmodule",
        ),
    ) as (modules, _):
        import lightning

        modules.update(torch=torch, lightning=lightning)
        api = OfficialPublicAPI(OfficialPublicModules.from_mapping(modules))
        data = bind_official_datasets(
            modules["gspp.data.datamodule"], OmegaConf.create(config), prepared, cross, cohort
        )
        api.require_perturbation_coverage(
            data, (*prepared.train_conditions, *prepared.val_conditions, *cohort["condition_ids"])
        )
        atomic_json(
            output / "training_index_adapter.json",
            api.normalize_training_perturbation_indices(data),
        )
        model = api.build_model(
            data_module=data,
            model_args=config["model"],
            graph_args=config["graph"],
            learning_rate=RECIPE.learning_rate,
            weight_decay=RECIPE.weight_decay,
            device="cuda:0",
            match_control_for_eval=config["match_cntr_for_eval"],
            scheduler_args=RECIPE.scheduler_args,
            min_lr=RECIPE.min_lr,
        )
        best, last = output / "best.ckpt", output / "last.ckpt"
        if replay_checkpoint is None:
            trainer = api.fit_with_validation(
                model=model,
                training_only_data_module=data,
                checkpoint_path=best,
                accelerator="gpu",
                epochs=1 if args.smoke else RECIPE.epochs,
                progress_path=output / "validation_progress.json",
                r50=True,
                last_checkpoint_path=last,
                released_recipe=None if args.smoke else RECIPE,
                allowed_expression_indices=prepared.allowed_indices,
            )
            schedule = trainer.lr_scheduler_configs[0].scheduler.state_dict()
            warmup, cosine = schedule["_schedulers"]
            if (
                schedule["_milestones"] != [RECIPE.warmup_epochs]
                or warmup["total_iters"] != RECIPE.warmup_epochs
                or cosine["T_max"] != RECIPE.epochs - RECIPE.warmup_epochs
                or cosine["eta_min"] != RECIPE.min_lr
                or any(
                    group["initial_lr"] != RECIPE.learning_rate
                    or group["weight_decay"] != RECIPE.weight_decay
                    for group in trainer.optimizers[0].param_groups
                )
            ):
                raise ValueError(
                    "official C2 optimizer schedule differed from the published recipe"
                )
            receipt.update(
                epochs_completed=trainer.gradpert_last_epoch,
                best_epoch=trainer.gradpert_best_epoch,
                checkpoint_sha256=sha256_file(best),
            )
            if args.smoke:
                return {**receipt, "status": "trained_validation_only"}
        else:
            best = replay_checkpoint
            receipt["checkpoint_sha256"] = sha256_file(best)
            model_panel = load_panel(source, cross, cohort)
        saved = torch.load(best, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"], strict=True)
        api.restore_post_fit_device(model, "cuda:0")
        model.eval().requires_grad_(False)
        for property_name in ("train_data", "val_data"):
            delattr(data, property_name)
        del saved
        if replay_checkpoint is None:
            del trainer
        gc.collect()
        torch.cuda.empty_cache()

        def predict_controls(condition, controls):
            return api.predict_exact_controls(
                trained_model=model,
                perturbation_genes=tuple(condition.split("+")),
                perturbation_to_id=data.pert2id,
                input_controls=controls,
                batch_size=data.batch_size,
            )

        scored = score(
            predict_controls,
            modules["gspp.metrics"],
            source,
            cross,
            cohort,
            rows,
            compatibility,
            output,
            score_genes=model_panel["score_genes"] if model_panel is not None else None,
            journal_key=receipt["checkpoint_sha256"] if replay_checkpoint is not None else None,
        )
        if any(path.suffix.lower() in {".pkl", ".pickle"} for path in output.rglob("*")):
            raise ValueError("C2 default metrics-only output contains a pickle")
        return {**receipt, **scored, "status": "evaluated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-compatibility", type=Path, required=True)
    parser.add_argument("--fit-compatibility-sha", required=True)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--smoke-run", type=Path)
    args = parser.parse_args()
    if any(not p.resolve().is_relative_to(STUDY) for p in (args.output, args.fit_compatibility)):
        raise ValueError("C2 science outputs must stay in the study root")
    identity = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    _, runtime = load_runtime_contract(repository_root=REPO, checkout_root=UPSTREAM)
    inspect_cuda_runtime(contract=runtime, device="cuda:0")
    with lock(SERVER / "runtime/v2-gpu-0.lock"):
        admit_selected_idle(snapshot(os.getpid()), 0, "GPU-23bb1466-b82b-982d-8eaf-9a0002c6a4e7")
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        args.output.mkdir(exist_ok=False)
        atomic_json(
            args.output / "owner.json", {"owner": process_info(os.getpid()), "physical_gpu": 0}
        )
        result = fit(args, args.output, identity)
        atomic_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
