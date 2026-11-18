"""Fit the frozen public TxPert cross-cell GAT on the sealed C2 partition."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import anndata
import numpy as np
import torch
from audit_crosscell_study_fits import COHORT_SHA, FOLDS_SHA, SOURCE_SHA, STUDY
from evaluate_crosscell_k562 import SERVER, read_rows, source_reference
from omegaconf import OmegaConf

from benchmarks.common import official_module_session
from benchmarks.common.study_fit_partition import crosscell_fit_rows, open_crosscell_fit_partition
from benchmarks.txpert.official_api import OfficialPublicAPI, OfficialPublicModules
from benchmarks.txpert.released_recipe import CROSS_CELL_GAT, ZENODO_RECORD
from benchmarks.txpert.runner import _materialize_official_cell_context
from benchmarks.txpert.runtime import inspect_cuda_runtime, load_runtime_contract
from gradpert.data._io import atomic_json, read_json
from gradpert.evaluation.metrics import compute_condition_metrics_v2, macro_summarize
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.v2.crosscell import frozen_rows
from scripts.v2.run_group import lock
from scripts.v2.run_small_softmax import admit_selected_idle
from scripts.v2.shared_gpu import process_info, snapshot

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = SERVER / "upstreams/txpert"
UPSTREAM_COMMIT = "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
OFFICIAL_CONFIG_SHA = "36c11ed5e188301c699ba2ef853493808683368f9a94f8005c857ac3e4396048"
CROSS = SERVER / "crosscell-fixed-v1"
RECIPE = CROSS_CELL_GAT


def cohort_inputs():
    path = STUDY / "frozen/cohorts.json"
    source = CROSS / "source/K562_cross_cell_lines/de_adata_test.h5ad"
    fold_path = CROSS / "folds-baadb85/manifest.json"
    for candidate, digest in ((path, COHORT_SHA), (source, SOURCE_SHA), (fold_path, FOLDS_SHA)):
        if sha256_file(candidate) != digest:
            raise ValueError(f"fixed study input changed: {candidate}")
    cross = read_json(path)["cross_cell"]
    cohort = cross["engineered_k562"]
    fold = read_json(fold_path)["folds"][cross["target_cell_line"]]
    archive = fold_path.parent / fold["row_index_file"]
    if not sha256_file(archive) == fold["row_index_sha256"] == cross["frozen_fold_sha256"]:
        raise ValueError("frozen fold archive changed")
    adata = anndata.read_h5ad(source, backed="r")
    try:
        if sha256_json(list(adata.var_names)) != cross["expression_gene_order_sha256"]:
            raise ValueError("frozen cross-cell expression axis changed")
        rows = frozen_rows(archive, fold, adata.obs)
    finally:
        adata.file.close()
    return source, cross, cohort, rows


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


def score(model, data, api, metrics, source, cross, cohort, rows, compatibility, output):
    ids = cohort["condition_ids"]
    draw_path = Path(cohort["shared_frozen_draws_path"])
    if sha256_file(draw_path) != cross["shared_frozen_draws_sha256"]:
        raise ValueError("paired control and truth rows changed")
    frozen = read_json(draw_path)["draws"]
    draws = {condition: frozen[condition] for condition in ids}
    if sha256_json(draws) != cohort["per_condition_pairing_sha256"]:
        raise ValueError("C2 paired test cohort changed")
    adata = anndata.read_h5ad(source, backed="r")
    try:
        conditions = adata.obs["condition"].astype(str).to_numpy()
        batches = adata.obs["batch"].astype(str).to_numpy()
        target_controls = rows["target_control"]
        fit = crosscell_fit_rows(rows, conditions, cohort, compatibility=compatibility)
        reference = source_reference(adata, conditions, np.sort(np.r_[fit["train"], fit["val"]]))
        predicted = {}
        for index, (condition, draw) in enumerate(draws.items(), start=1):
            control_ids = np.asarray(draw["input_control_rows"])
            if len(control_ids) != 300 or not np.isin(control_ids, target_controls).all():
                raise ValueError("prediction requires 300 ordered K562 controls")
            controls = read_rows(adata, control_ids)
            predicted[condition] = (
                api.predict_exact_controls(
                    trained_model=model,
                    perturbation_genes=tuple(condition.split("+")),
                    perturbation_to_id=data.pert2id,
                    input_controls=controls,
                    batch_size=data.batch_size,
                ),
                controls,
            )
            if index == 1 or index % 10 == 0 or index == len(ids):
                atomic_json(output / "evaluation_progress.json", {"done": index, "total": len(ids)})
        evaluation_rows = np.sort(
            np.concatenate([target_controls] + [d["truth_rows"] for d in draws.values()])
        )
        reference_adata = anndata.AnnData(
            X=adata.X[evaluation_rows], obs=adata.obs.iloc[evaluation_rows].copy()
        )
        metrics.cache_perturbation_means(reference_adata, cross["target_cell_line"])
        if len(metrics.PERT_MEANS[cross["target_cell_line"]]) != len(ids):
            raise ValueError("official retrieval candidate count differs from the frozen cohort")
        retrieval = metrics.RetrievalMetric(subset=False)
        pearson, mse, rank = [], [], []
        scores_path = output / "conditions.jsonl"
        with scores_path.open("x") as stream:
            for condition, (prediction, controls) in predicted.items():
                truth_rows = np.asarray(draws[condition]["truth_rows"])
                if not np.array_equal(
                    truth_rows, rows["test"][conditions[rows["test"]] == condition]
                ):
                    raise ValueError(f"frozen truth IDs changed for {condition}")
                truth = read_rows(adata, truth_rows)
                matching = target_controls[np.isin(batches[target_controls], batches[truth_rows])]
                control_mean = np.asarray(adata.X[matching].mean(axis=0)).ravel()
                values = compute_condition_metrics_v2(
                    condition_id=condition,
                    prediction=prediction,
                    input_control=controls,
                    truth=truth,
                    metric_control_pool_mean=control_mean,
                    de_gene_indices=(),
                    systema_reference=reference,
                    include_deg=False,
                )
                pearson.append(values)
                error = prediction.mean(0, dtype=np.float64) - truth.mean(0, dtype=np.float64)
                mse.append(float(np.mean(error**2)))
                rank.append(
                    float(
                        retrieval(
                            prediction,
                            truth,
                            cross["target_cell_line"],
                            controls,
                            reference_adata,
                            condition,
                        )
                    )
                )
                stream.write(
                    json.dumps(
                        {**asdict(values), "mean_mse": mse[-1], "retrieval": rank[-1]},
                        allow_nan=False,
                    )
                    + "\n"
                )
    finally:
        adata.file.close()
    return {
        "conditions": len(ids),
        "condition_ids_sha256": cohort["condition_ids_sha256"],
        "paired_rows_sha256": cohort["per_condition_pairing_sha256"],
        "test_panel_genes": cross["expression_panel_genes"],
        "test_controls_per_condition": 300,
        "test_updates": 0,
        "mean_mse": float(np.mean(mse)),
        "retrieval_mean": float(np.mean(rank)),
        "pearson": [asdict(value) for value in macro_summarize(pearson)],
        "conditions_file_sha256": sha256_file(scores_path),
    }


def fit(args, output, identity):
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
    if args.smoke_run:
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
            raise ValueError("official C2 optimizer schedule differed from the published recipe")
        receipt.update(
            epochs_completed=trainer.gradpert_last_epoch,
            best_epoch=trainer.gradpert_best_epoch,
            checkpoint_sha256=sha256_file(best),
        )
        if args.smoke:
            return {**receipt, "status": "trained_validation_only"}
        saved = torch.load(best, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"], strict=True)
        api.restore_post_fit_device(model, "cuda:0")
        model.eval().requires_grad_(False)
        for property_name in ("train_data", "val_data"):
            delattr(data, property_name)
        del saved, trainer
        gc.collect()
        torch.cuda.empty_cache()
        scored = score(
            model,
            data,
            api,
            modules["gspp.metrics"],
            source,
            cross,
            cohort,
            rows,
            compatibility,
            output,
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
