"""Evaluate the published TxPert GAT on the frozen natural K562 cohort."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import anndata
import joblib
import lightning
import numpy as np
import torch
from evaluate_crosscell_k562 import SOURCE, read_rows, source_reference
from freeze_perturbation_holdout_study import CROSS, ROOT
from gspp import metrics, predictor
from gspp.data import datamodule, graphmodule
from omegaconf import OmegaConf

from benchmarks.txpert.official_api import OfficialPublicAPI, OfficialPublicModules
from gradpert.data._io import atomic_json, read_json
from gradpert.evaluation.metrics import compute_condition_metrics_v2, macro_summarize
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.run_group import lock
from scripts.v2.run_small_softmax import admit_selected_idle
from scripts.v2.shared_gpu import process_info, snapshot

UPSTREAM = ROOT / "upstreams/txpert"
COMMIT = "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
WEIGHT_NAME = "K562_unseen_cell_gat.ckpt"


def evaluate(args, owner):
    config_path = UPSTREAM / "configs/config-x-cell-gat.yaml"
    config = OmegaConf.load(config_path)
    if config.checkpoint_name != WEIGHT_NAME or args.checkpoint.name != WEIGHT_NAME:
        raise ValueError("unexpected TxPert checkpoint or recipe")
    if not Path(predictor.__file__).resolve().is_relative_to(UPSTREAM.resolve()):
        raise ValueError("TxPert import did not come from the frozen checkout")
    if (
        subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip()
        != COMMIT
        or subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    ):
        raise ValueError("TxPert upstream differs from the pinned source")

    cross = read_json(args.cohorts)["cross_cell"]
    cohort = cross["natural_k562"]
    condition_ids = cohort["condition_ids"]
    draws_path = Path(cohort["shared_frozen_draws_path"])
    if sha256_file(draws_path) != cross["shared_frozen_draws_sha256"]:
        raise ValueError("frozen 300-control draws changed")
    all_draws = read_json(draws_path)["draws"]
    draws = {condition: all_draws[condition] for condition in condition_ids}
    if sha256_json(draws) != cohort["per_condition_pairing_sha256"]:
        raise ValueError("paired test cohort changed")

    data = datamodule.PertDataModule(**OmegaConf.to_container(config.datamodule, resolve=True))
    cached = CROSS / "source/K562_cross_cell_lines"
    predefined = joblib.load(cached / "splits/train_test_split.pkl")
    train, val, test = (set(predefined[role]) for role in ("train", "val", "test"))
    if (
        set(condition_ids) & (train | val)
        or set(condition_ids) - test
        or set(data.train_cell_types) != set(cross["source_cell_lines"])
        or data.val_cell_type is not None
        or data.test_cell_type != cross["target_cell_line"]
        or any(gene not in data.pert2id for c in condition_ids for gene in c.split("+"))
    ):
        raise ValueError("published recipe does not cover the locked natural K562 cohort")

    gspp_graph = graphmodule.GSPGraph(
        pert2id=data.pert2id,
        gene2id=data.gene2id,
        **OmegaConf.to_container(config.graph, resolve=True),
    )
    count = cross["expression_panel_genes"]
    model = (
        predictor.PertPredictor.load_from_checkpoint(
            args.checkpoint,
            strict=True,
            input_dim=count,
            output_dim=count,
            adata_output_dim=count,
            num_perts=len(data.pert2id),
            graph=gspp_graph,
            pert_names=list(data.pert2id),
            model_args=OmegaConf.to_container(config.model, resolve=True),
            device="cuda:0",
            map_location="cpu",
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    if model.basal_mode != "construct_control":
        raise ValueError("official frozen predict and forward paths differ for this checkpoint")
    api = OfficialPublicAPI(
        OfficialPublicModules(predictor, datamodule, graphmodule, lightning, torch)
    )
    input_path = cached / "de_adata_test.h5ad"
    if sha256_file(input_path) != cross["source_h5ad_sha256"]:
        raise ValueError("cross-cell expression input changed")
    adata = anndata.read_h5ad(input_path, backed="r")
    try:
        if sha256_json(list(map(str, adata.var_names))) != cross["expression_gene_order_sha256"]:
            raise ValueError("cross-cell gene order changed")
        index_path = CROSS / "folds-baadb85/fold-K562.npz"
        if sha256_file(index_path) != cross["frozen_fold_sha256"]:
            raise ValueError("source reference row index changed")
        with np.load(index_path, allow_pickle=False) as archive:
            fit_rows = np.sort(np.r_[archive["train"], archive["val"]])
            ctrl_rows = archive["target_control"]
        conditions = adata.obs["condition"].astype(str).to_numpy()
        cell_lines = adata.obs["cell_line"].astype(str).to_numpy()
        if set(conditions[fit_rows]) & set(condition_ids) or "K562" in cell_lines[fit_rows]:
            raise ValueError("common source reference contains a held-out condition or line")
        reference = source_reference(adata, conditions, fit_rows)
        predictions = {}
        for position, condition in enumerate(condition_ids, start=1):
            draw = draws[condition]
            input_rows = np.asarray(draw["input_control_rows"])
            if not np.isin(input_rows, ctrl_rows).all():
                raise ValueError("predictor received non-control inputs")
            controls = read_rows(adata, input_rows)
            predicted = api.predict_exact_controls(
                trained_model=model,
                perturbation_genes=condition.split("+"),
                perturbation_to_id=data.pert2id,
                input_controls=controls,
                batch_size=int(config.datamodule.batch_size),
            )
            predictions[condition] = (controls, predicted)
            if position == 1 or position % 10 == 0 or position == len(condition_ids):
                atomic_json(
                    args.output / "progress.json",
                    {"conditions_inferred": position, "conditions_total": len(condition_ids)},
                )

        # Evaluate only after all frozen predictions are complete.
        ids = np.sort(np.concatenate([ctrl_rows] + [d["truth_rows"] for d in draws.values()]))
        selected = anndata.AnnData(X=adata.X[ids], obs=adata.obs.iloc[ids].copy())
        metrics.cache_perturbation_means(selected, "K562")
        retrieve = metrics.RetrievalMetric(subset=False)
        if len(metrics.PERT_MEANS["K562"]) != len(condition_ids):
            raise ValueError("official retrieval candidate count changed")
        qc = []
        seen_scores = []
        batches = adata.obs["batch"].astype(str).to_numpy()
        with (args.output / "conditions.jsonl").open("x") as stream:
            for condition, (controls, predicted) in predictions.items():
                truth_ids = draws[condition]["truth_rows"]
                truth = read_rows(adata, truth_ids)
                use = ctrl_rows[np.isin(batches[ctrl_rows], batches[truth_ids])]
                mean_control = np.asarray(adata.X[use].mean(axis=0)).reshape(-1)
                result = compute_condition_metrics_v2(
                    condition_id=condition,
                    prediction=predicted,
                    input_control=controls,
                    truth=truth,
                    metric_control_pool_mean=mean_control,
                    de_gene_indices=(),
                    systema_reference=reference,
                    include_deg=False,
                )
                seen_scores.append(result)
                rank = float(retrieve(predicted, truth, "K562", controls, selected, condition))
                mse = float(
                    np.mean(
                        (predicted.mean(0, dtype=np.float64) - truth.mean(0, dtype=np.float64)) ** 2
                    )
                )
                qc.append((rank, mse))
                stream.write(
                    json.dumps(
                        {**asdict(result), "retrieval": rank, "mean_mse": mse}, allow_nan=False
                    )
                    + "\n"
                )
    finally:
        adata.file.close()
    condition_file = args.output / "conditions.jsonl"
    return {
        **owner,
        "configuration_sha256": sha256_file(config_path),
        "official_split_sha256": sha256_file(cached / "splits/train_test_split.pkl"),
        "published_protocol_fit_exposure": (
            "inferred from linked official config/cache, no per-row training receipt"
        ),
        "split_name": "crosscell_natural_unseen_41_fixed_panel",
        "conditions": len(qc),
        "controls_per_condition": 300,
        "genes": count,
        "optimizer_updates": 0,
        "retrieval_candidates": len(condition_ids),
        "retrieval_mean": float(np.mean([rank for rank, _ in qc])),
        "mean_mse": float(np.mean([mse for _, mse in qc])),
        "pearson": [asdict(score) for score in macro_summarize(seen_scores)],
        "condition_scores_sha256": sha256_file(condition_file),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", required=True, type=Path)
    parser.add_argument("--cohorts-sha", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publication", required=True, type=Path)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--gpu", required=True, choices=("0", "1"))
    parser.add_argument("--uuid", required=True)
    args = parser.parse_args()
    if any(
        not p.resolve().is_relative_to(ROOT)
        for p in (args.cohorts, args.checkpoint, args.output, args.publication)
    ):
        raise ValueError("scientific evaluation stays on the server")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu
        or os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("CUDA execution envelope changed")
    if (
        sha256_file(args.cohorts) != args.cohorts_sha
        or sha256_file(args.checkpoint) != args.checkpoint_sha
    ):
        raise ValueError("cohort or weights changed")
    identity = inspect_source_identity(
        SOURCE,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    with lock(ROOT / "runtime" / f"v2-gpu-{args.gpu}.lock"):
        admit_selected_idle(snapshot(os.getpid()), int(args.gpu), args.uuid)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        args.output.mkdir(exist_ok=False)
        started = time.monotonic()
        owner = {
            "source_commit": identity.commit,
            "upstream_commit": COMMIT,
            "cohorts_sha256": args.cohorts_sha,
            "checkpoint_sha256": args.checkpoint_sha,
            "owner": process_info(os.getpid()),
            "gpu": args.gpu,
            "gpu_uuid": args.uuid,
        }
        atomic_json(args.output / "OWNER.json", owner)
        try:
            receipt = evaluate(args, owner)
            if sha256_file(args.checkpoint) != args.checkpoint_sha:
                raise ValueError("weights changed during evaluation")
            atomic_json(
                args.output / "COMPLETE.json",
                {**receipt, "wall_seconds": time.monotonic() - started},
            )
        except BaseException as error:
            atomic_json(
                args.output / "FAILURE.json",
                {**owner, "error": str(error), "error_type": type(error).__name__},
            )
            raise


if __name__ == "__main__":
    main()
