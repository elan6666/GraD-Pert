"""Run the released scFoundation/GEARS adapter on the fixed Jurkat J2 data."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch

from benchmarks.crosscell import scfoundation_c2 as released
from benchmarks.crosscell.remaining_c2_readiness import hgnc_links, reconcile
from benchmarks.crosscell.study_paths import COHORT_SHA, REPO, SERVER, STUDY
from benchmarks.jurkat_j2.study import (
    VOCABULARY,
    context,
    evaluate,
    evaluate_segment,
    fit_data,
    frozen_receipt,
    score_axis,
)
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot


def initialize(output, fit_batches, *, frozen_graph=None):
    Gears, network = released.released_module(require_crosscell_artifacts=False)
    from gears.utils import get_coexpression_network_from_train

    prepared = fit_data(released.SEED)
    genes = np.asarray(prepared.gene_ids)
    columns = score_axis(genes)
    visible = genes[columns].tolist()
    with VOCABULARY.open() as stream:
        native = [x["gene_name"] for x in csv.DictReader(stream, delimiter="\t")]
    gene_map = {g: i for i, g in enumerate(native)}
    links = hgnc_links()
    expression = reconcile(visible, set(native), links, verified_aliases={})
    mapped = [expression["aliases"].get(g, g) for g in visible]
    positions = np.asarray([gene_map[g] for g in mapped])
    with context() as data:
        tests = tuple(data.split.test_conditions)
    names = (*prepared.train_conditions, *prepared.val_conditions, *tests)
    targets = {g for n in names for g in n.split("+") if g != "ctrl"}
    with released.GO.open("rb") as stream:
        support = reconcile(targets, set(pickle.load(stream)), links, verified_aliases={})
    if support["unresolved"] or support["collisions"]:
        raise ValueError("unsupported GO targets in the frozen J2 panel")
    index = {str(row): i for i, row in enumerate(prepared.adata.obs_names)}
    roles = {
        key: np.asarray([index[row] for row in values])
        for key, values in (
            ("train", prepared.train_row_ids),
            ("val", prepared.val_row_ids),
            ("source_control", prepared.control_row_ids),
        )
    }
    conditions = prepared.adata.obs["condition"].astype(str).to_numpy()
    matrix = prepared.adata.X[:, columns].tocsr()
    coex = frozen_graph if frozen_graph is not None else output / "coexpression.npz"
    if not coex.is_file():
        rows = roles["train"]
        data = anndata.AnnData(
            matrix[rows],
            obs=pd.DataFrame({"condition": conditions[rows]}, index=rows.astype(str)),
            var=pd.DataFrame({"gene_name": mapped}, index=mapped),
        )
        (output / "j2").mkdir(exist_ok=True)
        edges = get_coexpression_network_from_train(
            data,
            0.4,
            20,
            str(output),
            "j2",
            "frozen",
            released.SEED,
            0.75,
            {"train": list(prepared.train_conditions)},
        )
        if edges.empty or not np.isfinite(edges.importance).all():
            raise ValueError("the official J2 graph has no usable co-expression edges")
        graph = network(edges, native, gene_map)
        np.savez(coex, edge_index=graph.edge_index.numpy(), edge_weight=graph.edge_weight.numpy())
    graph = released.read_coexpression(coex)
    by_context = defaultdict(list)
    for row in roles["source_control"]:
        by_context[prepared.row_context_ids[row]].append(row)
    rng = np.random.default_rng(released.SEED)
    paired = np.full(len(conditions), -1, dtype=int)
    for row in np.r_[roles["train"], roles["val"]]:
        paired[row] = rng.choice(by_context[prepared.row_context_ids[row]])
    model, positions, target_map, val_loader, provenance = released.initialize_fitting(
        output,
        Gears,
        network,
        native,
        positions,
        support["aliases"],
        matrix,
        conditions,
        roles,
        paired,
        tests,
        graph,
        visible,
        fit_batches,
        20,
    )
    return model, positions, target_map, val_loader, provenance, columns


J2_SCORE_BATCHES = (6, 12, 24)
FROZEN_SMOKE_COMMIT = "bf55526b24bd80cacb71001c15ca4df68dbeabd0"


def select_score_batch(predict, name, values):
    """Measure ordered source controls and accept only equivalent predictions."""
    rows = []
    reference = None
    for size in J2_SCORE_BATCHES:
        try:
            predict(name, values[:size], size)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            predicted = predict(name, values, size)
        except torch.cuda.OutOfMemoryError:
            if size == released.BATCH:
                raise
            rows.append({"batch": size, "status": "out_of_memory"})
        else:
            seconds = time.monotonic() - start
            if reference is None:
                if predicted.shape != values.shape or not np.isfinite(predicted).all():
                    raise ValueError("invalid official validation prediction")
                reference = predicted
            same_shape = predicted.shape == reference.shape
            finite = np.isfinite(predicted).all()
            equivalent = (
                same_shape and finite and np.allclose(predicted, reference, rtol=1e-5, atol=1e-5)
            )
            rows.append(
                {
                    "batch": size,
                    "status": "equivalent" if equivalent else "prediction_drift",
                    "seconds_per_300_controls": seconds,
                    "cells_per_second": len(values) / seconds,
                    "max_abs_difference": float(np.max(np.abs(predicted - reference)))
                    if same_shape and finite
                    else None,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            del predicted
        torch.cuda.empty_cache()
    baseline = rows[0]
    viable = [
        row
        for row in rows
        if row["status"] == "equivalent"
        and row["seconds_per_300_controls"] < baseline["seconds_per_300_controls"] * 0.95
    ]
    selected = min(viable, key=lambda row: row["seconds_per_300_controls"], default=baseline)
    return selected["batch"], rows


def run(output, identity, *, smoke, fit_batches, resume, assist=None):
    started = time.monotonic()
    torch.manual_seed(released.SEED)
    parent = assist[0] if assist is not None else output
    checkpoint = parent / "checkpoint.pt"
    coexpression = parent / "coexpression.npz"
    if resume:
        fit_record = frozen_receipt(parent, {**identity, "run_id": parent.name}, checkpoint.name)
        if sha256_file(coexpression) != fit_record["coexpression_sha256"]:
            raise ValueError("the frozen J2 co-expression graph changed")
    model, positions, targets, val_loader, provenance, columns = initialize(
        output, fit_batches, frozen_graph=coexpression if resume else None
    )
    if resume:
        frozen = model.model.cuda()
    else:
        begin = time.monotonic()
        model.train(epochs=1, result_dir=str(output), lr=0.01, weight_decay=5e-4)
        frozen = model.best_model
        fit_record = {
            **identity,
            **provenance,
            "checkpoint_sha256": sha256_file(checkpoint),
            "coexpression_sha256": sha256_file(output / "coexpression.npz"),
            "fit_seconds": time.monotonic() - begin,
            "optimizer_updates": (provenance["fit_microbatches"] + released.ACCUMULATION - 1)
            // released.ACCUMULATION,
            "checkpoint_epoch": 1,
            "validation_de_basis": "shared_train_only_de_for_unseen_val",
            "test_updates": 0,
        }
        atomic_json(output / "training_result.json", fit_record)
    frozen.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    frozen.eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(name, values, batch_size):
        predictions = []
        for i in range(0, len(values), batch_size):
            chunk = values[i : i + batch_size]
            batch = released.Batch(
                chunk, positions, [targets[name]] * len(chunk), [name] * len(chunk)
            )
            predictions.append(frozen(batch.to("cuda"))[:, positions].cpu().numpy())
        return np.concatenate(predictions)

    if smoke:
        sample = next(iter(val_loader))
        values = sample.x.reshape(-1, released.GENES + 1).numpy()[:, positions]
        begin = time.monotonic()
        trial = predict(sample.pert[0], np.resize(values, (300, len(positions))), released.BATCH)
        if trial.shape != (300, len(columns)) or not np.isfinite(trial).all():
            raise ValueError("the source-only J2 smoke did not return finite predictions")
        atomic_json(
            output / "result.json",
            {
                **fit_record,
                "status": "trained_validation_only",
                "test_conditions_accessed": 0,
                "validation_prediction_shape": list(trial.shape),
                "seconds_per_condition": time.monotonic() - begin,
            },
        )
        return
    batch_receipt = parent / "inference-batch.json"
    if assist is not None and not batch_receipt.is_file():
        raise ValueError("the primary evaluator has not pinned its inference batch")
    if batch_receipt.exists():
        batch_plan = read_json(batch_receipt)
        if (
            batch_plan["checkpoint_sha256"] != fit_record["checkpoint_sha256"]
            or batch_plan["selected_batch"] not in J2_SCORE_BATCHES
        ):
            raise ValueError("the checkpoint or frozen inference batch changed")
        score_batch = batch_plan["selected_batch"]
    elif (output / "condition-receipts").exists():
        score_batch = released.BATCH
    else:
        source = np.unique(val_loader.paired[val_loader.paired >= 0])
        selected = np.random.default_rng(released.SEED).choice(source, 300, replace=False)
        values = val_loader.matrix[selected].toarray()
        condition = val_loader.conditions[val_loader.rows[0]]
        score_batch, trials = select_score_batch(predict, condition, values)
        atomic_json(
            batch_receipt,
            {
                "checkpoint_sha256": fit_record["checkpoint_sha256"],
                "selected_batch": score_batch,
                "validation_controls": 300,
                "test_conditions_accessed": 0,
                "candidate_batches": trials,
            },
        )

    def predict_projected(name, values, _ids, axis):
        return predict(name, values[:, axis], score_batch)

    if assist is not None:
        measured = evaluate(
            predict_projected,
            output,
            deadline=started + 10500,
            journal_identity=fit_record["checkpoint_sha256"],
            assist=assist,
        )
        if sha256_file(checkpoint) != fit_record["checkpoint_sha256"]:
            raise ValueError("the assisted checkpoint changed")
        atomic_json(
            output / "result.json",
            {
                **identity,
                **measured,
                "status": "evaluation_partition_complete",
                "primary_run_id": parent.name,
                "checkpoint_sha256": fit_record["checkpoint_sha256"],
                "inference_batch_receipt_sha256": sha256_file(batch_receipt),
            },
        )
        return
    measured = evaluate_segment(
        predict_projected,
        output,
        fit_record,
        deadline=started + 10500,
    )
    if measured is None:
        return
    if sha256_file(checkpoint) != fit_record["checkpoint_sha256"]:
        raise ValueError("the frozen GEARS checkpoint changed")
    atomic_json(
        output / "result.json",
        {
            **fit_record,
            **measured,
            "status": "evaluated",
            "inference_batch": score_batch,
            "inference_batch_receipt_sha256": sha256_file(batch_receipt)
            if batch_receipt.exists()
            else None,
            "evaluation_source_commit": identity["source_commit"],
        },
    )


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
    phase.add_argument("--resume", action="store_true")
    phase.add_argument("--assist-parent", type=Path)
    parser.add_argument("--assist-start", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != STUDY or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("invalid J2 study path or GPU")
    assist = None
    if args.assist_parent is not None:
        parent = args.assist_parent.resolve()
        if (
            parent.parent != STUDY
            or parent == output
            or args.assist_start is None
            or not 1 <= args.assist_start <= 592
        ):
            raise ValueError("invalid J2 primary run or condition partition")
        assist = (parent, args.assist_start)
    elif args.assist_start is not None:
        raise ValueError("--assist-start requires --assist-parent")
    source = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    identity = {
        "model": "scfoundation",
        "scenario": "J2",
        "run_id": output.name,
        "source_commit": source.commit,
        "official_commit": released.COMMIT,
        "foundation_weights_sha256": released.FOUNDATION_WEIGHTS_SHA,
        "cohort_sha256": COHORT_SHA,
        "formal_eligible": source.formal_eligible,
        "gpu": args.gpu,
        "config_sha256": sha256_json({"optimizer_updates": 300, "batch": 6, "seed": 1}),
    }
    if args.smoke_run:
        smoke = read_json(args.smoke_run / "result.json")
        if (
            any(
                smoke[k] != v
                for k, v in identity.items()
                if k not in {"run_id", "gpu", "source_commit"}
            )
            or smoke["source_commit"] not in {source.commit, FROZEN_SMOKE_COMMIT}
            or smoke["test_conditions_accessed"] != 0
            or smoke["status"] != "trained_validation_only"
            or smoke["validation_prediction_shape"] != [300, 4486]
        ):
            raise ValueError("the official J2 smoke did not pass")
    with lock(SERVER / "runtime" / f"v2-gpu-{args.gpu}.lock"):
        admit_selected_idle(snapshot(os.getpid()), args.gpu, args.gpu_uuid)
        if not args.resume:
            output.mkdir()
        atomic_json(output / "worker.json", {"owner": process_info(os.getpid()), "gpu": args.gpu})
        run(
            output,
            identity,
            smoke=args.smoke,
            fit_batches=5 if args.smoke else 1500,
            resume=args.resume or assist is not None,
            assist=assist,
        )


if __name__ == "__main__":
    main()
