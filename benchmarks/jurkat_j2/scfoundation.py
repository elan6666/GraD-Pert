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
from benchmarks.jurkat_j2.study import VOCABULARY, context, evaluate, fit_data, score_axis
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot


def initialize(output, fit_batches):
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
    coex = output / "coexpression.npz"
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


def run(output, identity, *, smoke, fit_batches, resume):
    started = time.monotonic()
    torch.manual_seed(released.SEED)
    model, positions, targets, val_loader, provenance, columns = initialize(output, fit_batches)
    checkpoint = output / "checkpoint.pt"
    if resume:
        fit_record = read_json(output / "training_result.json")
        if fit_record["checkpoint_sha256"] != sha256_file(checkpoint) or any(
            fit_record[k] != v for k, v in identity.items() if k != "gpu"
        ):
            raise ValueError("the scoped J2 evaluation cannot resume a changed checkpoint")
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
    def predict(name, values):
        predictions = []
        for i in range(0, len(values), released.BATCH):
            chunk = values[i : i + released.BATCH]
            batch = released.Batch(
                chunk, positions, [targets[name]] * len(chunk), [name] * len(chunk)
            )
            predictions.append(frozen(batch.to("cuda"))[:, positions].cpu().numpy())
        return np.concatenate(predictions)

    if smoke:
        sample = next(iter(val_loader))
        values = sample.x.reshape(-1, released.GENES + 1).numpy()[:, positions]
        begin = time.monotonic()
        trial = predict(sample.pert[0], np.resize(values, (300, len(positions))))
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
    try:
        measured = evaluate(
            lambda name, values, _ids, axis: predict(name, values[:, axis]),
            output,
            journal_identity=fit_record["checkpoint_sha256"],
            deadline=started + 10500,
        )
    except TimeoutError:
        last = read_json(output / "evaluation_progress.json")
        atomic_json(
            output / "segment.json",
            {
                "status": "evaluation_segment_complete",
                "checkpoint_sha256": fit_record["checkpoint_sha256"],
                "conditions_completed": last["done"],
                "conditions_total": last["total"],
                "test_updates": 0,
            },
        )
        return
    if sha256_file(checkpoint) != fit_record["checkpoint_sha256"]:
        raise ValueError("the frozen GEARS checkpoint changed")
    atomic_json(output / "result.json", {**fit_record, **measured, "status": "evaluated"})


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
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != STUDY or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("invalid J2 study path or GPU")
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
            any(smoke[k] != v for k, v in identity.items() if k not in {"run_id", "gpu"})
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
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
