"""Evaluate the sealed epoch-five K562 leave-one-line-out checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE), str(SOURCE / "scripts/v2")]

from gradpert.config import load_experiment_config  # noqa: E402
from gradpert.data._io import atomic_json, read_json  # noqa: E402
from gradpert.data.controls import stable_draw_seed  # noqa: E402
from gradpert.evaluation.metrics import compute_condition_metrics_v2, macro_summarize  # noqa: E402
from gradpert.execution.identity import inspect_source_identity  # noqa: E402
from gradpert.hashing import sha256_file, sha256_json  # noqa: E402
from gradpert.training.v2.checkpoint import load_evaluation_checkpoint  # noqa: E402
from gradpert.training.v2.crosscell import DATASET, frozen_rows  # noqa: E402
from gradpert.training.v2.evaluation import predict_controls  # noqa: E402
from gradpert.training.v2.evaluation_graph import prepare_evaluation_graph  # noqa: E402
from gradpert.training.v2.runtime import prepare_runtime  # noqa: E402
from scripts.v2.run_group import lock  # noqa: E402
from scripts.v2.run_small_softmax import admit_selected_idle  # noqa: E402
from scripts.v2.shared_gpu import process_info, snapshot  # noqa: E402

SERVER = Path("/data/yilangliu/GraD-Pert")
SEED = 20260824
STRATA = ("seen_perturbation", "unseen_perturbation")


def freeze_controls(conditions, contexts, test_rows, control_rows):
    """Freeze only row metadata before expression inference; retain all truth rows."""
    pools = {
        c: control_rows[contexts[control_rows] == c] for c in np.unique(contexts[control_rows])
    }
    draws = {}
    for condition in sorted(set(conditions[test_rows])):
        truth = test_rows[conditions[test_rows] == condition]
        label = str(condition)
        truth_contexts = contexts[truth]
        available = sorted(set(truth_contexts))
        if any(context not in pools for context in available):
            raise ValueError(f"missing K562 control pool for {label}")
        rng = np.random.default_rng(
            stable_draw_seed(dataset_id=DATASET, split_name="test", condition_id=label)
        )
        sampled = rng.choice(truth_contexts, size=300, replace=True)
        sampled_rows = [int(rng.choice(pools[context])) for context in sampled]
        draws[label] = {
            "truth_rows": truth.tolist(),
            "input_control_rows": sampled_rows,
            "metric_control_contexts": available,
        }
    return draws, pools


def source_reference(adata, conditions, rows):
    """Equal condition weighting over original source train+validation rows."""
    ids, codes = np.unique(conditions[rows], return_inverse=True)
    sums = np.zeros((len(ids), adata.n_vars), dtype=np.float64)
    for start in range(0, len(rows), 4096):
        selected = rows[start : start + 4096]
        active, inverse = np.unique(codes[start : start + len(selected)], return_inverse=True)
        indicator = sparse.csr_matrix(
            (np.ones(len(selected), dtype=np.float64), (inverse, np.arange(len(selected)))),
            shape=(len(active), len(selected)),
        )
        sums[active] += (indicator @ adata.X[selected]).toarray()
    return (sums / np.bincount(codes)[:, None]).mean(axis=0)


def read_rows(adata, rows):
    ordered, reverse = np.unique(rows, return_inverse=True)
    return np.asarray(adata.X[ordered].toarray(), dtype=np.float32)[reverse]


def evaluate(args, plan, stop, training_identity, checkpoint, output):
    config = load_experiment_config(plan["config"])
    if config.dataset_id != DATASET:
        raise ValueError("expected the K562 leave-one-line-out configuration")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.9, 0)
    device = torch.device("cuda:0")
    params = config.model.parameters
    with prepare_runtime(
        config,
        data_root=Path(plan["data_root"]),
        run_seed=plan["seed"],
        device=torch.device("cpu"),
        purpose="evaluation",
    ) as runtime:
        if runtime.identity != training_identity["data"] or runtime.optimizer.steps:
            raise ValueError("checkpoint runtime data identity differs")
        history = load_evaluation_checkpoint(
            checkpoint,
            runtime.objective,
            training_identity=training_identity,
            checkpoint_sha256=stop["last"]["sha256"],
        )["history"]
        if [r["epoch"] for r in history] != [1, 2, 3, 4, 5] or history[-1][
            "optimizer_steps"
        ] != 13235:
            raise ValueError("checkpoint is not the sealed epoch-five endpoint")
        model = runtime.objective.student.to(device).eval().requires_grad_(False)
        graph = prepare_evaluation_graph(model, runtime.index, device=device)
        graph_ids = {gene: i for i, gene in enumerate(runtime.index.gene_ids)}
        manifest = read_json(Path(str(params["graph_manifest_path"].value)))
        adata = runtime.data._adata
        rows = frozen_rows(Path(manifest["fold_index"]), manifest["fold"], adata.obs)
        conditions = np.asarray(runtime.data.condition_ids)
        contexts = np.asarray(runtime.data.context_ids)
        draws, pools = freeze_controls(conditions, contexts, rows["test"], rows["target_control"])
        training_conditions = set(conditions[rows["train"]])
        groups = {
            condition: STRATA[int(condition not in training_conditions)] for condition in draws
        }
        counts = {group: list(groups.values()).count(group) for group in STRATA}
        fold = manifest["fold"]
        if counts != dict(
            zip(
                STRATA,
                (fold["test_seen_condition_count"], fold["test_unseen_condition_count"]),
                strict=True,
            )
        ):
            raise ValueError("perturbation exposure differs from frozen fit partition")
        protocol = {
            "dataset": DATASET,
            "split": "test",
            "evaluation_seed": SEED,
            "expression_gene_order_sha256": manifest["expression_gene_order_sha256"],
            "fold_index_sha256": fold["row_index_sha256"],
            "training_conditions_sha256": sha256_json(sorted(training_conditions)),
            "input_controls": "300 K562 controls resampled by condition truth batch proportions",
            "metric_controls": "all K562 controls in the condition's truth batches, cell weighted",
            "systema_reference": "equal-weight source train+validation perturbation centroids",
            "metric_axis": "all 3352 expression genes",
            "condition_counts": counts,
        }
        frozen = output / "FROZEN_ROWS.json"
        atomic_json(frozen, {"protocol": protocol, "draws": draws, "groups": groups})
        atomic_json(
            output / "progress.json", {"stage": "reference", "conditions_total": len(draws)}
        )
        reference = source_reference(adata, conditions, np.sort(np.r_[rows["train"], rows["val"]]))
        reference_sha = sha256_json(reference.tolist())
        pools_means = {}
        for context, ids in pools.items():
            pools_means[context] = (len(ids), np.asarray(adata.X[ids].sum(axis=0)).reshape(-1))
        metrics = {group: [] for group in (*STRATA, "all")}
        detail = []
        with (output / "conditions.jsonl").open("x") as stream, torch.no_grad():
            for i, (condition, draw) in enumerate(draws.items(), 1):
                targets = tuple(graph_ids[g] for g in condition.split("+") if g != "ctrl")
                controls = read_rows(adata, draw["input_control_rows"])
                predicted = predict_controls(
                    model,
                    runtime.index,
                    controls,
                    targets,
                    device=device,
                    cell_batch=int(config.training.eval_batch_size.value),
                    query_count=runtime.options.eval_query_count,
                    graph_memory=graph,
                )
                truth = read_rows(adata, draw["truth_rows"])
                baselines = [pools_means[c] for c in draw["metric_control_contexts"]]
                metric_mean = np.sum([v for _, v in baselines], axis=0, dtype=np.float64)
                metric_mean /= sum(n for n, _ in baselines)
                result = compute_condition_metrics_v2(
                    condition_id=condition,
                    prediction=predicted,
                    input_control=controls,
                    truth=truth,
                    metric_control_pool_mean=metric_mean,
                    de_gene_indices=(),
                    systema_reference=reference,
                    include_deg=False,
                )
                group = groups[condition]
                metrics[group].append(result)
                metrics["all"].append(result)
                record = {**asdict(result), "group": group, "truth_count": len(truth)}
                detail.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if i == 1 or i % 10 == 0 or i == len(draws):
                    atomic_json(
                        output / "progress.json",
                        {
                            "stage": "inference",
                            "conditions_completed": i,
                            "conditions_total": len(draws),
                            "latest_condition": condition,
                            "wall_seconds": time.monotonic() - args.started,
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        },
                    )
                    admit_selected_idle(snapshot(os.getpid()), int(args.gpu), args.uuid)
        if len(detail) != fold["test_condition_count"] or runtime.optimizer.steps:
            raise ValueError("evaluation coverage or optimizer state differs")
        if sha256_file(checkpoint) != stop["last"]["sha256"]:
            raise ValueError("checkpoint content changed")
        return {
            "protocol": protocol,
            "checkpoint_epoch": 5,
            "checkpoint_sha256": stop["last"]["sha256"],
            "production_training_steps": 13235,
            "evaluation_optimizer_updates": 0,
            "frozen_rows_sha256": sha256_file(frozen),
            "systema_reference_sha256": reference_sha,
            "condition_rows_sha256": sha256_file(output / "conditions.jsonl"),
            "scores": {
                group: [asdict(s) for s in macro_summarize(items)]
                for group, items in metrics.items()
            },
            "wall_seconds": time.monotonic() - args.started,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--gpu", choices=("0", "1"), required=True)
    parser.add_argument("--uuid", required=True)
    args = parser.parse_args()
    args.started = time.monotonic()
    if any(
        not path.resolve().is_relative_to(SERVER)
        for path in (args.stop, args.output, args.publication)
    ):
        raise ValueError("scientific evaluation stays on server")
    if (
        os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu
    ):
        raise ValueError("CUDA execution envelope changed")
    with lock(SERVER / "runtime" / f"v2-gpu-{args.gpu}.lock"):
        admit_selected_idle(snapshot(os.getpid()), int(args.gpu), args.uuid)
        stop = read_json(args.stop)
        root = args.stop.resolve().parents[3]
        run = root / "runs" / stop["run_id"]
        plan = read_json(run / "launch.json")
        training = read_json(run / "run_manifest.json")
        checkpoint = Path(stop["checkpoint_retained"]).resolve(strict=True)
        if (
            stop["stage"] != "stopped_after_epoch_5"
            or stop["optimizer_steps"] != 13235
            or stop["last"]["epoch"] != 5
            or stop["plan_sha256"] != sha256_file(run / "launch.json")
            or stop["history_sha256"] != sha256_file(run / "fit/history.json")
            or not checkpoint.is_relative_to(args.stop.resolve().parent)
            or not checkpoint.samefile(run / "fit/epoch-0005.pt")
            or training["run_id"] != stop["run_id"]
            or training["config_sha256"] != sha256_file(Path(plan["config"]))
        ):
            raise ValueError("sealed epoch-five training endpoint differs")
        source = inspect_source_identity(
            SOURCE,
            formal=True,
            expected_repository=plan["resolved_config"]["source_code"]["repository"],
            publication_receipt=args.publication,
            expected_publication_receipt_sha256=args.publication_sha,
        )
        args.output.mkdir(exist_ok=False)
        owner = {
            "run_id": stop["run_id"],
            "plan_sha256": stop["plan_sha256"],
            "checkpoint_sha256": stop["last"]["sha256"],
            "training_source": plan["source_commit"],
            "evaluation_source": source.commit,
            "owner": process_info(os.getpid()),
            "gpu": args.gpu,
            "gpu_uuid": args.uuid,
        }
        atomic_json(args.output / "OWNER.json", owner)
        try:
            result = evaluate(args, plan, stop, training, checkpoint, args.output)
            atomic_json(args.output / "COMPLETE.json", {**owner, **result})
        except BaseException as error:
            atomic_json(
                args.output / "FAILURE.json",
                {
                    **owner,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise


if __name__ == "__main__":
    main()
