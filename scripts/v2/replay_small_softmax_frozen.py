"""Read-only Jurkat checkpoint replay and server-only result acceptance."""

from __future__ import annotations

import argparse
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

from gradpert.data._io import atomic_json, read_json
from gradpert.execution.v2_checkpoint_eval import execute_evaluation_plan, resolve_evaluation_plan
from gradpert.hashing import sha256_file
from scripts.v2.run_dataset_transfer_300 import native_run
from scripts.v2.run_group import lock
from scripts.v2.run_small_softmax import admit_selected_idle
from scripts.v2.shared_gpu import process_info, snapshot
from scripts.v2.small_softmax_workflow_server import condition_aggregates

ROOT = Path("/data/yilangliu/GraD-Pert")
ARMS = ("baseline", "deep-dual")
SCHEMA = "jurkat-frozen-replay-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_replay(arm, original, replay, output):
    plan, previous_plan = replay["plan"], original["plan"]
    for key in set(previous_plan) - {"output_root", "evaluation_source", "runtime_sha256", "gpu"}:
        require(plan[key] == previous_plan[key], f"evaluation recipe changed: {key}")
    require(
        plan["response_diagnostics"]
        and plan["checkpoint_role"] == "last"
        and plan["checkpoint_epoch"] == 2
        and replay["zero_pkl"]
        and not any(output.rglob("*.pkl")),
        "replay escaped the sealed read-only validation protocol",
    )
    old, new = original["result"], replay["result"]
    for key in (
        "split",
        "metric_scope",
        "control_manifest_sha256",
        "reference_sha256",
        "query_recipe",
        "selection_gene_ids",
        "expression_exposure",
    ):
        require(old[key] == new[key], f"frozen evaluator identity differs: {key}")
    require(len(old["conditions"]) == len(new["conditions"]) == 445, "condition count changed")
    maximum = {"condition_pearson": 0.0, "condition_mse": 0.0, "macro_pearson": 0.0}
    for a, b in zip(old["conditions"], new["conditions"], strict=True):
        for field in ("condition_id", "control_row_ids", "truth_row_ids"):
            require(a[field] == b[field], f"ordered evaluation rows changed: {field}")
        maximum["condition_mse"] = max(maximum["condition_mse"], abs(a["loss"] - b["loss"]))
        pairs = [(a["metrics"], b["metrics"])] + [
            (a["metric_gene_groups"][key], b["metric_gene_groups"][key])
            for key in ("seen_expression", "unseen_expression")
        ]
        for left, right in pairs:
            if left is None or right is None:
                require(left is right, "condition metric scope changed")
                continue
            for x, y in zip(left["results"], right["results"], strict=True):
                for field in ("metric_id", "reason", "gene_count"):
                    require(x[field] == y[field], f"condition metric coverage changed: {field}")
                if x["value"] is not None:
                    maximum["condition_pearson"] = max(
                        maximum["condition_pearson"], abs(x["value"] - y["value"])
                    )
                else:
                    require(y["value"] is None, "null Pearson changed")
        effects = b["response_diagnostics"]
        require(
            math.isclose(
                effects["amplitude_mse"] + effects["direction_mse"],
                effects["mse"]["model"],
                rel_tol=1e-8,
                abs_tol=1e-12,
            ),
            "effect error decomposition does not close",
        )
    aggregate = []
    for scope in ("all", "seen", "unseen"):
        old_scope = old if scope == "all" else old["metric_gene_groups"][scope + "_expression"]
        new_scope = new if scope == "all" else new["metric_gene_groups"][scope + "_expression"]
        if scope != "all":
            for key in ("gene_count", "gene_ids_sha256", "unavailable_reason"):
                require(
                    old_scope[key] == new_scope[key], f"expression group identity changed: {key}"
                )
        for x, y in zip(old_scope["metrics"], new_scope["metrics"], strict=True):
            for key in (
                "metric_id",
                "finite_condition_count",
                "total_condition_count",
                "unavailable_reasons",
            ):
                require(x[key] == y[key], f"macro scope changed: {key}")
            if x["macro_mean"] is not None:
                maximum["macro_pearson"] = max(
                    maximum["macro_pearson"], abs(x["macro_mean"] - y["macro_mean"])
                )
            else:
                require(y["macro_mean"] is None, "null macro changed")
            aggregate.append(
                {
                    "scope": scope + "_expression",
                    "metric": y["metric_id"],
                    "candidate": {
                        "macro_mean": y["macro_mean"],
                        "valid": y["finite_condition_count"],
                    },
                }
            )
    macro_mse_delta = abs(old["prediction_loss"] - new["prediction_loss"])
    require(len(aggregate) == 9, "missing macro Pearson results")
    require(
        maximum["macro_pearson"] <= 2e-6
        and macro_mse_delta <= 2e-6
        and maximum["condition_pearson"] <= 2e-5
        and maximum["condition_mse"] <= 2e-5,
        f"{arm} replay parity differs: {maximum}, macro MSE {macro_mse_delta}",
    )
    return aggregate, {**maximum, "macro_mse": macro_mse_delta}


def queue(queue_path, checksum):
    require(sha256_file(queue_path) == checksum, "sealed queue hash changed")
    spec = read_json(queue_path)
    root = queue_path.resolve(strict=True).parent
    require(
        root.is_relative_to(ROOT / "development")
        and spec["schema"] == SCHEMA
        and tuple(arm["id"] for arm in spec["arms"]) == ARMS,
        "wrong evaluation queue",
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == str(spec["gpu_index"])
        and os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True"
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "sealed allocator or GPU environment missing",
    )
    source = Path(__file__).resolve().parents[2]
    owner = process_info(os.getpid())
    require(owner is not None, "queue owner identity missing")

    def state(status, **fields):
        atomic_json(
            root / "STATE.json",
            {
                "status": status,
                "queue_sha256": checksum,
                "owner": owner,
                **fields,
            },
        )

    def guard():
        admit_selected_idle(snapshot(os.getpid()), spec["gpu_index"], spec["gpu_uuid"])

    active = "preflight"
    try:
        with lock(ROOT / f"runtime/v2-gpu-{spec['gpu_index']}.lock"):
            guard()
            for item in spec["arms"]:
                active = item["id"]
                state("running", arm=active)
                arm = root / active
                arm.mkdir(exist_ok=False)
                stage = Path(item["stage"]).resolve(strict=True)
                inputs = {
                    "PLAN.json": item["training_plan_sha256"],
                    "MAIN_EXPRESSION_VAL_ACCEPTANCE.json": item["acceptance_sha256"],
                    "expression-val/COMPLETE.json": item["original_evaluation_sha256"],
                }
                for name, expected in inputs.items():
                    require(
                        sha256_file(stage / name) == expected, f"historical input changed: {name}"
                    )
                accepted = read_json(stage / "MAIN_EXPRESSION_VAL_ACCEPTANCE.json")
                original = read_json(stage / "expression-val/COMPLETE.json")
                previous = original["plan"]
                train_file = Path(previous["training_run_root"]) / "TRAIN_COMPLETE.json"
                require(
                    sha256_file(train_file) == accepted["training_sha256"],
                    "training receipt differs",
                )
                train = read_json(train_file)
                require(
                    accepted["run_id"]
                    == item["run_id"]
                    == train["identity"]["run_id"]
                    == previous["training_identity"]["run_id"]
                    and accepted["checkpoint_sha256"]
                    == item["checkpoint_sha256"]
                    == train["last"]["sha256"]
                    == previous["checkpoint_sha256"]
                    and accepted["evaluation_sha256"] == inputs["expression-val/COMPLETE.json"]
                    and train["epoch"] == 2
                    and accepted["validation_conditions"] == 445
                    and previous["cuda_memory_fraction"] == 0.9
                    and previous["cpu_training_state"]
                    and accepted["status"]
                    == "main_independently_accepted_frozen_expression_validation",
                    "checkpoint differs from historical acceptance",
                )
                runtime = arm / "RUNTIME.json"
                atomic_json(
                    runtime,
                    {
                        "data_root": previous["data_root"],
                        "publication_receipt": spec["publication_receipt"],
                        "publication_sha256": spec["publication_sha256"],
                        "publication_remote_ref": spec["remote_ref"],
                    },
                )
                plan = resolve_evaluation_plan(
                    Namespace(
                        config=Path(previous["config"]),
                        training_run_root=Path(previous["training_run_root"]),
                        checkpoint=Path(previous["checkpoint"]),
                        checkpoint_sha256=item["checkpoint_sha256"],
                        checkpoint_role="last",
                        runtime=runtime,
                        output_root=arm / "frozen",
                        gpu=str(spec["gpu_index"]),
                        split="val",
                        metric_scope="expression_only",
                        response_diagnostics=True,
                    )
                )
                plan.update(cpu_training_state=True, cuda_memory_fraction=0.9)
                require(
                    plan["evaluation_source"]["commit"] == spec["evaluation_commit"]
                    and plan["evaluation_source"]["repository_root"] == str(source),
                    "evaluation checkout does not match queue",
                )
                plan_file = arm / "FROZEN_PLAN.json"
                atomic_json(plan_file, plan)
                native_run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--execute",
                        str(plan_file),
                        "--sha256",
                        sha256_file(plan_file),
                    ],
                    arm / "EVAL.log",
                    {},
                    admission_check=guard,
                )
                complete = arm / "frozen/COMPLETE.json"
                receipt = read_json(complete)
                require(receipt["plan"] == plan, "replay did not execute the sealed plan")
                macros, parity = validate_replay(active, original, receipt, arm)
                require(
                    sha256_file(Path(plan["checkpoint"])) == item["checkpoint_sha256"],
                    "checkpoint changed during replay",
                )
                replay_hash = sha256_file(complete)
                aggregate = arm / "condition-aggregates.json"
                atomic_json(
                    aggregate,
                    condition_aggregates(
                        receipt,
                        {"run_id": item["run_id"], "response_diagnostics": True},
                        replay_hash,
                    ),
                )
                atomic_json(
                    arm / "ACCEPTED.json",
                    {
                        "schema": SCHEMA,
                        "run_id": item["run_id"],
                        "arm": active,
                        "status": "accepted",
                        "queue_sha256": checksum,
                        "optimizer_updates": 0,
                        "checkpoint_sha256": item["checkpoint_sha256"],
                        "validation_conditions": 445,
                        "training_commit": train["identity"]["source"]["commit"],
                        "evaluation_commit": spec["evaluation_commit"],
                        "evaluation_sha256": replay_hash,
                        "frozen_plan_sha256": sha256_file(plan_file),
                        "original_evaluation_sha256": item["original_evaluation_sha256"],
                        "aggregate_sha256": sha256_file(aggregate),
                        "comparison": macros,
                        "candidate_mse": receipt["result"]["prediction_loss"],
                        "parity_max_abs": parity,
                        "reference_sha256": receipt["result"]["reference_sha256"],
                        "control_manifest_sha256": receipt["result"]["control_manifest_sha256"],
                    },
                )
            state(
                "completed",
                accepted={arm: sha256_file(root / arm / "ACCEPTED.json") for arm in ARMS},
            )
    except Exception as error:
        state("failed", arm=active, error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--queue", type=Path)
    group.add_argument("--execute", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.execute:
        require(sha256_file(args.execute) == args.sha256, "sealed evaluation plan hash changed")
        execute_evaluation_plan(read_json(args.execute))
    else:
        queue(args.queue, args.sha256)
