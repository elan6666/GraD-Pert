"""Independent, receipt-backed v2 checkpoint evaluation on one or more GPUs."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json

SERVER_ROOT = Path("/data/yilangliu")


def resolve_evaluation_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Reject mismatched checkpoint/config/source before allocating a GPU."""
    from gradpert.execution.train_entry import repository_root

    repository = repository_root()
    config_path = args.config.resolve(strict=True)
    config = load_experiment_config(config_path)
    if config.model_id != "gradpert_v2":
        raise ValueError("independent checkpoint evaluation requires v2")
    training_root = args.training_run_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output_root.resolve()
    if not all(path.is_relative_to(SERVER_ROOT) for path in (training_root, checkpoint, output)):
        raise ValueError("checkpoints and evaluation artifacts stay on the server")
    if output.exists() or not checkpoint.is_relative_to(training_root / "fit"):
        raise ValueError("evaluation requires a new output root and checkpoint inside its run")
    training_identity = read_json(training_root / "run_manifest.json")
    journal = read_json(training_root / "fit/epoch_state.json")
    if journal["identity"] != training_identity:
        raise ValueError("training journal differs from its manifest")
    if sha256_file(checkpoint) != args.checkpoint_sha256:
        raise ValueError("checkpoint checksum mismatch")
    matches = [
        (role, selected)
        for role in ("best", "last")
        if (selected := journal[role]) is not None
        and selected["sha256"] == args.checkpoint_sha256
        and checkpoint == (training_root / "fit" / selected["file"]).resolve(strict=True)
    ]
    if not matches:
        raise ValueError("checkpoint is not the verified best or last of this run")
    role, selected = matches[0]
    checkpoint_identity = selected.get("training_identity", training_identity)
    if sha256_json(config.model_dump(mode="json")) != checkpoint_identity["resolved_config_sha256"]:
        raise ValueError("checkpoint evaluation configuration differs from its training source")
    runtime_path = args.runtime.resolve(strict=True)
    runtime = read_json(runtime_path)
    publication = Path(runtime["publication_receipt"]).resolve(strict=True)
    if sha256_file(publication) != runtime["publication_sha256"]:
        raise ValueError("evaluation source publication receipt checksum mismatch")
    source = inspect_source_identity(
        repository,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=publication,
        expected_publication_receipt_sha256=runtime["publication_sha256"],
    )
    gpu_ids = args.gpu.split(",")
    if (
        not gpu_ids
        or len(set(gpu_ids)) != len(gpu_ids)
        or any(not re.fullmatch(r"[0-9]+|GPU-[a-fA-F0-9-]+", gpu) for gpu in gpu_ids)
    ):
        raise ValueError("evaluation GPUs must be distinct physical selectors")
    data_root = Path(runtime["data_root"]).resolve(strict=True)
    if not data_root.is_relative_to(SERVER_ROOT):
        raise ValueError("evaluation data must stay on the server")
    return {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "training_run_root": str(training_root),
        "training_identity": checkpoint_identity,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_role": role,
        "checkpoint_epoch": selected["epoch"],
        "split": args.split,
        "gpu": gpu_ids,
        "data_root": str(data_root),
        "output_root": str(output),
        "evaluation_source": source.payload(),
        "runtime_sha256": sha256_file(runtime_path),
    }


def evaluate_worker(plan: dict[str, Any], index: int) -> dict[str, Any]:
    """Each process owns one visible GPU and a disjoint condition subset."""
    import torch

    from gradpert.evaluation.data import CanonicalEvaluationData
    from gradpert.evaluation.state import load_evaluation_state
    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.evaluation import evaluate
    from gradpert.training.v2.exposure import checkpoint_expression_groups
    from gradpert.training.v2.runtime import prepare_runtime

    config = load_experiment_config(plan["config"])
    if sha256_file(Path(plan["config"])) != plan["config_sha256"]:
        raise ValueError("evaluation configuration changed after planning")
    device = torch.device("cuda:0")
    with prepare_runtime(
        config,
        data_root=Path(plan["data_root"]),
        run_seed=int(plan["training_identity"]["data"]["run_seed"]),
        device=device,
        purpose="evaluation",
    ) as runtime:
        if runtime.identity != plan["training_identity"]["data"]:
            raise ValueError("evaluation data identity differs from checkpoint training data")
        progress = load_evaluation_checkpoint(
            Path(plan["checkpoint"]),
            runtime.objective,
            training_identity=plan["training_identity"],
            checkpoint_sha256=plan["checkpoint_sha256"],
        )
        split: Literal["val", "test"] = plan["split"]
        common = {
            "dataset_id": config.dataset_id,
            "protocol_id": config.data.protocol_id,
            "data_root": Path(plan["data_root"]),
        }
        reference = load_evaluation_state(
            **common, validation_only=split == "val", evaluation_protocol="v2"
        )
        with CanonicalEvaluationData(**common, split_name=split) as data:
            frozen_order = tuple(draw.condition_id for draw in data.control_manifest.draws)
            selected = frozen_order[index :: len(plan["gpu"])]
            if not selected:
                raise ValueError("more evaluation workers than frozen conditions")
            groups, exposure = checkpoint_expression_groups(
                progress["history"],
                checkpoint_epoch=int(plan["checkpoint_epoch"]),
                expression_gene_ids=data.expression_gene_ids,
                allowed_gene_indices=(
                    tuple(int(i) for i in runtime.allowed_expression_ids)
                    if runtime.allowed_expression_ids is not None
                    else None
                ),
            )
            result = evaluate(
                runtime.objective.student,
                runtime.index,
                data,
                reference,
                expected_split=split,
                device=device,
                cell_batch=int(config.training.eval_batch_size.value),
                query_count=runtime.options.eval_query_count,
                metric_gene_groups=groups,
                condition_ids=selected,
            )
            result["expression_exposure"] = exposure
            return {"frozen_condition_order": list(frozen_order), "result": result}


def execute_evaluation_plan(plan: dict[str, Any]) -> dict[str, Any]:
    from gradpert.training.v2.evaluation import merge_evaluation_shards

    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    aliases = {
        value.strip(): line.split(",")[1].strip()
        for line in devices.splitlines()
        for value in line.split(",")
    }
    if any(gpu not in aliases for gpu in plan["gpu"]) or len(
        {aliases[gpu] for gpu in plan["gpu"]}
    ) != len(plan["gpu"]):
        raise ValueError("evaluation requires distinct available physical GPUs")
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "plan.json", plan)
    processes = []
    try:
        for index, gpu in enumerate(plan["gpu"]):
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            repository = Path(plan["evaluation_source"]["repository_root"])
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(repository / "src"), str(repository), environment.get("PYTHONPATH", "")]
            )
            with (root / f"worker-{index}.log").open("w") as log:
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "gradpert.execution.v2_checkpoint_eval",
                            "--worker-plan",
                            str(root / "plan.json"),
                            "--worker-index",
                            str(index),
                        ],
                        env=environment,
                        cwd=plan["evaluation_source"]["repository_root"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
        return_codes = [process.wait() for process in processes]
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"checkpoint evaluation workers failed: {return_codes}")
        shards = [read_json(root / f"worker-{index}.json") for index in range(len(processes))]
        orders = [tuple(shard["frozen_condition_order"]) for shard in shards]
        if any(order != orders[0] for order in orders[1:]):
            raise ValueError("evaluation workers disagree on frozen condition order")
        result = merge_evaluation_shards(
            [shard["result"] for shard in shards], ordered_conditions=orders[0]
        )
        result["expression_exposure"] = shards[0]["result"]["expression_exposure"]
        receipt = {"plan": plan, "result": result, "zero_pkl": not any(root.rglob("*.pkl"))}
        if not receipt["zero_pkl"]:
            raise ValueError("independent evaluation unexpectedly produced a PKL")
        atomic_json(root / "COMPLETE.json", receipt)
        return receipt
    except BaseException as error:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        atomic_json(
            root / "FAILURE.json", {"error_type": type(error).__name__, "error": str(error)}
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-plan", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    args = parser.parse_args()
    plan = read_json(args.worker_plan)
    if not 0 <= args.worker_index < len(plan["gpu"]):
        raise ValueError("worker index differs from sealed evaluation GPU list")
    result = evaluate_worker(plan, args.worker_index)
    atomic_json(Path(plan["output_root"]) / f"worker-{args.worker_index}.json", result)


if __name__ == "__main__":
    main()
