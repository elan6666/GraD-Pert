"""Complete-path CUDA capacity probe; outputs are engineering evidence only."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path


def probe_policy(
    integration_only: bool, steps: int | None, benchmark_only: bool = False
) -> tuple[int, int, str]:
    if integration_only and benchmark_only:
        raise ValueError("integration-only and benchmark-only are mutually exclusive")
    if integration_only:
        if steps not in (None, 1):
            raise ValueError("integration-only mode requires exactly one optimizer update")
        return 1, 0, "integration_only"
    if benchmark_only:
        steps = 5 if steps is None else steps
        if steps < 3:
            raise ValueError("benchmark-only mode requires at least three updates")
        return steps, 1, "benchmark_only"
    steps = 128 if steps is None else steps
    if steps < 128:
        raise ValueError("capacity evidence requires at least 128 sustained updates")
    return steps, 8, "capacity_only"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="physical GPU list; use torchrun for two GPUs")
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha256", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--integration-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    try:
        args.steps, warmup_steps, kind = probe_policy(
            args.integration_only, args.steps, args.benchmark_only
        )
    except ValueError as error:
        parser.error(str(error))
    if not args.output.resolve().is_relative_to("/data/yilangliu"):
        parser.error("capacity artifacts stay on the server")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    devices = args.gpu.split(",")
    if (
        len(devices) != world
        or len(set(devices)) != world
        or not set(devices) <= {"0", "1"}
        or not 0 <= rank < world
    ):
        parser.error("GPU list must match torchrun world size, using distinct physical GPUs")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    from gradpert.config import load_experiment_config
    from gradpert.config.step_schedule import load_training_schedule
    from gradpert.data._io import atomic_json
    from gradpert.evaluation.data import CanonicalEvaluationData
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.hashing import sha256_file
    from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
    from gradpert.training.v2.distributed import primary_call
    from gradpert.training.v2.engine import optimizer_step, slice_cells
    from gradpert.training.v2.evaluation import predict_controls
    from gradpert.training.v2.runtime import prepare_runtime

    config = load_experiment_config(args.config)
    source = inspect_source_identity(
        Path(__file__).resolve().parents[2],
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    environment = inspect_environment(Path(__file__).resolve().parents[2], device_name="cuda:0")
    free, total = torch.cuda.mem_get_info()
    if total - free > 512 * 1024**2:
        raise RuntimeError("capacity probe requires idle GPU; existing work is preserved")
    if world > 1:
        torch.cuda.set_device(0)
        torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
    primary_call(lambda: args.output.mkdir(parents=True, exist_ok=False))
    receipt = {
        "kind": kind,
        "data_root": str(args.data_root.resolve()),
        "inference_exercised": kind == "capacity_only",
        "source": source.payload(),
        "environment": environment.payload(),
        "config_sha256": sha256_file(args.config),
        "gpu": args.gpu,
        "world_size": world,
        "steps_requested": args.steps,
        "steps_completed": 0,
        "status": "running",
    }
    primary_call(lambda: atomic_json(args.output / "receipt.json", receipt))
    try:
        schedule = load_training_schedule(config.training.scheduler.value)
        if schedule is None:
            raise ValueError("probe requires the sealed training schedule")
        with prepare_runtime(
            config,
            data_root=args.data_root,
            run_seed=config.training.run_seeds[0],
            device=torch.device("cuda:0"),
        ) as runtime:
            total_steps = int(config.training.max_epochs.value) * runtime.steps_per_epoch
            receipt["data"] = runtime.identity
            receipt["optimizer_routes"] = runtime.optimizer.routes
            durations = []
            cells = []
            gradient_reduction_seconds = []
            torch.cuda.reset_peak_memory_stats()
            step = 0
            training_started = time.perf_counter()
            for epoch in itertools.count():
                for batch in runtime.batches(epoch):
                    global_cells = len(batch.control)
                    global_conditions = batch.condition_index if world > 1 else None
                    if world > 1:
                        batch = slice_cells(
                            batch, global_cells * rank // world, global_cells * (rank + 1) // world
                        )
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    timings = {}
                    terms = optimizer_step(
                        runtime.objective,
                        runtime.optimizer,
                        batch,
                        microbatch=runtime.options.microbatch,
                        lr=schedule.at_step(step, total_steps)["learning_rate"],
                        momentum=runtime.options.teacher_end
                        - (runtime.options.teacher_end - runtime.options.teacher_start)
                        * (1 + math.cos(math.pi * step / (total_steps - 1)))
                        / 2,
                        bf16=True,
                        global_condition_index=global_conditions,
                        timings=timings,
                    )
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter() - started)
                    cells.append(global_cells)
                    gradient_reduction_seconds.append(
                        timings.get("gradient_reduction_seconds", 0.0)
                    )
                    step += 1
                    receipt["steps_completed"] = step
                    receipt["last_terms"] = terms
                    if step == max(1, args.steps // 2):
                        path = args.output / "resume.pt"
                        save_checkpoint(
                            path,
                            runtime.objective,
                            runtime.optimizer,
                            identity=runtime.identity,
                            progress={"step": step},
                            generator=runtime.generator,
                        )
                        progress = load_checkpoint(
                            path,
                            runtime.objective,
                            runtime.optimizer,
                            identity=runtime.identity,
                            generator=runtime.generator,
                        )
                        if progress["step"] != step or runtime.optimizer.steps != step:
                            raise ValueError("capacity checkpoint resume progress mismatch")
                        receipt["resume_checkpoint_sha256"] = sha256_file(path)
                    if step % 8 == 0:
                        primary_call(lambda: atomic_json(args.output / "receipt.json", receipt))
                    if step >= args.steps:
                        break
                if step >= args.steps:
                    break
            receipt["training_wall_seconds"] = time.perf_counter() - training_started
            receipt["end_to_end_training_cells_per_second"] = (
                sum(cells) / receipt["training_wall_seconds"]
            )

            # Exercise the identical 300-control inference path on one validation
            # condition. No test truth, scientific score or hyperparameter selection.
            def validation_probe():
                validation = {}
                with CanonicalEvaluationData(
                    dataset_id=config.dataset_id,
                    protocol_id=config.data.protocol_id,
                    data_root=args.data_root,
                    split_name="val",
                ) as evaluation:
                    draw = evaluation.control_manifest.draws[0]
                    controls = evaluation.load_control_rows(tuple(draw.ordered_row_ids))
                    genes = {g: i for i, g in enumerate(runtime.index.gene_ids)}
                    targets = tuple(
                        genes[g]
                        for g in draw.condition_id.split("+")
                        if g != evaluation.split.control_condition_id
                    )
                    torch.cuda.synchronize()
                    inference_started = time.perf_counter()
                    predicted = predict_controls(
                        runtime.objective.student,
                        runtime.index,
                        controls.expression,
                        targets,
                        device=runtime.device,
                        cell_batch=int(config.training.eval_batch_size.value),
                        query_count=runtime.options.eval_query_count,
                    )
                    torch.cuda.synchronize()
                    validation["inference_seconds"] = time.perf_counter() - inference_started
                    validation["evaluation_cell_batch"] = int(config.training.eval_batch_size.value)
                    validation["validation_condition_count"] = len(
                        evaluation.control_manifest.draws
                    )
                    from gradpert.training.validation import mean_expression_mse

                    truth = evaluation.load_truth_rows(draw.condition_id)
                    validation["validation_prediction_loss"] = mean_expression_mse(
                        predicted, truth.expression
                    )
                    validation["inference_shape"] = list(predicted.shape)
                    validation["inference_condition"] = draw.condition_id
                    validation["inference_control_manifest_sha256"] = (
                        evaluation.control_manifest_file_sha256
                    )
                return validation

            if kind == "capacity_only":
                receipt.update(primary_call(validation_probe))
            local_measurement = {
                "rank": rank,
                "physical_gpu": devices[rank],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "training_wall_seconds": receipt["training_wall_seconds"],
                "update_seconds": durations[warmup_steps:],
                "gradient_reduction_seconds": gradient_reduction_seconds[warmup_steps:],
            }
            measurements = [local_measurement]
            if world > 1:
                measurements = [None] * world
                torch.distributed.all_gather_object(measurements, local_measurement)
            durations_measured = [
                max(times)
                for times in zip(*(m["update_seconds"] for m in measurements), strict=True)
            ]
            receipt["rank_measurements"] = measurements
            receipt["communication_timing_scope"] = (
                "synchronized gradient averaging only; excludes center/metric collectives; "
                "includes reduction packing and rank wait"
            )
            receipt["end_to_end_training_cells_per_second"] = sum(cells) / max(
                m["training_wall_seconds"] for m in measurements
            )
            receipt.update(
                status="passed",
                peak_allocated_bytes=max(m["peak_allocated_bytes"] for m in measurements),
                peak_reserved_bytes=max(m["peak_reserved_bytes"] for m in measurements),
                measured_update_seconds=durations_measured,
                cells_per_second=sum(cells[warmup_steps:]) / sum(durations_measured),
                coverage=(
                    "one optimizer update; checkpoint reload; "
                    "no sustained-capacity or inference evidence"
                    if kind == "integration_only"
                    else (
                        "bounded training throughput only; checkpoint reload; "
                        "no sustained-capacity or inference evidence"
                        if kind == "benchmark_only"
                        else "128+ updates; checkpoint continuation; single-condition validation"
                    )
                ),
            )
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(args.output / f"rank-{rank}-failure.json", receipt)
        if rank == 0:
            atomic_json(args.output / "receipt.json", receipt)
        raise
    primary_call(lambda: atomic_json(args.output / "receipt.json", receipt))
    if rank == 0:
        print(json.dumps(receipt, indent=2))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
