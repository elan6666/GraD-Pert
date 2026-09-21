"""Complete-path CUDA capacity probe; outputs are engineering evidence only."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", choices=("0", "1"), required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha256", required=True)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    if args.steps < 128:
        parser.error("capacity evidence requires at least 128 sustained updates")
    if not args.output.resolve().is_relative_to("/data/yilangliu"):
        parser.error("capacity artifacts stay on the server")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    from gradpert.config import load_experiment_config
    from gradpert.config.step_schedule import load_training_schedule
    from gradpert.data._io import atomic_json
    from gradpert.evaluation.data import CanonicalEvaluationData
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.hashing import sha256_file
    from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
    from gradpert.training.v2.engine import optimizer_step
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
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "kind": "capacity_only",
        "source": source.payload(),
        "environment": environment.payload(),
        "config_sha256": sha256_file(args.config),
        "gpu": args.gpu,
        "steps_requested": args.steps,
        "steps_completed": 0,
        "status": "running",
    }
    atomic_json(args.output / "receipt.json", receipt)
    try:
        free, total = torch.cuda.mem_get_info()
        if total - free > 512 * 1024**2:
            raise RuntimeError("capacity probe requires idle GPU; existing work is preserved")
        schedule = load_training_schedule(config.training.scheduler.value)
        if schedule is None:
            raise ValueError("probe requires the sealed training schedule")
        with prepare_runtime(
            config,
            data_root=args.data_root,
            run_seed=config.training.run_seeds[0],
            device=torch.device("cuda:0"),
        ) as runtime:
            receipt["data"] = runtime.identity
            receipt["optimizer_routes"] = runtime.optimizer.routes
            durations = []
            cells = []
            torch.cuda.reset_peak_memory_stats()
            step = 0
            training_started = time.perf_counter()
            for epoch in itertools.count():
                for batch in runtime.batches(epoch):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    terms = optimizer_step(
                        runtime.objective,
                        runtime.optimizer,
                        batch,
                        microbatch=runtime.options.microbatch,
                        lr=schedule.at_step(step, 50 * runtime.steps_per_epoch)["learning_rate"],
                        momentum=runtime.options.teacher_end
                        - (runtime.options.teacher_end - runtime.options.teacher_start)
                        * (1 + math.cos(math.pi * step / (50 * runtime.steps_per_epoch - 1)))
                        / 2,
                        bf16=True,
                    )
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter() - started)
                    cells.append(len(batch.control))
                    step += 1
                    receipt["steps_completed"] = step
                    receipt["last_terms"] = terms
                    if step == args.steps // 2:
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
                        atomic_json(args.output / "receipt.json", receipt)
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
                predicted = predict_controls(
                    runtime.objective.student,
                    runtime.index,
                    controls.expression,
                    targets,
                    device=runtime.device,
                    cell_batch=int(config.training.eval_batch_size.value),
                    query_count=runtime.options.eval_query_count,
                )
                from gradpert.training.validation import mean_expression_mse

                truth = evaluation.load_truth_rows(draw.condition_id)
                receipt["validation_prediction_loss"] = mean_expression_mse(
                    predicted, truth.expression
                )
                receipt["inference_shape"] = list(predicted.shape)
                receipt["inference_condition"] = draw.condition_id
                receipt["inference_control_manifest_sha256"] = (
                    evaluation.control_manifest_file_sha256
                )
            receipt.update(
                status="passed",
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                measured_update_seconds=durations[8:],
                cells_per_second=sum(cells[8:]) / sum(durations[8:]),
                coverage="128+ updates; checkpoint continuation; single-condition validation",
            )
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(args.output / "receipt.json", receipt)
        raise
    atomic_json(args.output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
