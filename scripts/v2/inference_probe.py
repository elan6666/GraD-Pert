"""Measure inference batches on one frozen validation population, without truth."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from evaluate_context import engineering_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "training-run", "data-root", "publication", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--engineering-receipt-sha256", required=True)
    parser.add_argument("--publication-sha256", required=True)
    parser.add_argument("--gpu", choices=("0", "1"), required=True)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument(
        "--partition-full-axis",
        action="store_true",
        help="Cover the full expression axis in query-count blocks, as in default evaluation",
    )
    parser.add_argument("--cell-batches", type=int, nargs="+", default=[2, 8, 16, 32])
    args = parser.parse_args()
    if (
        args.query_count < 1
        or args.cell_batches != sorted(set(args.cell_batches))
        or args.cell_batches[0] < 1
    ):
        parser.error("positive query count and unique increasing cell batches required")
    if args.output.exists():
        parser.error("probe output must be new")
    if not all(
        p.resolve().is_relative_to("/data/yilangliu")
        for p in (args.data_root, args.training_run, args.output)
    ):
        parser.error("scientific data and probe output remain on the server")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import numpy as np
    import torch

    from gradpert.config import load_experiment_config
    from gradpert.data._io import atomic_json
    from gradpert.evaluation.data import CanonicalEvaluationData
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.hashing import sha256_file, sha256_json
    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.evaluation import predict_controls, predict_query_set
    from gradpert.training.v2.runtime import prepare_runtime

    config = load_experiment_config(args.config)
    training, checkpoint, checkpoint_path = engineering_checkpoint(
        args.training_run, args.engineering_receipt_sha256
    )
    if sha256_file(args.config) != training["config_sha256"]:
        raise ValueError("probe configuration differs from checkpoint training")
    root = Path(__file__).resolve().parents[2]
    source = inspect_source_identity(
        root,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    free, total = torch.cuda.mem_get_info()
    if total - free > 512 * 1024**2:
        raise RuntimeError("inference capacity measurement requires an idle GPU")
    device = torch.device("cuda:0")
    receipt = {
        "kind": "engineering_inference_batch_probe",
        "scientific_result": False,
        "physical_gpu": args.gpu,
        "status": "running",
        "source": source.payload(),
        "environment": inspect_environment(root, device_name="cuda:0").payload(),
        "training_source": training["source"],
        "checkpoint": checkpoint,
        "training_receipt_sha256": args.engineering_receipt_sha256,
        "config_sha256": training["config_sha256"],
        "timing_scope": (
            "one end-to-end call per batch, including graph encoding and transfers; "
            "first point may include cold overhead"
        ),
        "query_count": args.query_count,
        "query_recipe": (
            "ordered_blocks_covering_full_expression_axis"
            if args.partition_full_axis
            else "single_fixed_context"
        ),
        "cell_batches": args.cell_batches,
        "equivalence_tolerance": {"atol": 2e-5, "rtol": 2e-5},
        "points": [],
    }
    atomic_json(args.output, receipt)
    try:
        with prepare_runtime(
            config,
            data_root=args.data_root,
            run_seed=training["data"]["run_seed"],
            device=device,
            purpose="evaluation",
        ) as runtime:
            if runtime.identity != training["data"]:
                raise ValueError("inference data/graph/GenePT identity differs from checkpoint")
            receipt["checkpoint_progress"] = load_evaluation_checkpoint(
                checkpoint_path,
                runtime.objective,
                training_identity=training["data"],
                checkpoint_sha256=checkpoint["sha256"],
            )
            with CanonicalEvaluationData(
                dataset_id=config.dataset_id,
                protocol_id=config.data.protocol_id,
                data_root=args.data_root,
                split_name="val",
            ) as data:
                if args.query_count > len(data.expression_gene_ids):
                    raise ValueError("query count exceeds the expression axis")
                draw = data.control_manifest.draws[0]
                controls = data.load_control_rows(tuple(draw.ordered_row_ids))
                if len(controls.ordered_row_ids) != 300 or tuple(controls.ordered_row_ids) != tuple(
                    draw.ordered_row_ids
                ):
                    raise ValueError("probe requires the exact frozen 300-control population")
                positions = {g: i for i, g in enumerate(runtime.index.gene_ids)}
                targets = tuple(
                    positions[g]
                    for g in draw.condition_id.split("+")
                    if g != data.split.control_condition_id
                )
                queries = np.arange(args.query_count)
                receipt.update(
                    condition_id=draw.condition_id,
                    ordered_control_row_ids=list(controls.ordered_row_ids),
                    control_manifest_sha256=data.control_manifest_file_sha256,
                    query_gene_ids=list(
                        data.expression_gene_ids
                        if args.partition_full_axis
                        else data.expression_gene_ids[: args.query_count]
                    ),
                    data_identity=runtime.identity,
                )
                reference = None
                for batch in args.cell_batches:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    if args.partition_full_axis:
                        prediction = predict_controls(
                            runtime.objective.student,
                            runtime.index,
                            controls.expression,
                            targets,
                            device=device,
                            cell_batch=batch,
                            query_count=args.query_count,
                        )
                    else:
                        prediction = predict_query_set(
                            runtime.objective.student,
                            runtime.index,
                            controls.expression,
                            targets,
                            queries,
                            device=device,
                            cell_batch=batch,
                        )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if reference is None:
                        reference = prediction.copy()
                    difference = float(np.max(np.abs(prediction - reference)))
                    equivalent = bool(np.allclose(prediction, reference, atol=2e-5, rtol=2e-5))
                    receipt["points"].append(
                        {
                            "cell_batch": batch,
                            "seconds": elapsed,
                            "cells_per_second": 300 / elapsed,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "max_absolute_difference": difference,
                            "equivalent": equivalent,
                        }
                    )
                    atomic_json(args.output, receipt)
                    if not equivalent:
                        raise ValueError(
                            "inference batch changed predictions beyond declared tolerance"
                        )
                receipt["ordered_control_row_ids_sha256"] = sha256_json(
                    list(controls.ordered_row_ids)
                )
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(args.output, receipt)
        raise
    atomic_json(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
