"""Fresh Student-only CUDA inference capacity on one frozen validation population."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

PARITY_ATOL = 2e-6
PARITY_RTOL = 2e-6
KIND = "eval_batch_capacity"
EXPECTED_SHAPE = [300, 5000]
DATA_HASH_FIELDS = (
    "canonical_sha256",
    "split_sha256",
    "expression_gene_order_sha256",
    "graph_gene_order_sha256",
    "graph_manifest_sha256",
    "genept_sha256",
)


def validate_eval_binding(queue: dict, args: Any, source_commit: str) -> dict:
    """Only an exact sealed experiment leaf may use the shared-resource exception."""
    from generate_ffn_five import PRIORITY as FFN_PRIORITY
    from generate_ffn_five import SCHEMA as FFN_SCHEMA
    from generate_fgpo_family import ARMS as FGPO_ARMS
    from generate_fgpo_family import SCHEMA as FGPO_SCHEMA
    from shared_gpu import DYNAMIC_SCHEMA, validate_policy

    from gradpert.hashing import sha256_file

    validate_policy(queue["shared_gpu_policy"])
    if (
        queue.get("schema") != "unified-ddp-queue-1"
        or queue.get("family")
        not in ("br400-best-last-1", "br400-remaining-last-1", FFN_SCHEMA, FGPO_SCHEMA)
        or queue["shared_gpu_policy"]["schema"] != DYNAMIC_SCHEMA
        or queue.get("source_commit") != source_commit
        or args.gpu not in ("0", "1")
        or not 1 <= args.cell_batch <= 300
        or args.reference_batch != 128
    ):
        raise ValueError("evaluation probe requires the exact BR400 shared-resource queue")
    if queue.get("family") == "br400-remaining-last-1" and (
        queue.get("mode") != queue["family"]
        or [row["name"] for row in queue["rows"]] != ["F1", "N1", "GM1"]
        or queue.get("steps") != 10
    ):
        raise ValueError("remaining evaluation requires the exact last-only three-arm scope")
    if queue.get("family") == FFN_SCHEMA and (
        queue.get("mode") != FFN_SCHEMA
        or [row["name"] for row in queue["rows"]] != list(FFN_PRIORITY)
        or queue.get("steps") != 10
    ):
        raise ValueError("FFN evaluation requires the exact last-only five-arm scope")
    if queue.get("family") == FGPO_SCHEMA and (
        queue.get("mode") != FGPO_SCHEMA
        or [row["name"] for row in queue["rows"]] != list(FGPO_ARMS)
        or queue.get("steps") != 10
    ):
        raise ValueError("new B0 evaluation requires the exact ten-arm scope")
    rows = [row for row in queue["rows"] if row["name"] == args.row_name]
    if len(rows) != 1:
        raise ValueError("evaluation probe must select one sealed queue row")
    row, plan = rows[0], rows[0]["plan"]
    binding = {
        "row_name": args.row_name,
        "gpu": args.gpu,
        "cell_batch": args.cell_batch,
        "reference_batch": args.reference_batch,
        "output": str(args.output.resolve()),
    }
    if (
        queue.get("eval_bindings", []).count(binding) != 1
        or args.output.resolve()
        != args.shared_queue.parent.resolve() / f"{row['name']}-eval-b{args.cell_batch}-g{args.gpu}"
        or args.config.resolve() != Path(plan["config"]).resolve()
        or sha256_file(args.config) != plan["config_sha256"]
        or args.data_root.resolve() != Path(plan["data_root"]).resolve()
        or args.publication.resolve() != Path(plan["publication"]).resolve()
        or args.publication_sha256 != plan["publication_sha256"]
        or not args.config.resolve().is_relative_to(Path(queue["source"]).resolve())
    ):
        raise ValueError("evaluation probe differs from its sealed row/batch/device binding")
    return row


def compare_predictions(reference: Any, prediction: Any) -> dict[str, Any]:
    """Fixed existing inference tolerances; no scores or tolerance adaptation."""
    import numpy as np

    if reference.shape != prediction.shape or list(reference.shape) != EXPECTED_SHAPE:
        raise ValueError("capacity comparison requires the complete frozen [300,5000] population")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("capacity comparison predictions must be finite")
    difference = np.abs(reference.astype(np.float64) - prediction.astype(np.float64))
    allowed = PARITY_ATOL + PARITY_RTOL * np.abs(reference.astype(np.float64))
    result = {
        "allclose": bool((difference <= allowed).all()),
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "max_abs_error": float(difference.max()),
        "max_scaled_error": float((difference / allowed).max()),
        "mean_abs_error": float(difference.mean()),
        "scope": "fresh_seeded_student_engineering_parity_not_trained_checkpoint_effect",
        "random_order": "native_eval_seeded_gene_identity_order_unchanged",
        "precision": "native_float32_inference_finite_precision_not_bitwise_equality",
    }
    if not result["allclose"]:
        error = ValueError("evaluation cell batching exceeded the fixed inference parity tolerance")
        error.comparison = result
        raise error
    return result


def validate_receipt(queue: dict, row: dict, binding: dict, entry: dict) -> dict:
    """Independently accept only the sealed successful engineering result."""
    from gradpert.config import load_experiment_config
    from gradpert.config.v2 import V2Options
    from gradpert.data._io import read_json
    from gradpert.hashing import sha256_file, sha256_json

    path = Path(entry["receipt"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("evaluation capacity receipt hash changed")
    receipt = read_json(path)
    args = argparse.Namespace(
        config=Path(row["plan"]["config"]),
        data_root=Path(row["plan"]["data_root"]),
        publication=Path(row["plan"]["publication"]),
        publication_sha256=row["plan"]["publication_sha256"],
        output=path.parent,
        shared_queue=Path(queue["output"]) / "queue.json",
        row_name=row["name"],
        gpu=receipt.get("gpu"),
        cell_batch=receipt.get("cell_batch", 0),
        reference_batch=receipt.get("reference_batch"),
    )
    validate_eval_binding(queue, args, receipt.get("source", {}).get("commit"))
    expected_binding = {
        "row_name": args.row_name,
        "gpu": args.gpu,
        "cell_batch": args.cell_batch,
        "reference_batch": args.reference_batch,
        "output": str(path.parent.resolve()),
    }
    if binding != expected_binding:
        raise ValueError("evaluation receipt differs from the requested leaf binding")
    source = receipt.get("source", {})
    comparison = receipt.get("comparison", {})
    data = receipt.get("data", {})
    config = load_experiment_config(args.config)
    architecture, options = V2Options.parse_parameters(config.model.parameters)
    if (
        receipt.get("kind") != KIND
        or receipt.get("status") != "passed"
        or receipt.get("config_sha256") != row["plan"]["config_sha256"]
        or receipt.get("data_root") != str(args.data_root.resolve())
        or receipt.get("shared_queue_sha256") != sha256_file(args.shared_queue)
        or receipt.get("shared_gpu_policy") != queue["shared_gpu_policy"]
        or receipt.get("inference_shape") != EXPECTED_SHAPE
        or receipt.get("architecture") != architecture.payload()
        or receipt.get("evaluation_query_count") != options.eval_query_count
        or receipt.get("cell_batch") != int(config.training.eval_batch_size.value)
        or receipt.get("runtime_purpose") != "evaluation"
        or receipt.get("only_student_on_cuda") is not True
        or data.get("graph_manifest_sha256") != options.graph_manifest_sha256
        or data.get("genept_sha256") != options.genept_sha256
        or receipt.get("split") != "val"
        or receipt.get("no_truth_access") is not True
        or receipt.get("no_scientific_scores") is not True
        or source.get("dirty") is not False
        or source.get("formal_eligible") is not True
        or source.get("published_commit") != queue["source_commit"]
        or source.get("publication_receipt_sha256") != row["plan"]["publication_sha256"]
        or comparison.get("allclose") is not True
        or comparison.get("atol") != PARITY_ATOL
        or comparison.get("rtol") != PARITY_RTOL
        or receipt.get("model_state_unchanged") is not True
        or receipt.get("rng_unchanged") is not True
        or receipt.get("model_state_before_sha256") != receipt.get("model_state_after_sha256")
        or receipt.get("rng_before_sha256") != receipt.get("rng_after_sha256")
        or not isinstance(receipt.get("ordered_control_row_ids"), list)
        or len(receipt["ordered_control_row_ids"]) != 300
        or sha256_json(receipt["ordered_control_row_ids"])
        != receipt.get("ordered_control_row_ids_sha256")
        or any(
            not isinstance(data.get(name), str) or len(data[name]) != 64
            for name in (
                "canonical_sha256",
                "split_sha256",
                "expression_gene_order_sha256",
                "graph_gene_order_sha256",
                "graph_manifest_sha256",
                "genept_sha256",
            )
        )
        or any(
            not isinstance(receipt.get(name), str) or len(receipt[name]) != 64
            for name in (
                "inference_control_manifest_sha256",
                "ordered_control_row_ids_sha256",
                "reference_prediction_sha256",
                "prediction_sha256",
                "model_state_before_sha256",
                "model_state_after_sha256",
                "rng_before_sha256",
                "rng_after_sha256",
            )
        )
        or any(
            not isinstance(comparison.get(name), (int, float))
            or not math.isfinite(comparison[name])
            or comparison[name] < 0
            for name in ("max_abs_error", "max_scaled_error", "mean_abs_error")
        )
        or comparison.get("max_scaled_error", 2) > 1
        or any(
            not isinstance(receipt.get(name), (int, float))
            or not math.isfinite(receipt[name])
            or receipt[name] <= 0
            for name in (
                "inference_seconds",
                "reference_seconds",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        )
    ):
        raise ValueError("evaluation capacity evidence is incomplete or differs from its binding")
    from run_group import validate_preflight

    state = read_json(Path(queue["output"]) / "state.json")
    if (
        state.get("queue_sha256") != sha256_file(args.shared_queue)
        or state.get("source_commit") != queue["source_commit"]
    ):
        raise ValueError("evaluation capacity lost its sealed controller state identity")
    native_entry = state.get("preflights", {}).get(row["name"])
    if (
        not isinstance(native_entry, dict)
        or Path(native_entry.get("receipt", "")).resolve()
        != Path(queue["output"]).resolve() / f"{row['name']}-preflight/receipt.json"
    ):
        raise ValueError("evaluation capacity requires its exact completed native preflight")
    validate_preflight(row["plan"], native_entry)
    native = read_json(Path(native_entry["receipt"]))
    if (
        native.get("kind") != "preflight_only"
        or native.get("steps_requested") != 10
        or native.get("steps_completed") != 10
        or any(native.get("data", {}).get(name) != data[name] for name in DATA_HASH_FIELDS)
        or native.get("inference_control_manifest_sha256")
        != receipt["inference_control_manifest_sha256"]
        or native.get("ordered_control_row_ids_sha256") != receipt["ordered_control_row_ids_sha256"]
    ):
        raise ValueError("fresh evaluation capacity data/controls differ from its native preflight")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-root", "output", "publication", "shared-queue"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--publication-sha256", required=True)
    parser.add_argument("--gpu", required=True, choices=("0", "1"))
    parser.add_argument("--row-name", required=True)
    parser.add_argument("--cell-batch", type=int, required=True)
    parser.add_argument("--reference-batch", type=int, default=128)
    parser.add_argument("--resource-owner-pid", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.cell_batch <= 300 or args.reference_batch != 128:
        parser.error("cell batch must be 1..300 with fixed reference batch 128")
    if any(
        not path.resolve().is_relative_to("/data/yilangliu")
        for path in (args.output, args.data_root, args.shared_queue, args.publication)
    ):
        parser.error("evaluation capacity data and artifacts stay on the server")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("launch requires PYTORCH_ALLOC_CONF=expandable_segments:True")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or args.resource_owner_pid <= 0:
        parser.error("evaluation capacity requires one leaf under its CPU controller")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from shared_gpu import admit, descendant, parent_map, process_info, snapshot

    from gradpert.config import load_experiment_config
    from gradpert.data._io import atomic_json, read_json
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.hashing import sha256_file, sha256_json

    config = load_experiment_config(args.config)
    source = inspect_source_identity(
        Path(__file__).resolve().parents[2],
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    queue = read_json(args.shared_queue)
    validate_eval_binding(queue, args, source.commit)
    if Path(queue["source"]).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("evaluation helper differs from its sealed published source")
    if int(config.training.eval_batch_size.value) != args.cell_batch:
        raise ValueError("probe cell batch differs from its published evaluation config")
    parent = process_info(args.resource_owner_pid)
    if (
        parent is None
        or parent["uid"] != os.getuid()
        or parent["script_basename"] != "run_br400_group.py"
        or not descendant(os.getpid(), args.resource_owner_pid, parent_map())
    ):
        raise RuntimeError("evaluation probe is not a descendant of its exact CPU controller")
    policy = queue["shared_gpu_policy"]
    observed = snapshot(args.resource_owner_pid)
    admit(policy, observed)
    if any(process["owned"] for process in observed["processes"]):
        raise RuntimeError("evaluation probe needs released project GPU workers")
    args.output.mkdir(exist_ok=False)
    receipt: dict[str, Any] = {
        "kind": KIND,
        "status": "running",
        "source": source.payload(),
        "config_sha256": sha256_file(args.config),
        "data_root": str(args.data_root.resolve()),
        "gpu": args.gpu,
        "cell_batch": args.cell_batch,
        "reference_batch": args.reference_batch,
        "row_name": args.row_name,
        "shared_queue_sha256": sha256_file(args.shared_queue),
        "shared_gpu_policy": policy,
        "split": "val",
        "no_truth_access": True,
        "no_scientific_scores": True,
        "checkpoint_loaded": False,
        "engineering_only": True,
        "process_started_unix": time.time(),
        "timing_limitations": (
            "reference128 runs first; candidate timing is warmed by reference; "
            "fresh process per candidate"
        ),
    }
    atomic_json(args.output / "receipt.json", receipt)
    try:
        import numpy as np
        import torch
        from capacity_probe import state_fingerprint

        from gradpert.evaluation.data import CanonicalEvaluationData
        from gradpert.training.v2.evaluation import predict_controls
        from gradpert.training.v2.runtime import prepare_runtime

        torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], 0)
        receipt["environment"] = inspect_environment(
            Path(__file__).resolve().parents[2], device_name="cuda:0"
        ).payload()
        preparation_started = time.perf_counter()
        with (
            prepare_runtime(
                config,
                data_root=args.data_root,
                run_seed=config.training.run_seeds[0],
                device=torch.device("cpu"),
                purpose="evaluation",
            ) as runtime,
            CanonicalEvaluationData(
                dataset_id=config.dataset_id,
                protocol_id=config.data.protocol_id,
                data_root=args.data_root,
                split_name="val",
            ) as evaluation,
        ):
            receipt["data"] = runtime.identity
            receipt["runtime_preparation_seconds"] = time.perf_counter() - preparation_started
            model = runtime.objective.student.to(torch.device("cuda:0")).eval()
            if any(parameter.is_cuda for parameter in runtime.objective.teacher.parameters()):
                raise ValueError("evaluation capacity must leave Teacher on CPU")
            if runtime.optimizer.steps != 0:
                raise ValueError("fresh evaluation capacity must not update optimizer")
            draw = evaluation.control_manifest.draws[0]
            controls = evaluation.load_control_rows(tuple(draw.ordered_row_ids))
            if list(controls.expression.shape) != EXPECTED_SHAPE:
                raise ValueError(
                    "evaluation capacity requires frozen300 controls/all5000 expression genes"
                )
            gene_positions = {gene: i for i, gene in enumerate(runtime.index.gene_ids)}
            targets = tuple(
                gene_positions[gene]
                for gene in draw.condition_id.split("+")
                if gene != evaluation.split.control_condition_id
            )
            receipt.update(
                inference_condition=draw.condition_id,
                inference_control_manifest_sha256=evaluation.control_manifest_file_sha256,
                ordered_control_row_ids=list(draw.ordered_row_ids),
                ordered_control_row_ids_sha256=sha256_json(list(draw.ordered_row_ids)),
                target_gene_ids=[runtime.index.gene_ids[target] for target in targets],
                inference_shape=EXPECTED_SHAPE,
                architecture=model.options.payload(),
                evaluation_query_count=runtime.options.eval_query_count,
                runtime_purpose=runtime.purpose,
                only_student_on_cuda=True,
            )
            atomic_json(args.output / "receipt.json", receipt)
            state_before = state_fingerprint(model.state_dict())
            rng_before = state_fingerprint(
                (
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state_all(),
                    np.random.get_state(),
                    runtime.generator.bit_generator.state,
                )
            )
            outputs = []
            for name, cell_batch in (
                ("reference", args.reference_batch),
                ("inference", args.cell_batch),
            ):
                admit(policy, snapshot(args.resource_owner_pid))
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction = predict_controls(
                    model,
                    runtime.index,
                    controls.expression,
                    targets,
                    device=torch.device("cuda:0"),
                    cell_batch=cell_batch,
                    query_count=runtime.options.eval_query_count,
                )
                torch.cuda.synchronize()
                receipt[name + "_seconds"] = time.perf_counter() - started
                receipt[name + "_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                receipt[name + "_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                outputs.append(prediction)
                atomic_json(args.output / "receipt.json", receipt)
                admit(policy, snapshot(args.resource_owner_pid))
            receipt["comparison"] = compare_predictions(*outputs)
            state_after = state_fingerprint(model.state_dict())
            rng_after = state_fingerprint(
                (
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state_all(),
                    np.random.get_state(),
                    runtime.generator.bit_generator.state,
                )
            )
            receipt.update(
                reference_prediction_sha256=state_fingerprint(outputs[0]),
                prediction_sha256=state_fingerprint(outputs[1]),
                peak_allocated_bytes=receipt["inference_peak_allocated_bytes"],
                peak_reserved_bytes=receipt["inference_peak_reserved_bytes"],
                model_state_before_sha256=state_before,
                model_state_after_sha256=state_after,
                rng_before_sha256=rng_before,
                rng_after_sha256=rng_after,
                model_state_unchanged=state_before == state_after,
                rng_unchanged=rng_before == rng_after,
            )
            if not receipt["model_state_unchanged"] or not receipt["rng_unchanged"]:
                raise ValueError("evaluation capacity changed model state or inference RNG")
        receipt["status"] = "passed"
        atomic_json(args.output / "receipt.json", receipt)
        binding = next(
            item
            for item in queue["eval_bindings"]
            if (
                item["row_name"] == args.row_name
                and item["gpu"] == args.gpu
                and item["cell_batch"] == args.cell_batch
            )
        )
        validate_receipt(
            queue,
            next(row for row in queue["rows"] if row["name"] == args.row_name),
            binding,
            {
                "receipt": str(args.output / "receipt.json"),
                "sha256": sha256_file(args.output / "receipt.json"),
            },
        )
    except BaseException as error:
        if hasattr(error, "comparison"):
            receipt["comparison"] = error.comparison
        receipt.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            cuda_oom=(
                isinstance(error, torch.cuda.OutOfMemoryError) if "torch" in locals() else False
            ),
        )
        atomic_json(args.output / "receipt.json", receipt)
        raise
    atomic_json(args.output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
