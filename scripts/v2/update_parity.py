"""Compare two complete distributed updates from the same checkpoint and RNG.

This diagnostic copies gradients to CPU and is never throughput evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch


def tree_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        digest.update(type(item).__name__.encode())
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(
                item.detach()
                .cpu()
                .reshape(-1)
                .clone(memory_format=torch.contiguous_format)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            for field in dataclasses.fields(item):
                digest.update(field.name.encode())
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(child) for child in value)
    return value


def compare_trees(
    reference: Any, actual: Any, *, atol: float = 3e-5, rtol: float = 3e-4
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": True,
        "tensor_count": 0,
        "max_absolute": 0.0,
        "failures": [],
    }

    def failure(path: str, message: str) -> None:
        result["passed"] = False
        if len(result["failures"]) < 20:
            result["failures"].append({"path": path, "message": message[:1000]})

    def visit(a: Any, b: Any, path: str) -> None:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            result["tensor_count"] += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                failure(path, "shape/dtype differs")
                return
            if a.numel():
                result["max_absolute"] = max(
                    result["max_absolute"], (a.double() - b.double()).abs().max().item()
                )
            try:
                torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
            except AssertionError as error:
                failure(path, str(error))
        elif isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for key in a:
                visit(a[key], b[key], path + "/" + str(key))
        elif isinstance(a, (tuple, list)) and type(a) is type(b) and len(a) == len(b):
            for i, (left, right) in enumerate(zip(a, b, strict=True)):
                visit(left, right, path + "/" + str(i))
        elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
            if not math.isclose(a, b, abs_tol=atol, rel_tol=rtol):
                failure(path, f"{a} != {b}")
        elif tree_digest(a) != tree_digest(b):
            failure(path, "exact value differs")

    visit(reference, actual, "root")
    return result


def execution_changes(reference: Any, candidate: Any, repeat: bool) -> list[str]:
    assert reference.relay_kernel == "eager" and not reference.relay_validate_once
    changed = [
        f.name
        for f in dataclasses.fields(reference)
        if getattr(reference, f.name) != getattr(candidate, f.name)
    ]
    if repeat:
        assert not changed, "reference repeat must use identical architecture"
    else:
        assert changed in (["relay_kernel"], ["relay_validate_once"]), (
            "one execution factor at a time"
        )
    return changed


def disable_sequence_checkpoint(objective: Any) -> None:
    """Diagnostic override on BOTH sides; never changes graph checkpointing."""
    for model in (objective.student, objective.teacher):
        model.cell.checkpoint_layers = False
        model.response.checkpoint_layers = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha256", required=True)
    parser.add_argument("--reference-repeat", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-sequence-checkpoint", action="store_true")
    parser.add_argument("--candidate-no-sequence-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.candidate_no_sequence_checkpoint and (
        args.no_sequence_checkpoint or args.reference_repeat
    ):
        parser.error("candidate-only checkpoint diagnostic cannot combine checkpoint/repeat flags")
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))
    devices = args.gpu.split(",")
    if world != 2 or sorted(devices) != ["0", "1"] or not 0 <= rank < world:
        parser.error("use torchrun with both physical GPUs")
    if not args.output.resolve().is_relative_to("/data/yilangliu"):
        parser.error("diagnostic artifacts stay on the server")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("required allocator missing")
    # No CUDA API has been called before this per-rank assignment.
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.autograd.set_multithreading_enabled(False)
    from gradpert.config import load_experiment_config
    from gradpert.config.step_schedule import (
        LRWarmupCosine,
        StepWarmupCosine,
        load_training_schedule,
    )
    from gradpert.config.v2 import V2Options
    from gradpert.data._io import atomic_json
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.hashing import sha256_file
    from gradpert.modeling.v2.model import RelayGraphLayer
    from gradpert.modeling.v2.operators import RelayDeltaAttention
    from gradpert.training.checkpoint import _rng_state
    from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
    from gradpert.training.v2.distributed import primary_call
    from gradpert.training.v2.engine import optimizer_step, slice_cells
    from gradpert.training.v2.runtime import prepare_runtime

    config, candidate = (
        load_experiment_config(args.config),
        load_experiment_config(args.candidate_config),
    )
    architecture, options = V2Options.parse_parameters(config.model.parameters)
    schedule = load_training_schedule(config.training.scheduler.value)
    if not isinstance(schedule, (LRWarmupCosine, StepWarmupCosine)):
        raise ValueError("parity diagnostic requires the sealed step schedule")
    candidate_architecture, candidate_options = V2Options.parse_parameters(
        candidate.model.parameters
    )
    changed = execution_changes(
        architecture,
        candidate_architecture,
        args.reference_repeat or args.candidate_no_sequence_checkpoint,
    )
    assert candidate_options == options
    left, right = config.model_dump(mode="json"), candidate.model_dump(mode="json")
    for field in changed:
        left["model"]["parameters"].pop(field, None)
        right["model"]["parameters"].pop(field, None)
    assert left == right, "only the selected execution option may differ"
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
        raise RuntimeError("parity diagnostic requires idle GPUs")
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
    primary_call(lambda: args.output.mkdir(parents=True, exist_ok=False))
    receipt: dict[str, Any] = {
        "status": "running",
        "kind": "two_update_parity",
        "rank": rank,
        "source": source.payload(),
        "config_sha256": sha256_file(args.config),
        "candidate_config_sha256": sha256_file(args.candidate_config),
        "environment": inspect_environment(root, device_name="cuda:0").payload(),
        "atol": 3e-5,
        "rtol": 3e-4,
        "steps": [],
        "execution_change": {name: getattr(candidate_architecture, name) for name in changed},
        "reference_repeat": args.reference_repeat,
        "candidate_sequence_checkpoint_disabled_diagnostic_only": (
            args.candidate_no_sequence_checkpoint
        ),
        "deterministic_diagnostic_only": args.deterministic,
        "sequence_checkpoint_disabled_both_sides_diagnostic_only": args.no_sequence_checkpoint,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    receipt_path = args.output / f"rank-{rank}-receipt.json"
    atomic_json(receipt_path, receipt)
    try:
        with prepare_runtime(
            config, data_root=args.data_root, run_seed=1, device=torch.device("cuda:0")
        ) as runtime:
            if args.no_sequence_checkpoint:
                disable_sequence_checkpoint(runtime.objective)
            receipt["data"] = runtime.identity
            total_steps = int(config.training.max_epochs.value) * runtime.steps_per_epoch
            initial = args.output / "initial.pt"
            save_checkpoint(
                initial,
                runtime.objective,
                runtime.optimizer,
                identity=runtime.identity,
                progress={"step": 0},
                generator=runtime.generator,
            )
            receipt["initial_checkpoint_sha256"] = sha256_file(initial)
            references = []
            original_step = runtime.optimizer.step
            gradients: dict[str, Any] = {}

            def capture_step(lr: float) -> None:
                gradients.clear()
                gradients.update(
                    {
                        name: None if p.grad is None else cpu_copy(p.grad)
                        for name, p in runtime.objective.student.named_parameters()
                    }
                )
                original_step(lr)

            runtime.optimizer.step = capture_step  # type: ignore[method-assign]
            try:
                for kernel in ("reference", "candidate"):
                    if kernel == "candidate":
                        load_checkpoint(
                            initial,
                            runtime.objective,
                            runtime.optimizer,
                            identity=runtime.identity,
                            generator=runtime.generator,
                        )
                        if args.candidate_no_sequence_checkpoint:
                            disable_sequence_checkpoint(runtime.objective)
                        for model in (runtime.objective.student, runtime.objective.teacher):
                            model.options = candidate_architecture
                            for module in model.modules():
                                if isinstance(module, RelayDeltaAttention):
                                    module.replay_sequences = (
                                        candidate_architecture.relay_kernel == "cudagraphs"
                                    )
                                    module.compiled_chunks = (
                                        candidate_architecture.relay_kernel == "inductor"
                                    )
                                if isinstance(module, RelayGraphLayer):
                                    module.validate_once = (
                                        candidate_architecture.relay_validate_once
                                    )
                    for step, batch in enumerate(itertools.islice(runtime.batches(0), 2)):
                        input_digest = tree_digest(batch)
                        rng_before = tree_digest(_rng_state())
                        count, conditions = len(batch.control), batch.condition_index
                        local = slice_cells(
                            batch, count * rank // world, count * (rank + 1) // world
                        )
                        metrics = optimizer_step(
                            runtime.objective,
                            runtime.optimizer,
                            local,
                            microbatch=options.microbatch,
                            lr=schedule.at_step(step, total_steps)["learning_rate"],
                            momentum=options.teacher_end
                            - (options.teacher_end - options.teacher_start)
                            * (1 + math.cos(math.pi * step / (total_steps - 1)))
                            / 2,
                            bf16=True,
                            global_condition_index=conditions,
                        )
                        snapshot = {
                            "losses": metrics,
                            "gradients": cpu_copy(gradients),
                            "objective": cpu_copy(runtime.objective.state_dict()),
                            "optimizer": cpu_copy(runtime.optimizer.state_dict()),
                        }
                        record = {
                            "input_sha256": input_digest,
                            "rng_before_sha256": rng_before,
                            "rng_after_sha256": tree_digest(_rng_state()),
                            "view_rng_sha256": tree_digest(runtime.generator.bit_generator.state),
                        }
                        if kernel == "reference":
                            references.append((snapshot, record))
                        else:
                            reference, expected_record = references[step]
                            compared = {
                                name: compare_trees(reference[name], snapshot[name])
                                for name in snapshot
                            }
                            passed = record == expected_record and all(
                                x["passed"] for x in compared.values()
                            )
                            receipt["steps"].append(
                                {
                                    "step": step + 1,
                                    "passed": passed,
                                    "comparison": compared,
                                    "exact_inputs_rng": record == expected_record,
                                    "eager": expected_record,
                                    "candidate": record,
                                }
                            )
                            atomic_json(receipt_path, receipt)
                            # All ranks reach the same decision before another update.
                            flag = torch.tensor(int(passed), device="cuda")
                            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
                            if not flag.item():
                                raise AssertionError(
                                    "full-update parity failed on at least one rank"
                                )
                    save_checkpoint(
                        args.output / f"{kernel}-final.pt",
                        runtime.objective,
                        runtime.optimizer,
                        identity={"data": runtime.identity, "kernel": kernel},
                        progress={"step": 2},
                        generator=runtime.generator,
                    )
            finally:
                runtime.optimizer.step = original_step  # type: ignore[method-assign]
        if candidate_architecture.relay_kernel == "cudagraphs":
            from torch._dynamo.utils import counters
            from torch._inductor.cudagraph_trees import get_manager

            manager = get_manager(0, create_if_none_exists=False)
            assert manager is not None, "no graph captured in complete update"
            receipt["replay_graphs"] = int(manager.graph_counter.__reduce__()[1][0])
            receipt["replay_skips"] = counters["inductor"]["cudagraph_skips"]
            assert receipt["replay_graphs"] > 0 and receipt["replay_skips"] == 0
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(receipt_path, receipt)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
