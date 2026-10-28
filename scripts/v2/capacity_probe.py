"""Complete-path CUDA capacity probe; outputs are engineering evidence only."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any


def staged_joint_probe_lr(
    schedule: Any, step: int, *, formal_pretrain_steps: int, formal_total_steps: int
) -> dict[str, Any]:
    """Map engineering5+5 updates onto the first five steps of each formal LR clock."""
    if type(step) is not int or not 0 <= step < 10:
        raise ValueError("staged-joint engineering LR requires updates0 through9")
    if not 5 <= formal_pretrain_steps <= formal_total_steps - 5:
        raise ValueError("both formal LR clocks require at least five updates")
    prefix = step < 5
    local_step = step if prefix else step - 5
    horizon = formal_pretrain_steps if prefix else formal_total_steps - formal_pretrain_steps
    return dict(
        engineering_update=step + 1,
        phase="ssl_pretrain" if prefix else "joint",
        phase_local_step=local_step,
        formal_global_step=local_step if prefix else formal_pretrain_steps + local_step,
        formal_clock_horizon=horizon,
        learning_rate=schedule.at_step(local_step, horizon)["learning_rate"],
    )


def state_fingerprint(value: Any) -> str:
    """Hash nested optimizer/RNG state exactly without holding a second model copy."""
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"array:{item.dtype}:{item.shape}:".encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"mapping[")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
            digest.update(b"]")
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}[".encode())
            for element in item:
                visit(element)
            digest.update(b"]")
        else:
            digest.update(f"{type(item).__name__}:{item!r};".encode())

    visit(value)
    return digest.hexdigest()


def validation_state(runtime: Any) -> dict[str, str]:
    """Capture state that a joint-only validation is forbidden to update."""
    import random

    import numpy as np
    import torch

    objective = runtime.objective
    return {
        "student": state_fingerprint(objective.student.state_dict()),
        "teacher": state_fingerprint(objective.teacher.state_dict()),
        "centers": state_fingerprint(
            {name: value for name, value in objective.named_buffers() if name.endswith("_center")}
        ),
        "optimizer": state_fingerprint(runtime.optimizer.state_dict()),
        "gradients": state_fingerprint(
            {name: parameter.grad for name, parameter in objective.named_parameters()}
        ),
        "numpy_rng": state_fingerprint(
            (np.random.get_state(), runtime.generator.bit_generator.state)
        ),
        "python_rng": state_fingerprint(random.getstate()),
        "torch_rng": state_fingerprint(
            (
                torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if runtime.device.type == "cuda" else [],
            )
        ),
        "pending_statistics": state_fingerprint(objective.pending),
        "auxiliary_statistics": state_fingerprint(
            getattr(objective, "auxiliary_rng_counter", None)
        ),
        "module_modes": state_fingerprint(
            {
                name: (module.training, getattr(module, "randomize_relay_order", None))
                for name, module in objective.named_modules()
            }
        ),
    }


def staged_restore_fingerprint(runtime: Any, *, continuing_joint: bool) -> str:
    """Retain historical full-state proof and bind R9 phase flags plus every RNG."""
    full_state = (
        runtime.objective.state_dict(),
        runtime.optimizer.state_dict(),
        runtime.generator.bit_generator.state,
    )
    if (
        getattr(getattr(runtime, "options", None), "training_mode", None) == "frozen_readout"
        or getattr(runtime.objective, "spectral_enabled", False)
        or getattr(runtime.objective, "response_anchor_enabled", False)
    ):
        return state_fingerprint((full_state, validation_state(runtime)))
    if not continuing_joint:
        return state_fingerprint(full_state)
    objective = runtime.objective
    return state_fingerprint(
        dict(
            full_state=full_state,
            update_state=update_parity_state(runtime),
            phase_flags={
                name: getattr(objective, name)
                for name in (
                    "prediction_enabled",
                    "teacher_updates_enabled",
                    "decode_ssl_views",
                    "training_stage",
                    "pretrain_steps",
                )
            },
            optimizer_phase=dict(
                training_stage=runtime.optimizer.training_stage,
                stage_reset_count=runtime.optimizer.stage_reset_count,
            ),
        )
    )


def inference_identity(evaluation: Any, draw: Any) -> dict[str, str]:
    """Audit the actual inference population independently of validation mode."""
    from gradpert.hashing import sha256_json

    return {
        "ordered_control_row_ids_sha256": sha256_json(list(draw.ordered_row_ids)),
        "inference_expression_gene_order_sha256": evaluation.manifest.expression_gene_order_sha256,
    }


def update_parity_state(runtime: Any) -> dict[str, str]:
    """Extend validation snapshots with all objective buffers and temperature clock."""
    state = validation_state(runtime)
    state["objective_buffers"] = state_fingerprint(dict(runtime.objective.named_buffers()))
    state["temperature_updates"] = state_fingerprint(runtime.objective._temperature_updates)
    return state


def joint_validation_check(
    runtime: Any, *, batch_size: int, bf16: bool, on_batch_complete: Any = None
) -> dict[str, Any]:
    """Run real fixed validation batches and independently verify read-only state."""
    from gradpert.training.v2.joint_validation import evaluate_joint_loss

    before = validation_state(runtime)
    started = time.perf_counter()
    batches = runtime.validation_batches(batch_size=batch_size)
    try:
        # One complete native batch is the bounded engineering gate; formal
        # per-epoch validation still traverses the entire frozen validation split.
        joint = evaluate_joint_loss(
            runtime.objective,
            itertools.islice(batches, 1),
            bf16=bf16,
            on_batch_complete=on_batch_complete,
        )
    finally:
        batches.close()
    elapsed = time.perf_counter() - started
    after = validation_state(runtime)
    unchanged = {name: before[name] == after[name] for name in before}
    if not all(unchanged.values()):
        raise ValueError(
            "joint validation mutated protected state: "
            + ", ".join(name for name, same in unchanged.items() if not same)
        )
    if not math.isfinite(joint["joint_loss"]) or any(
        not math.isfinite(value) for value in joint["components"].values()
    ):
        raise FloatingPointError("joint validation check produced nonfinite losses")
    if joint["batch_count"] < 1 or joint["cell_count"] < 1:
        raise ValueError("joint validation check requires nonempty real validation data")
    return {
        "status": "passed",
        "validation_mode": "joint_only",
        "split": "val",
        "finite": True,
        "scope": "one_complete_native_validation_batch_engineering_only",
        "full_validation": False,
        "batch_size": batch_size,
        "seconds": elapsed,
        "state_unchanged": unchanged,
        "before_sha256": before,
        "after_sha256": after,
        **joint,
    }


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
    steps = 10 if steps is None else steps
    if steps < 10:
        raise ValueError("training preflight requires at least 10 complete updates")
    # Explicit historical stress runs retain their original evidence label.
    return (steps, 8, "capacity_only") if steps >= 128 else (steps, 2, "preflight_only")


def validate_metric_scalar_mode(args: argparse.Namespace) -> None:
    """Keep scalar-packing and state-hash experiments outside formal preflight."""
    if not (args.metric_scalar_batch or args.record_update_state):
        return
    incompatible = (
        "integration_only",
        "shared_queue",
        "profile_last_update",
        "profile_pt1_stages",
        "profile_memory",
        "profile_operator_table",
        "sync_phase_timing",
        "cpu_prefetch",
        "no_sequence_checkpoint",
        "fused_sinkhorn",
        "save_no_grad_sinkhorn",
        "fused_gram",
        "gram_io",
        "aten_kda",
        "reference_unused_ssl_decoder",
        "joint_validation_check",
    )
    if not args.benchmark_only or any(getattr(args, name) for name in incompatible):
        raise ValueError("metric scalars require an isolated unprofiled benchmark-only diagnostic")


def main() -> None:
    process_started = time.perf_counter()
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
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--profile-last-update", action="store_true")
    parser.add_argument("--profile-pt1-stages", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-operator-table", action="store_true")
    parser.add_argument("--sync-phase-timing", action="store_true")
    parser.add_argument("--no-sequence-checkpoint", action="store_true")
    parser.add_argument("--cpu-prefetch", action="store_true")
    parser.add_argument("--fused-sinkhorn", action="store_true")
    parser.add_argument("--save-no-grad-sinkhorn", action="store_true")
    parser.add_argument("--fused-gram", action="store_true")
    parser.add_argument("--gram-io", action="store_true")
    parser.add_argument("--aten-kda", action="store_true")
    parser.add_argument("--metric-scalar-batch", action="store_true")
    parser.add_argument("--record-update-state", action="store_true")
    parser.add_argument("--shared-queue", type=Path)
    parser.add_argument("--resource-owner-pid", type=int)
    parser.add_argument("--joint-validation-check", action="store_true")
    parser.add_argument("--reference-unused-ssl-decoder", action="store_true")
    args = parser.parse_args()
    try:
        validate_metric_scalar_mode(args)
    except ValueError as error:
        parser.error(str(error))
    if args.gram_io:
        if args.fused_gram:
            parser.error("choose one Gram implementation")
        if (
            args.shared_queue
            or args.integration_only
            or args.profile_last_update
            or args.profile_pt1_stages
            or args.sync_phase_timing
            or args.save_no_grad_sinkhorn
            or args.reference_unused_ssl_decoder
            or args.joint_validation_check
        ):
            parser.error("Gram IO requires an isolated unprofiled benchmark-only diagnostic")
        args.fused_gram = True
    if args.aten_kda and (
        not args.benchmark_only
        or args.integration_only
        or args.shared_queue
        or args.fused_gram
        or args.fused_sinkhorn
        or args.save_no_grad_sinkhorn
        or args.cpu_prefetch
        or args.no_sequence_checkpoint
        or args.profile_last_update
        or args.profile_pt1_stages
        or args.sync_phase_timing
        or args.reference_unused_ssl_decoder
        or args.joint_validation_check
    ):
        parser.error("ATen KDA requires an isolated unprofiled benchmark-only diagnostic")
    if bool(args.shared_queue) != bool(args.resource_owner_pid):
        parser.error("shared preflight requires both the sealed queue and controller PID")
    if args.fused_gram and (
        not args.benchmark_only
        or args.fused_sinkhorn
        or args.cpu_prefetch
        or args.no_sequence_checkpoint
    ):
        parser.error("Gram fusion requires an isolated benchmark-only diagnostic")
    try:
        args.steps, warmup_steps, kind = probe_policy(
            args.integration_only, args.steps, args.benchmark_only
        )
    except ValueError as error:
        parser.error(str(error))
    if args.joint_validation_check and kind not in ("preflight_only", "capacity_only"):
        parser.error("joint-validation-check requires a ten-update or longer training preflight")
    if args.warmup_steps is not None:
        if not args.benchmark_only or not 1 <= args.warmup_steps < args.steps:
            parser.error("custom warmup requires benchmark-only and measured updates")
        warmup_steps = args.warmup_steps
    if args.profile_last_update and (
        not args.benchmark_only or args.steps - warmup_steps < 2 or args.sync_phase_timing
    ):
        parser.error("profile requires benchmark-only, two post-warmup updates and no sync timing")
    if args.profile_pt1_stages and (
        kind != "preflight_only"
        or args.steps != 10
        or args.profile_last_update
        or args.sync_phase_timing
        or args.cpu_prefetch
        or args.shared_queue
    ):
        parser.error(
            "PT1 stage profiles require an idle-GPU ten-update diagnostic "
            "without other profiling modes"
        )
    profiling = args.profile_last_update or args.profile_pt1_stages
    if args.profile_memory and not profiling:
        parser.error("profile-memory requires a profiling mode")
    if args.profile_operator_table and not profiling:
        parser.error("operator table requires a profiling mode")
    if args.cpu_prefetch and not args.benchmark_only:
        parser.error("CPU prefetch is currently benchmark-only diagnostic")
    if args.fused_sinkhorn and (
        not args.benchmark_only or args.cpu_prefetch or args.no_sequence_checkpoint
    ):
        parser.error("fused Sinkhorn requires an isolated benchmark-only diagnostic")
    if args.save_no_grad_sinkhorn and (
        not args.benchmark_only
        or args.fused_sinkhorn
        or args.fused_gram
        or args.cpu_prefetch
        or args.no_sequence_checkpoint
    ):
        parser.error("saved no-grad Sinkhorn requires an isolated benchmark-only diagnostic")
    if args.no_sequence_checkpoint and not args.benchmark_only:
        parser.error("no-sequence-checkpoint is benchmark-only diagnostic, not capacity evidence")
    profiled_indices = (
        {3, 8}
        if args.profile_pt1_stages
        else ({args.steps - 1} if args.profile_last_update else set())
    )
    measured_indices = [i for i in range(warmup_steps, args.steps) if i not in profiled_indices]
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
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("launch requires PYTORCH_ALLOC_CONF=expandable_segments:True")
    import numpy as np
    import torch
    from profile_capture import ProfileCapture, environment_snapshot, host_snapshot

    from gradpert.config import load_experiment_config
    from gradpert.config.step_schedule import (
        LRWarmupCosine,
        StepWarmupCosine,
        load_training_schedule,
    )
    from gradpert.config.v2 import V2Options
    from gradpert.data._io import atomic_json
    from gradpert.evaluation.data import CanonicalEvaluationData
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.hashing import sha256_file, sha256_json
    from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
    from gradpert.training.v2.distributed import primary_call
    from gradpert.training.v2.engine import optimizer_step, slice_cells
    from gradpert.training.v2.evaluation import predict_controls
    from gradpert.training.v2.runtime import prepare_runtime

    config = load_experiment_config(args.config)
    architecture, _ = V2Options.parse_parameters(config.model.parameters)
    if args.save_no_grad_sinkhorn and architecture.sinkhorn_backend not in ("auto", "triton"):
        parser.error("saved no-grad Sinkhorn requires a fused backend")
    source = inspect_source_identity(
        Path(__file__).resolve().parents[2],
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    environment = inspect_environment(Path(__file__).resolve().parents[2], device_name="cuda:0")
    free, total = torch.cuda.mem_get_info()
    shared_policy = None
    if args.shared_queue:
        from shared_gpu import admit, descendant, parent_map, snapshot, validate_probe_binding

        from gradpert.data._io import read_json

        shared_queue = read_json(args.shared_queue)
        shared_policy = shared_queue["shared_gpu_policy"]
        validate_probe_binding(shared_queue, args, source.commit, kind, world)
        if not descendant(os.getpid(), args.resource_owner_pid, parent_map()):
            raise RuntimeError("shared preflight is not a descendant of its published controller")
        admit(shared_policy, snapshot(args.resource_owner_pid))
        torch.cuda.set_per_process_memory_fraction(shared_policy["max_own_memory_fraction"], 0)
    elif total - free > 512 * 1024**2:
        raise RuntimeError("capacity probe requires idle GPU; existing work is preserved")
    else:
        # Idle engineering references retain the project's 90% memory ceiling.
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
    if world > 1:
        torch.cuda.set_device(0)
        torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
    primary_call(lambda: args.output.mkdir(parents=True, exist_ok=False))
    receipt = {
        "kind": "profile_only" if profiling else kind,
        "data_root": str(args.data_root.resolve()),
        "inference_exercised": not profiling and kind in ("preflight_only", "capacity_only"),
        "source": source.payload(),
        "environment": environment.payload(),
        "config_sha256": sha256_file(args.config),
        "gpu": args.gpu,
        "world_size": world,
        "steps_requested": args.steps,
        "steps_completed": 0,
        "status": "running",
        "warmup_steps": warmup_steps,
        "profile_last_update": args.profile_last_update,
        "profile_pt1_stages": args.profile_pt1_stages,
        "profile_memory": args.profile_memory,
        "profile_operator_table": args.profile_operator_table,
        "sync_phase_timing": args.sync_phase_timing,
        "cpu_prefetch_diagnostic_only": args.cpu_prefetch,
        "fused_sinkhorn_diagnostic_only": args.fused_sinkhorn,
        "save_no_grad_sinkhorn_diagnostic_only": args.save_no_grad_sinkhorn,
        "fused_gram_diagnostic_only": args.fused_gram,
        "gram_io_diagnostic_only": args.gram_io,
        "aten_kda_diagnostic_only": args.aten_kda,
        "metric_scalar_batch_diagnostic_only": args.metric_scalar_batch,
        "update_state_recording_diagnostic_only": args.record_update_state,
        "sequence_checkpoint_disabled_diagnostic_only": args.no_sequence_checkpoint,
        "joint_validation_check_requested": args.joint_validation_check,
        "hardware": environment_snapshot(torch),
        "host_before": host_snapshot(),
    }
    if shared_policy is not None:
        receipt["shared_gpu_policy"] = shared_policy
        receipt["shared_queue_sha256"] = sha256_file(args.shared_queue)
    primary_call(lambda: atomic_json(args.output / "receipt.json", receipt))
    capture = None
    candidate_contexts = ExitStack()
    try:
        if args.gram_io:
            from gram_io_candidate import candidate_scope

            candidate_contexts.enter_context(candidate_scope())
            receipt["gram_io_kernel_sha256"] = sha256_file(
                Path(__file__).resolve().parents[1] / "performance/gram_io_cuda.py"
            )
        if args.aten_kda:
            from kda_aten_candidate import candidate_scope

            load_started = time.perf_counter()
            candidate_contexts.enter_context(candidate_scope())
            receipt["aten_kda_load_seconds"] = time.perf_counter() - load_started
            receipt["aten_kda_cpp_sha256"] = sha256_file(
                Path(__file__).resolve().parents[1] / "performance/kda_aten.cpp"
            )
        schedule = load_training_schedule(config.training.scheduler.value)
        if not isinstance(schedule, (StepWarmupCosine, LRWarmupCosine)):
            raise ValueError("probe requires a sealed step-based training schedule")
        preparation_started = time.perf_counter()
        with prepare_runtime(
            config,
            data_root=args.data_root,
            run_seed=config.training.run_seeds[0],
            device=torch.device("cuda:0"),
        ) as runtime:
            anchor_gate = bool(getattr(runtime.options, "response_anchor_weight", 0))
            anchor_restores = []
            if anchor_gate and (kind != "preflight_only" or args.steps != 10):
                raise ValueError("response anchor requires native exact10 preflight")
            spectral_gate = runtime.objective.spectral_enabled
            spectral_restores = []
            spectral_steps = []
            amplitude_gate = bool(runtime.objective.spectral_amplitude_weight)
            amplitude_steps = []
            if spectral_gate and (kind != "preflight_only" or args.steps != 10):
                raise ValueError("spectral pair requires native exact10 preflight")
            readout_gate = runtime.options.training_mode == "frozen_readout"
            if readout_gate:
                from gradpert.training.v2.readout import frozen_state_proof, restore_frozen_parent

                receipt["readout_parent"] = restore_frozen_parent(
                    config,
                    data_identity=runtime.identity,
                    objective=runtime.objective,
                    optimizer=runtime.optimizer,
                )
                readout_initial_head = state_fingerprint(
                    runtime.objective.student.prediction.state_dict()
                )
                readout_initial_tensors = {
                    name: state_fingerprint(value)
                    for name, value in runtime.objective.student.prediction.state_dict().items()
                }
                readout_probe_features = torch.linspace(-1, 1, 1024, device=runtime.device).reshape(
                    2, 512
                )
                with torch.no_grad():
                    readout_initial_prediction = (
                        torch.nn.functional.softplus(
                            runtime.objective.student.prediction(readout_probe_features).float()
                        )
                        .detach()
                        .clone()
                    )
                readout_restores = []
            receipt["runtime_preparation_seconds"] = time.perf_counter() - preparation_started
            if args.fused_gram:
                from update_parity import enable_fused_gram

                receipt["fused_gram_module_count"] = enable_fused_gram(runtime.objective)
            if args.fused_sinkhorn:
                from update_parity import enable_fused_sinkhorn

                receipt["fused_sinkhorn_module_count"] = enable_fused_sinkhorn(runtime.objective)
            if args.save_no_grad_sinkhorn:
                from update_parity import set_no_grad_sinkhorn_storage

                receipt["sinkhorn_storage_module_count"] = set_no_grad_sinkhorn_storage(
                    runtime.objective, save=True
                )
            if args.no_sequence_checkpoint:
                from update_parity import disable_sequence_checkpoint

                disable_sequence_checkpoint(runtime.objective)
            if args.profile_last_update:
                capture = ProfileCapture(runtime, args.output, rank)
            total_steps = int(config.training.max_epochs.value) * runtime.steps_per_epoch
            staged_joint = runtime.options.training_mode == "ssl_then_joint"
            staged_gate = runtime.options.training_mode in ("ssl_then_mse", "ssl_then_joint")
            if args.profile_pt1_stages and (not staged_gate or staged_joint):
                raise ValueError("PT1 stage profiles require ssl_then_mse configuration")
            staged_profiles = []
            if args.reference_unused_ssl_decoder:
                if not staged_gate or staged_joint or args.shared_queue:
                    raise ValueError("unused-decoder reference is idle-GPU PT1 engineering only")
                runtime.objective.decode_ssl_views = True
                runtime.index.cache_full_context = False
                receipt["context_topology_cache_reference_disabled"] = True
            staged_states = []
            receipt["unused_ssl_decoder_reference"] = args.reference_unused_ssl_decoder
            staged_phases = []
            staged_resumes = []
            frozen_teacher = None
            initial_head = None
            initial_training_state = validation_state(runtime) if staged_joint else None
            continuous_moments = []
            joint_terms_active = []
            prefix_terms_active = []
            prefix_head_inactive = []
            staged_joint_lr_mapping = []
            if staged_gate:
                formal_pretrain_steps = runtime.objective.pretrain_steps
                if kind != "preflight_only" or args.steps != 10 or args.cpu_prefetch:
                    raise ValueError("PT1 engineering gate requires10 updates without CPU prefetch")
                receipt["staged_training_gate"] = {
                    "formal_pretrain_steps": formal_pretrain_steps,
                    "engineering_pretrain_steps": 5,
                    "engineering_joint_steps" if staged_joint else "engineering_mse_steps": 5,
                    "mode": runtime.options.training_mode,
                    "passed": False,
                }
                if staged_joint:
                    staged_joint_lr_mapping = [
                        staged_joint_probe_lr(
                            schedule,
                            step,
                            formal_pretrain_steps=formal_pretrain_steps,
                            formal_total_steps=total_steps,
                        )
                        for step in range(10)
                    ]
                    receipt["staged_training_gate"]["learning_rate_clocks"] = dict(
                        scope="engineering5plus5_first5_steps_of_each_formal_clock",
                        formal_pretrain_steps=formal_pretrain_steps,
                        formal_joint_steps=total_steps - formal_pretrain_steps,
                        formal_total_steps=total_steps,
                        mapping=staged_joint_lr_mapping,
                    )
                runtime.objective.pretrain_steps = 5
                initial_head = state_fingerprint(runtime.objective.student.prediction.state_dict())
            receipt["data"] = runtime.identity
            receipt["optimizer_routes"] = runtime.optimizer.routes
            loss_diagnostics_enabled = (
                runtime.options.prediction_reduction_override != "inherit"
                or runtime.options.process_diagnostics
            )
            if (
                not profiling
                and loss_diagnostics_enabled
                and kind
                in (
                    "preflight_only",
                    "capacity_only",
                    "integration_only",
                )
            ):
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                receipt["initial_loss_diagnostic"] = primary_call(runtime.loss_diagnostics)
                receipt["initial_loss_diagnostic_seconds"] = time.perf_counter() - started
                receipt["initial_loss_diagnostic_peak_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated()
                )
                primary_call(lambda: atomic_json(args.output / "receipt.json", receipt))
            # Hash the exact ordered row schedule before timing; no expression
            # arrays are read and this deterministic sampler does not advance RNG.
            remaining = args.steps
            schedule_hashes: list[str] = []
            for epoch_number in itertools.count():
                identities = runtime.data.training_batch_identity_specs(
                    epoch=0 if staged_joint and epoch_number == 1 else epoch_number,
                    batch_size=runtime.batch_size,
                    max_unique_conditions=runtime.options.max_conditions,
                )[: min(remaining, 5) if staged_gate else remaining]
                schedule_hashes.extend(
                    sha256_json(
                        {
                            "epoch": 0 if staged_joint and epoch_number == 1 else epoch_number,
                            "control": list(item.control_row_ids),
                            "truth": list(item.perturbed_row_ids),
                            "conditions": list(item.condition_ids),
                        }
                    )
                    for item in identities
                )
                remaining -= len(identities)
                if remaining <= 0:
                    break
            receipt["ordered_batch_schedule_sha256"] = sha256_json(schedule_hashes)
            receipt["batch_schedule_hashes"] = schedule_hashes
            receipt["view_generator_before_sha256"] = sha256_json(
                runtime.generator.bit_generator.state
            )
            durations = []
            update_states = []
            if args.record_update_state:
                update_states.append(dict(step=0, state=update_parity_state(runtime)))
            data_wait_seconds = []
            cells = []
            memory_samples = []
            gradient_reduction_seconds = []
            torch.cuda.reset_peak_memory_stats()
            step = 0
            training_started = time.perf_counter()
            receipt["startup_seconds_until_training"] = training_started - process_started
            next_batch_ready = training_started
            for epoch in itertools.count():
                batch_stream = runtime.batches(epoch, cpu_prefetch=args.cpu_prefetch)
                for batch in batch_stream:
                    # This gap includes CPU materialization, view assembly and
                    # host-to-device transfer after the preceding synchronized
                    # update. A prefetch optimization should reduce this gap.
                    torch.cuda.synchronize()
                    data_wait_seconds.append(time.perf_counter() - next_batch_ready)
                    global_cells = len(batch.control)
                    global_conditions = batch.condition_index if world > 1 else None
                    if world > 1:
                        batch = slice_cells(
                            batch, global_cells * rank // world, global_cells * (rank + 1) // world
                        )
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    timings: dict[str, float] = {}
                    if staged_joint and step == 5:
                        before = runtime.optimizer.state_dict()
                        moments = state_fingerprint(
                            {k: before[k] for k in ("muon", "adamw", "steps")}
                        )
                        runtime.objective.prepare_update(runtime.optimizer)
                        after = runtime.optimizer.state_dict()
                        continuous_moments.append(
                            moments
                            == state_fingerprint({k: after[k] for k in ("muon", "adamw", "steps")})
                        )
                    terms = optimizer_step(
                        runtime.objective,
                        runtime.optimizer,
                        batch,
                        microbatch=runtime.options.microbatch,
                        lr=staged_joint_lr_mapping[step]["learning_rate"]
                        if staged_joint
                        else runtime.options.finetune_learning_rate
                        if staged_gate and step >= 5
                        else schedule.at_step(step, total_steps)["learning_rate"],
                        momentum=runtime.options.teacher_end
                        - (runtime.options.teacher_end - runtime.options.teacher_start)
                        * (1 + math.cos(math.pi * step / (total_steps - 1)))
                        / 2,
                        bf16=True,
                        global_condition_index=global_conditions,
                        timings=timings if args.sync_phase_timing else None,
                        batch_metric_scalars=args.metric_scalar_batch,
                    )
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter() - started)
                    if args.record_update_state:
                        # Deliberately outside the update timer. These parity runs
                        # include state-copy overhead and are not speed evidence.
                        update_states.append(
                            dict(step=step + 1, state=update_parity_state(runtime), terms=terms)
                        )
                    cells.append(global_cells)
                    memory_samples.append(
                        {
                            "step": step + 1,
                            "allocated_bytes": torch.cuda.memory_allocated(),
                            "reserved_bytes": torch.cuda.memory_reserved(),
                            "host": host_snapshot(),
                        }
                    )
                    if args.sync_phase_timing:
                        gradient_reduction_seconds.append(
                            timings["gradient_reduction_seconds"] if world > 1 else 0.0
                        )
                    if spectral_gate:
                        spectral_steps.append(
                            dict(
                                completed_before=step,
                                warmup=terms.get("spectral_warmup"),
                                population=terms.get("spectral_input_count"),
                                rank=terms.get("spectral_rank"),
                                cosine=terms.get("spectral_cosine"),
                                weighted=terms.get("spectral_weighted"),
                            )
                        )
                    if amplitude_gate:
                        from generate_representation_round13 import AMPLITUDE_STEP_FIELDS

                        amplitude_steps.append(
                            dict(
                                completed_before=step,
                                **{key: terms.get(key) for key in AMPLITUDE_STEP_FIELDS},
                            )
                        )
                    step += 1
                    if args.profile_pt1_stages and step - 1 in profiled_indices:
                        staged_profiles.append(
                            dict(
                                completed_update=step,
                                phase=runtime.objective.training_stage,
                                capture=capture.close(operator_table=args.profile_operator_table),
                            )
                        )
                        capture = None
                    if staged_gate:
                        phase = runtime.objective.training_stage
                        expected_no_labels = phase == "ssl_pretrain"
                        if bool(batch.truth.isnan().all()) != expected_no_labels:
                            raise ValueError("PT1 pretraining numerical-label boundary changed")
                        staged_phases.append(phase)
                        if staged_joint and phase == "ssl_pretrain":
                            prefix_terms_active.append(
                                all(
                                    name in terms
                                    for name in (
                                        "ssl1_condition",
                                        "ssl2_dino",
                                        "au_weighted",
                                        "cr_weighted",
                                    )
                                )
                            )
                            head_parameters = set(runtime.objective.student.prediction.parameters())
                            routed_head = head_parameters | {
                                part
                                for parameter, parts in runtime.optimizer.views
                                if parameter in head_parameters
                                for part in parts
                            }
                            prefix_head_inactive.append(
                                all(p.grad is None for p in head_parameters)
                                and all(
                                    p not in opt.state
                                    for opt in (runtime.optimizer.muon, runtime.optimizer.adamw)
                                    if opt is not None
                                    for p in routed_head
                                )
                            )
                        if staged_joint and phase == "joint":
                            joint_terms_active.append(
                                all(
                                    name in terms
                                    for name in (
                                        "prediction",
                                        "ssl1_condition",
                                        "ssl2_dino",
                                        "au_weighted",
                                        "cr_weighted",
                                    )
                                )
                            )
                        if step in (5, 10):
                            staged_states.append(dict(step=step, state=validation_state(runtime)))
                        if step == 5:
                            if (
                                state_fingerprint(runtime.objective.student.prediction.state_dict())
                                != initial_head
                            ):
                                raise ValueError("pretraining updated the prediction head")
                            frozen_teacher = state_fingerprint(
                                {
                                    "teacher": runtime.objective.teacher.state_dict(),
                                    "centers": {
                                        n: b
                                        for n, b in runtime.objective.named_buffers()
                                        if n.endswith("_center")
                                    },
                                }
                            )
                    receipt["steps_completed"] = step
                    receipt["last_terms"] = terms
                    if kind != "benchmark_only" and (
                        step == max(1, args.steps // 2) or (staged_gate and step == 8)
                    ):
                        path = args.output / "resume.pt"
                        before_restore = (
                            staged_restore_fingerprint(runtime, continuing_joint=staged_joint)
                            if staged_gate or readout_gate or spectral_gate or anchor_gate
                            else None
                        )
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
                        if staged_gate or readout_gate or spectral_gate or anchor_gate:
                            same = before_restore == staged_restore_fingerprint(
                                runtime, continuing_joint=staged_joint
                            )
                            staged_resumes.append(same)
                            if readout_gate:
                                readout_restores.append(same)
                            if spectral_gate:
                                spectral_restores.append(same)
                            if anchor_gate:
                                anchor_restores.append(same)
                            if not same:
                                raise ValueError("PT1 stage checkpoint restore changed state")
                    if step % 8 == 0:
                        primary_call(lambda: atomic_json(args.output / "receipt.json", receipt))
                    if capture is not None and step == args.steps - 1:
                        capture.start(memory=args.profile_memory)
                    if args.profile_pt1_stages and step in profiled_indices:
                        directory = args.output / ("ssl_pretrain" if step < 5 else "mse_finetune")
                        directory.mkdir(exist_ok=True)
                        capture = ProfileCapture(runtime, directory, rank)
                        capture.start(memory=args.profile_memory)
                    next_batch_ready = time.perf_counter()
                    if step >= args.steps:
                        batch_stream.close()
                        break
                    if staged_gate and step == 5:
                        batch_stream.close()
                        break
                if step >= args.steps:
                    break
            receipt["training_wall_seconds"] = time.perf_counter() - training_started
            if staged_gate:
                final_teacher = state_fingerprint(
                    {
                        "teacher": runtime.objective.teacher.state_dict(),
                        "centers": {
                            n: b
                            for n, b in runtime.objective.named_buffers()
                            if n.endswith("_center")
                        },
                    }
                )
                checks = {
                    "phase_order": staged_phases == ["ssl_pretrain"] * 5 + ["mse_finetune"] * 5,
                    "teacher_centers_frozen_during_finetune": frozen_teacher == final_teacher,
                    "optimizer_reset_once": runtime.optimizer.stage_reset_count == 1,
                    "prediction_head_learned_only_in_finetune": initial_head
                    != state_fingerprint(runtime.objective.student.prediction.state_dict()),
                    "both_phase_resumes_exact": staged_resumes == [True, True],
                }
                if staged_joint:
                    checks = {
                        "phase_order": staged_phases == ["ssl_pretrain"] * 5 + ["joint"] * 5,
                        "teacher_centers_continuously_updated": frozen_teacher != final_teacher,
                        "teacher_updated_in_both_phases": all(
                            a["teacher"] != b["teacher"]
                            for a, b in zip(
                                [initial_training_state, staged_states[0]["state"]],
                                [staged_states[0]["state"], staged_states[1]["state"]],
                                strict=True,
                            )
                        ),
                        "centers_updated_in_both_phases": all(
                            a["centers"] != b["centers"]
                            for a, b in zip(
                                [initial_training_state, staged_states[0]["state"]],
                                [staged_states[0]["state"], staged_states[1]["state"]],
                                strict=True,
                            )
                        ),
                        "prefix_keeps_ssl_and_aux": prefix_terms_active == [True] * 5,
                        "prefix_head_has_no_grad_or_moments": prefix_head_inactive == [True] * 5,
                        "optimizer_never_reset": runtime.optimizer.stage_reset_count == 0,
                        "boundary_moments_retained": continuous_moments == [True],
                        "joint_keeps_ssl_prediction_and_aux": joint_terms_active == [True] * 5,
                        "prediction_head_learned_only_in_joint": initial_head
                        != state_fingerprint(runtime.objective.student.prediction.state_dict()),
                        "both_phase_resumes_exact": staged_resumes == [True, True],
                    }
                receipt["staged_training_gate"].update(
                    checks=checks,
                    passed=all(checks.values()),
                    full_state_fingerprints=staged_states,
                )
                if not all(checks.values()):
                    raise ValueError("PT1 full-update stage transition failed")
            if (
                not profiling
                and loss_diagnostics_enabled
                and kind
                in (
                    "preflight_only",
                    "capacity_only",
                    "integration_only",
                )
            ):
                started = time.perf_counter()
                receipt["updated_loss_diagnostic"] = primary_call(runtime.loss_diagnostics)
                receipt["updated_loss_diagnostic_seconds"] = time.perf_counter() - started
                receipt["training_and_diagnostic_peak_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated()
                )
            profile_summary = (
                capture.close(operator_table=args.profile_operator_table)
                if capture is not None
                else None
            )
            receipt["host_after"] = host_snapshot()
            receipt["end_to_end_training_cells_per_second"] = (
                sum(cells) / receipt["training_wall_seconds"]
            )

            if spectral_gate:
                checks = dict(
                    ten_native_updates=step == 10 and runtime.optimizer.steps == 10,
                    exact_restore=spectral_restores == [True],
                    complete_population=all(r["population"] == 320 for r in spectral_steps),
                    correct_warmup=all(
                        r["warmup"] == min(1.0, r["completed_before"] / 30) for r in spectral_steps
                    ),
                    first_update_zero=bool(spectral_steps) and spectral_steps[0]["weighted"] == 0.0,
                    finite_losses=all(
                        isinstance(r[k], (int, float)) and math.isfinite(r[k])
                        for r in spectral_steps
                        for k in ("rank", "cosine", "weighted")
                    ),
                )
                receipt["spectral_pair_gate"] = dict(
                    checks=checks,
                    passed=all(checks.values()),
                    updates=spectral_steps,
                    resume_checkpoint_sha256=receipt.get("resume_checkpoint_sha256"),
                    scope="ten_native_updates_and_exact_restore_not_sustained_capacity",
                )
                if not all(checks.values()):
                    raise ValueError("spectral pair exact10 full-population/warmup/restore failed")

            if amplitude_gate:
                from generate_representation_round13 import validate_amplitude_gate

                amplitude_proof = dict(
                    passed=all(checks.values()),
                    checks={**checks, "amplitude_energy_and_contributions": True},
                    updates=amplitude_steps,
                    resume_checkpoint_sha256=receipt.get("resume_checkpoint_sha256"),
                    scope="ten_native_updates_and_exact_restore_not_sustained_capacity",
                )
                receipt["spectral_amplitude_gate"] = amplitude_proof
                validate_amplitude_gate(
                    amplitude_proof,
                    arm=args.config.parents[1].name,
                    resume_checkpoint_sha256=receipt.get("resume_checkpoint_sha256"),
                )

            if readout_gate:
                proof = frozen_state_proof(runtime.objective, runtime.optimizer)
                changed = [
                    name
                    for name, value in runtime.objective.student.prediction.state_dict().items()
                    if readout_initial_tensors[name] != state_fingerprint(value)
                ]
                with torch.no_grad():
                    final_prediction = torch.nn.functional.softplus(
                        runtime.objective.student.prediction(readout_probe_features).float()
                    )
                proof.update(
                    prediction_changed=not torch.equal(
                        readout_initial_prediction, final_prediction
                    ),
                    changed_head_tensors=changed,
                    all_four_head_tensors_updated=len(changed) == 4,
                    native_softplus_preserved=runtime.objective.student.options.prediction_parameterization
                    == "softplus",
                    passed=True,
                    steps=step,
                    exact_restore=bool(readout_restores) and all(readout_restores),
                    head_changed=readout_initial_head
                    != state_fingerprint(runtime.objective.student.prediction.state_dict()),
                )
                if step != 10 or not all(
                    proof[k]
                    for k in (
                        "exact_restore",
                        "head_changed",
                        "prediction_changed",
                        "all_four_head_tensors_updated",
                        "native_softplus_preserved",
                    )
                ):
                    raise ValueError("R11 requires exact10 actual head updates and restore proof")
                receipt["frozen_readout_gate"] = proof

            if anchor_gate:
                checks = {
                    "ten_updates": step == 10,
                    "exact_full_state_rng_restore": bool(anchor_restores) and all(anchor_restores),
                    "complete40_group_with_main_gradients": terms.get(
                        "response_anchor_complete40_groups", 0
                    )
                    > 0,
                    "finite_anchor_loss": math.isfinite(
                        terms.get("response_anchor_loss", float("nan"))
                    ),
                }
                receipt["response_anchor_gate"] = {"checks": checks, "passed": all(checks.values())}
                if not all(checks.values()):
                    raise ValueError("response anchor native capacity/restore gate failed")

            # Exercise the identical 300-control inference path on one validation
            # condition. No test truth, scientific score or hyperparameter selection.
            def validation_probe() -> dict[str, Any]:
                validation: dict[str, Any] = {}
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
                    validation.update(inference_identity(evaluation, draw))
                return validation

            if args.joint_validation_check:
                joint_batch = min(
                    runtime.options.microbatch, int(config.training.eval_batch_size.value)
                )

                def progress(done: int, mean_loss: float) -> None:
                    receipt["joint_validation_progress"] = {
                        "batches_completed": done,
                        "joint_loss_running_mean": mean_loss,
                    }
                    atomic_json(args.output / "receipt.json", receipt)

                receipt["joint_validation_check"] = primary_call(
                    lambda: joint_validation_check(
                        runtime, batch_size=joint_batch, bf16=True, on_batch_complete=progress
                    )
                )
            if not profiling and kind in ("preflight_only", "capacity_only"):
                receipt.update(primary_call(validation_probe))
            local_measurement = {
                "rank": rank,
                "physical_gpu": devices[rank],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "training_wall_seconds": receipt["training_wall_seconds"],
                "startup_seconds_until_training": receipt["startup_seconds_until_training"],
                "runtime_preparation_seconds": receipt["runtime_preparation_seconds"],
                "warmup_update_seconds": durations[:warmup_steps],
                "warmup_data_wait_seconds": data_wait_seconds[:warmup_steps],
                "update_seconds": [durations[i] for i in measured_indices],
                "data_wait_seconds": [data_wait_seconds[i] for i in measured_indices],
                "gradient_reduction_seconds": [
                    gradient_reduction_seconds[i] for i in measured_indices
                ]
                if args.sync_phase_timing
                else [],
                "all_update_seconds": durations,
                "all_data_wait_seconds": data_wait_seconds,
                "update_states": update_states,
                "host": host_snapshot(),
                "profile": profile_summary,
                "staged_profiles": staged_profiles,
                "memory_after_update": memory_samples,
                "view_generator_after_sha256": sha256_json(runtime.generator.bit_generator.state),
                "profiled_update_seconds": [durations[i] for i in sorted(profiled_indices)],
            }
            measurements: list[Any] = [local_measurement]
            if world > 1:
                measurements = [None] * world
                torch.distributed.all_gather_object(measurements, local_measurement)
            if staged_gate:
                all_updates = [
                    max(values)
                    for values in zip(
                        *(m["all_update_seconds"] for m in measurements),
                        strict=True,
                    )
                ]
                all_waits = [
                    max(values)
                    for values in zip(
                        *(m["all_data_wait_seconds"] for m in measurements),
                        strict=True,
                    )
                ]
                receipt["staged_training_gate"]["phase_performance"] = {
                    name: {
                        "update_seconds": all_updates[start:end],
                        "data_wait_seconds": all_waits[start:end],
                        "steady_warmup_excluded_updates": 1,
                        "steady_median_update_seconds": float(
                            np.median(
                                [
                                    all_updates[i]
                                    for i in range(start + 1, end)
                                    if i not in profiled_indices
                                ]
                            )
                        ),
                        "steady_cells_per_second_including_data_wait": sum(
                            cells[i] for i in range(start + 1, end) if i not in profiled_indices
                        )
                        / sum(
                            all_updates[i] + all_waits[i]
                            for i in range(start + 1, end)
                            if i not in profiled_indices
                        ),
                        "profiled_completed_updates_excluded": [
                            i + 1 for i in range(start, end) if i in profiled_indices
                        ],
                    }
                    for name, start, end in (
                        ("ssl_pretrain", 0, 5),
                        ("joint" if staged_joint else "mse_finetune", 5, 10),
                    )
                }
            durations_measured = [
                max(times)
                for times in zip(*(m["update_seconds"] for m in measurements), strict=True)
            ]
            gaps_measured = [
                max(times)
                for times in zip(*(m["data_wait_seconds"] for m in measurements), strict=True)
            ]
            step_totals = [
                max(row)
                for row in zip(
                    *(
                        [
                            gap + duration
                            for gap, duration in zip(
                                m["data_wait_seconds"], m["update_seconds"], strict=True
                            )
                        ]
                        for m in measurements
                    ),
                    strict=True,
                )
            ]
            receipt["measured_step_seconds_including_data_wait"] = step_totals
            receipt["rank_measurements"] = measurements
            receipt["measured_data_wait_seconds"] = gaps_measured
            receipt["data_wait_fraction"] = sum(gaps_measured) / max(
                1e-9, sum(gaps_measured) + sum(durations_measured)
            )
            receipt["cpu_gpu_gap_scope"] = (
                "time after prior synchronized update until next batch and transfer complete; "
                "excludes update math and includes batch preparation"
            )
            receipt["communication_timing_scope"] = (
                "synchronized gradient averaging diagnostic; includes packing and rank wait"
                if args.sync_phase_timing
                else "disabled; no extra mid-step barriers, use separate profiler trace"
            )
            receipt["end_to_end_training_cells_per_second"] = sum(cells) / max(
                m["training_wall_seconds"] for m in measurements
            )

            receipt["measured_update_median_seconds"] = float(np.median(durations_measured))
            receipt["measured_update_p95_seconds"] = float(np.quantile(durations_measured, 0.95))
            receipt["measured_cells_per_second_including_data_wait"] = sum(
                cells[i] for i in measured_indices
            ) / sum(step_totals)
            from gradpert.modeling.v2.operators import ManifoldResidual

            backend_counts: dict[str, int] = {}
            for model in (runtime.objective.student, runtime.objective.teacher):
                for module in model.modules():
                    if isinstance(module, ManifoldResidual):
                        backend = module.resolved_sinkhorn_backend
                        backend_counts[backend] = backend_counts.get(backend, 0) + 1
            receipt["resolved_sinkhorn_module_counts"] = backend_counts
            receipt["pipeline_stats"] = runtime.data.pipeline_stats.payload()
            receipt["timing_limitations"] = (
                "step-boundary synchronization; checkpoints excluded from measured update/gap "
                "samples but included in wall time; profiled updates/export excluded from samples; "
                "wall throughput with profiler enabled is diagnostic only"
            )
            receipt.update(
                status="passed",
                peak_allocated_bytes=max(m["peak_allocated_bytes"] for m in measurements),
                peak_reserved_bytes=max(m["peak_reserved_bytes"] for m in measurements),
                measured_update_seconds=durations_measured,
                cells_per_second=sum(cells[i] for i in measured_indices) / sum(durations_measured),
                coverage=(
                    "instrumented engineering updates only; profiler/export overhead included; "
                    "not a formal preflight, capacity or scientific result"
                    if profiling
                    else (
                        "one optimizer update; checkpoint reload; "
                        "no sustained-capacity or inference evidence"
                        if kind == "integration_only"
                        else (
                            "bounded training only; no checkpoint in timed benchmark; "
                            "no sustained-capacity or inference evidence"
                            if kind == "benchmark_only"
                            else (
                                "short preflight; checkpoint continuation; "
                                "single-condition inference; "
                                "no sustained-capacity claim"
                                if kind == "preflight_only"
                                else "128+ updates; checkpoint continuation; "
                                "single-condition validation"
                            )
                        )
                    )
                ),
            )
    except BaseException as error:
        if capture is not None:
            capture.close(export=False)
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(args.output / f"rank-{rank}-failure.json", receipt)
        if rank == 0:
            atomic_json(args.output / "receipt.json", receipt)
        raise
    finally:
        candidate_contexts.close()
    primary_call(lambda: atomic_json(args.output / "receipt.json", receipt))
    if rank == 0:
        print(json.dumps(receipt, indent=2))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
