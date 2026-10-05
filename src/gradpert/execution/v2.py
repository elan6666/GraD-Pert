"""Version-two server lifecycle; historical native execution remains independent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, TypedDict

from gradpert.config import load_experiment_config
from gradpert.config.step_schedule import EndpointLRWarmupCosine, load_training_schedule
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.identity import inspect_environment, inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json

SERVER_ROOT = Path("/data/yilangliu")


class DatasetArgs(TypedDict):
    dataset_id: str
    protocol_id: str
    data_root: Path


def _write_live_progress(path: Path, payload: dict[str, Any]) -> None:
    """Keep optional status I/O outside the scientific success criterion."""
    try:
        atomic_json(path, payload)
    except OSError as error:
        print(f"v2 live-progress write failed: {error}", flush=True)


def _run_v2(plan: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    """Run the sealed fixed-epoch lifecycle, then test both checkpoint roles."""
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("v2 requires PYTORCH_ALLOC_CONF=expandable_segments:True")
    config = load_experiment_config(plan["config"])
    if config.model_id != "gradpert_v2" or config.training.formal_run_policy not in {
        "v2_fixed_1",
        "v2_fixed_3",
        "v2_fixed_5",
        "v2_fixed_6",
        "v2_fixed_20",
        "v2_fixed_50",
    }:
        raise ValueError("v2 execution requires its explicit fixed-epoch configuration")
    if sha256_file(Path(plan["config"])) != plan["config_sha256"]:
        raise ValueError("configuration changed after planning")
    if Path(plan["repository_root"]).resolve() != Path(__file__).resolve().parents[3]:
        raise ValueError("execution package differs from planned source checkout")
    root = Path(plan["run_root"]).resolve()
    data_root = Path(plan["data_root"]).resolve(strict=True)
    if not all(p.is_relative_to(SERVER_ROOT) for p in (root, data_root)):
        raise ValueError("v2 scientific execution stays on the server")
    source = inspect_source_identity(
        plan["repository_root"],
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=plan["publication"],
        expected_publication_receipt_sha256=plan["publication_sha256"],
    )
    if source.commit != plan["source_commit"]:
        raise ValueError("source changed after planning")
    environment = inspect_environment(plan["repository_root"], device_name="cuda:0")
    schedule = load_training_schedule(config.training.scheduler.value)
    if not isinstance(schedule, EndpointLRWarmupCosine):
        raise ValueError("v2 requires the project endpoint warmup-cosine schedule")
    import torch

    from gradpert.evaluation.data import CanonicalEvaluationData
    from gradpert.evaluation.state import load_evaluation_state, prepare_evaluation_state
    from gradpert.training.v2.continuation import restore_parent
    from gradpert.training.v2.distributed import primary_call
    from gradpert.training.v2.evaluation import evaluate
    from gradpert.training.v2.exposure import checkpoint_expression_groups
    from gradpert.training.v2.joint_validation import evaluate_epoch_validation
    from gradpert.training.v2.lifecycle import ConstantStageLR, fit, test_selected
    from gradpert.training.v2.reporting import export_curves
    from gradpert.training.v2.runtime import prepare_runtime

    device = torch.device("cuda:0")

    def initialize_root() -> None:
        if root.exists() and not resume:
            raise FileExistsError("new v2 execution requires a new run root")
        root.mkdir(parents=True, exist_ok=True)
        if not resume:
            atomic_json(root / "launch.json", plan)

    primary_call(initialize_root)
    with prepare_runtime(
        config, data_root=data_root, run_seed=plan["seed"], device=device
    ) as runtime:
        bootstrap_history = None
        bootstrap_best = None
        continuation_provenance = None
        if config.continuation is not None:
            bootstrap_history, bootstrap_best, continuation_provenance = restore_parent(
                config,
                data_identity=runtime.identity,
                objective=runtime.objective,
                optimizer=runtime.optimizer,
                generator=runtime.generator,
                steps_per_epoch=runtime.steps_per_epoch,
            )
        identity = {
            "source": source.payload(),
            "environment": environment.payload(),
            "data": runtime.identity,
            "config_sha256": plan["config_sha256"],
            "resolved_config_sha256": sha256_json(config.model_dump(mode="json")),
            "run_id": plan["run_id"],
        }
        if continuation_provenance is not None:
            identity["continuation"] = continuation_provenance
        manifest_path = root / "run_manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != identity:
            raise ValueError("existing run identity differs; results cannot be overwritten")
        primary_call(lambda: atomic_json(manifest_path, identity))
        primary_call(
            lambda: atomic_json(root / "resolved_config.json", config.model_dump(mode="json"))
        )
        common: DatasetArgs = {
            "dataset_id": config.dataset_id,
            "protocol_id": config.data.protocol_id,
            "data_root": data_root,
        }
        evaluation_protocol: Literal["v1", "v2"] = "v2"

        def prediction_validation() -> dict[str, Any]:
            # Called only by the primary rank's validation callback. Loss-only
            # validation never constructs the 300-control population or reference.
            prepare_evaluation_state(
                **common, validation_only=True, evaluation_protocol=evaluation_protocol
            )
            reference = load_evaluation_state(
                **common, validation_only=True, evaluation_protocol=evaluation_protocol
            )
            current_epoch = runtime.optimizer.steps // runtime.steps_per_epoch

            def val_progress(done: int, count: int, condition: str) -> None:
                _write_live_progress(
                    root / "fit/live_progress.json",
                    {
                        "schema_version": "gradpert-v2-live-progress-1",
                        "run_id": plan["run_id"],
                        "phase": "validation_prediction",
                        "epoch": current_epoch,
                        "epochs_total": int(config.training.max_epochs.value),
                        "conditions_completed": done,
                        "conditions_total": count,
                        "latest_condition": condition,
                    },
                )

            with CanonicalEvaluationData(**common, split_name="val") as data:
                return evaluate(
                    runtime.objective.student,
                    runtime.index,
                    data,
                    reference,
                    expected_split="val",
                    selection_gene_ids=(
                        tuple(int(i) for i in runtime.allowed_expression_ids)
                        if runtime.allowed_expression_ids is not None
                        else None
                    ),
                    device=device,
                    cell_batch=int(config.training.eval_batch_size.value),
                    query_count=runtime.options.eval_query_count,
                    on_condition_complete=val_progress,
                )

        def validate() -> dict[str, Any]:
            current_epoch = runtime.optimizer.steps // runtime.steps_per_epoch
            joint_batch_size = min(
                runtime.options.microbatch, int(config.training.eval_batch_size.value)
            )
            max_conditions = (
                0
                if runtime.options.max_conditions == 0
                else min(runtime.options.max_conditions, joint_batch_size)
            )
            joint_batch_count = runtime.data.validation_steps_per_epoch(
                batch_size=joint_batch_size,
                max_unique_conditions=max_conditions,
            )

            def joint_progress(done: int, mean_loss: float) -> None:
                _write_live_progress(
                    root / "fit/live_progress.json",
                    {
                        "schema_version": "gradpert-v2-live-progress-1",
                        "run_id": plan["run_id"],
                        "phase": "validation_joint",
                        "epoch": current_epoch,
                        "epochs_total": int(config.training.max_epochs.value),
                        "batches_completed": done,
                        "batches_total": joint_batch_count,
                        "joint_loss_running_mean": mean_loss,
                    },
                )

            joint_progress(0, 0.0)
            return evaluate_epoch_validation(
                runtime.objective,
                runtime.validation_batches(batch_size=joint_batch_size),
                root=root,
                batch_size=joint_batch_size,
                batch_count=joint_batch_count,
                view_seed=int(runtime.identity["run_seed"]) + 0x5A11D,
                bf16=True,
                on_batch_complete=joint_progress,
                prediction=(
                    prediction_validation
                    if runtime.options.validation_mode == "joint_and_prediction"
                    else None
                ),
            )

        journal = fit(
            runtime.objective,
            runtime.optimizer,
            root=root / "fit",
            identity=identity,
            generator=runtime.generator,
            epochs=int(config.training.max_epochs.value),
            steps_per_epoch=runtime.steps_per_epoch,
            batches=runtime.batches,
            validate=validate if runtime.options.validation_mode != "disabled" else None,
            schedule=(
                ConstantStageLR(config.continuation.learning_rate)
                if config.continuation is not None
                else schedule
            ),
            teacher_start=(
                runtime.options.teacher_end
                if config.continuation is not None
                else runtime.options.teacher_start
            ),
            teacher_end=runtime.options.teacher_end,
            microbatch=runtime.options.microbatch,
            bf16=True,
            resume=resume,
            bootstrap_history=bootstrap_history if not resume else None,
            bootstrap_best=bootstrap_best if not resume else None,
        )
        primary_call(lambda: export_curves(root / "fit"))
        # Test references and truth are accessed only after all training and selection.
        primary_call(
            lambda: prepare_evaluation_state(**common, evaluation_protocol=evaluation_protocol)
        )
        reference = load_evaluation_state(**common, evaluation_protocol=evaluation_protocol)
        with CanonicalEvaluationData(**common, split_name="test") as data:
            current_role = ["unknown"]
            current_groups: list[dict[str, tuple[int, ...]] | None] = [None]
            current_exposure: list[dict[str, Any] | None] = [None]
            history = primary_call(lambda: read_json(root / "fit/history.json"))

            def start_role(role: str) -> None:
                current_role[0] = role
                current_groups[0], current_exposure[0] = checkpoint_expression_groups(
                    history,
                    checkpoint_epoch=int(journal[role]["epoch"]),
                    expression_gene_ids=data.expression_gene_ids,
                    allowed_gene_indices=(
                        tuple(int(i) for i in runtime.allowed_expression_ids)
                        if runtime.allowed_expression_ids is not None
                        else None
                    ),
                )
                _write_live_progress(
                    root / "fit/live_progress.json",
                    {
                        "schema_version": "gradpert-v2-live-progress-1",
                        "run_id": plan["run_id"],
                        "phase": "test",
                        "checkpoint_role": role,
                        "conditions_completed": 0,
                        "conditions_total": len(data.control_manifest.draws),
                    },
                )

            def test_progress(done: int, count: int, condition: str) -> None:
                _write_live_progress(
                    root / "fit/live_progress.json",
                    {
                        "schema_version": "gradpert-v2-live-progress-1",
                        "run_id": plan["run_id"],
                        "phase": "test",
                        "checkpoint_role": current_role[0],
                        "conditions_completed": done,
                        "conditions_total": count,
                        "latest_condition": condition,
                    },
                )

            def test() -> dict[str, Any]:
                if current_groups[0] is None or current_exposure[0] is None:
                    raise ValueError("test metric groups require checkpoint exposure evidence")
                result = evaluate(
                    runtime.objective.student,
                    runtime.index,
                    data,
                    reference,
                    expected_split="test",
                    device=device,
                    cell_batch=int(config.training.eval_batch_size.value),
                    query_count=runtime.options.eval_query_count,
                    metric_gene_groups=current_groups[0],
                    on_condition_complete=test_progress,
                )
                result["expression_exposure"] = current_exposure[0]
                return result

            results = test_selected(
                runtime.objective,
                root=root / "fit",
                evaluation_identity=identity,
                test=test,
                on_role_start=start_role,
            )
        if any(root.rglob("*.pkl")):
            raise ValueError("v2 successful run must contain zero PKL files")
        complete = {
            "identity": identity,
            "epoch": journal["epoch"],
            "best": journal["best"],
            "last": journal["last"],
            "test_roles": list(results),
            "zero_pkl": True,
        }
        if continuation_provenance is not None:
            complete["continuation"] = continuation_provenance
        primary_call(lambda: atomic_json(root / "COMPLETE.json", complete))
        primary_call(
            lambda: _write_live_progress(
                root / "fit/live_progress.json",
                {
                    "schema_version": "gradpert-v2-live-progress-1",
                    "run_id": plan["run_id"],
                    "phase": "complete",
                    "epoch": journal["epoch"],
                    "epochs_total": int(config.training.max_epochs.value),
                    "test_roles": list(results),
                },
            )
        )
        return complete


def run_v2(plan: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    """Persist failures only inside a run demonstrably owned by this launch."""
    try:
        return _run_v2(plan, resume=resume)
    except BaseException as error:
        import torch

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            raise
        root = Path(plan.get("run_root", ".")).resolve()
        launch = root / "launch.json"
        if root.is_relative_to(SERVER_ROOT) and launch.is_file() and read_json(launch) == plan:
            manifest = root / "run_manifest.json"
            atomic_json(
                root / "FAILURE.json",
                {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "resume_requested": resume,
                    "identity": read_json(manifest) if manifest.is_file() else None,
                    "planned_source_commit": plan.get("source_commit"),
                    "provenance_note": "planned commit is not a substitute for run identity",
                },
            )
            _write_live_progress(
                root / "fit/live_progress.json",
                {
                    "schema_version": "gradpert-v2-live-progress-1",
                    "run_id": plan.get("run_id"),
                    "phase": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
