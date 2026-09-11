"""Three-epoch E3 diagnostic with native validation and no test lifecycle."""

from __future__ import annotations

import json
import os
from typing import Any

from scripts.performance import profile_native_a0 as profile
from scripts.performance.bounded_epochs import (
    BoundedEpochComplete,
    audit_trainer_boundary,
    bounded_epoch_factory,
)


def main(argv: list[str] | None = None) -> int:
    parser = profile._parser()
    parser.add_argument("--stop-before-epoch", type=int, choices=(1, 3), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.coordinate != "r50_e3_batch512" or args.phase != "capacity":
        raise ValueError("epoch diagnostic requires frozen E3 coordinate and capacity preflight")
    if not args.deterministic_algorithms:
        raise ValueError("epoch exactness requires deterministic algorithms")
    if args.capture_exact_state or args.checkpoint_roundtrip_after_step is not None:
        raise ValueError("epoch runner records boundary states; step profiler flags are invalid")
    if args.resume and args.stop_before_epoch != 3:
        raise ValueError("only the sealed epoch1-to-epoch3 continuation is allowed")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator contract missing")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic CUBLAS contract missing")
    implementation = os.environ.get("GRADPERT_SPARSE_UNION_IMPL", "cpu_vectorized")
    if implementation not in {"cpu_vectorized", "cpu_array"}:
        raise ValueError("unexpected union implementation")
    root = args.run_root.resolve()
    start = 1 if args.resume else 0
    segment = f"segment-{start}-to-{args.stop_before_epoch}"
    output = root / "bounded_evidence" / f"{segment}.json"
    if output.exists() or (not args.resume and root.exists()):
        raise FileExistsError("diagnostic evidence/root already exists")

    import torch

    import gradpert.execution.native as native
    from gradpert.execution.identity import inspect_source_identity
    from gradpert.hashing import sha256_file
    from gradpert.training.trainer import GraDPertTrainer

    profile._validate_thresholds(args)
    profile._coordinate_preflight(args)
    _, frozen_identity = profile._require_reference_a0(args, args.config)
    source = inspect_source_identity(
        args.repository_root,
        formal=False,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        development_commit=args.development_commit,
    )
    if source.dirty:
        raise ValueError("diagnostic source must be clean")
    initial_config_sha = sha256_file(args.config)
    if args.resume:
        prior_path = root / "bounded_evidence/segment-0-to-1.json"
        prior = json.loads(prior_path.read_text())
        if (prior["status"], prior["source"]["commit"], prior["implementation"]) != (
            "complete",
            source.commit,
            implementation,
        ):
            raise ValueError("resume parent identity/status mismatch")
        if prior["config_sha256"] != sha256_file(args.config):
            raise ValueError("resume config mismatch")
        if prior["boundaries"][-1]["checkpoints"]["last"]["sha256"] != sha256_file(
            root / "checkpoints/last.pt"
        ):
            raise ValueError("resume checkpoint changed since sealed boundary")
    snapshot = profile._host_snapshot(root.parent)
    predicates, gpu = profile._preflight_predicates(args, snapshot)
    if not predicates or not all(predicates.values()):
        raise ValueError(f"preflight resource failure: {predicates}")
    boundaries: list[dict[str, Any]] = []
    guard = {"validation_constructor_count": 0, "test_access_attempts": 0}
    original_fit = GraDPertTrainer.fit
    original_eval = native.CanonicalEvaluationData
    original_test = GraDPertTrainer.test_best_once
    previous_deterministic = torch.are_deterministic_algorithms_enabled()

    def guarded_eval(*pos: Any, **kw: Any) -> Any:
        if kw.get("split_name") != "val":
            guard["test_access_attempts"] += 1
            raise RuntimeError("bounded comparison forbids test data")
        guard["validation_constructor_count"] += 1
        return original_eval(*pos, **kw)

    def reject_test(*_: Any, **__: Any) -> None:
        guard["test_access_attempts"] += 1
        raise RuntimeError("bounded comparison forbids testing")

    def bounded_fit(trainer: Any, **kwargs: Any) -> Any:
        if trainer.progress.completed_epochs != start:
            raise ValueError("trainer did not start at the sealed epoch boundary")

        def boundary(epoch: int) -> None:
            record = audit_trainer_boundary(trainer, epoch)
            record["state"] = profile._exact_engine_state(trainer.engine)
            record["segment_initial_boundary"] = epoch == start
            boundaries.append(record)
            profile._atomic_json(root / "bounded_evidence" / f"{segment}-epoch{epoch}.json", record)

        kwargs["train_epoch_factory"] = bounded_epoch_factory(
            kwargs["train_epoch_factory"],
            stop_before_epoch=args.stop_before_epoch,
            on_boundary=boundary,
        )
        return original_fit(trainer, **kwargs)

    error = None
    complete = False
    GraDPertTrainer.fit = bounded_fit
    GraDPertTrainer.test_best_once = reject_test
    native.CanonicalEvaluationData = guarded_eval
    torch.use_deterministic_algorithms(True)
    try:
        native.run_native_experiment(
            config_path=args.config,
            data_root=args.data_root,
            run_root=root,
            run_id=args.run_id,
            run_seed=args.run_seed,
            device_name=args.device,
            repository_root=args.repository_root,
            formal=False,
            development_commit=args.development_commit,
            resume=args.resume,
            **profile._native_coordinate_arguments(args),
        )
        raise RuntimeError("native run returned without diagnostic stop")
    except BoundedEpochComplete:
        complete = bool(boundaries and boundaries[-1]["completed_epochs"] == args.stop_before_epoch)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        GraDPertTrainer.fit = original_fit
        GraDPertTrainer.test_best_once = original_test
        native.CanonicalEvaluationData = original_eval
        torch.use_deterministic_algorithms(previous_deterministic)
    pkl = list(root.rglob("*.pkl"))
    terminal_predicates = {}
    completed_host = None
    try:
        terminal_source = inspect_source_identity(
            args.repository_root,
            formal=False,
            expected_repository="https://github.com/elan6666/GraD-Pert.git",
            development_commit=args.development_commit,
        )
        completed_host = profile._host_snapshot(root.parent)
        terminal_predicates = {
            "source_unchanged": terminal_source.payload() == source.payload()
            and not terminal_source.dirty,
            "config_unchanged": sha256_file(args.config) == initial_config_sha,
            "disk_free": completed_host["disk_free_bytes"] >= args.minimum_disk_free_bytes,
            "host_memory": completed_host["host_available_bytes"]
            >= args.minimum_host_available_bytes,
        }
        if args.device.startswith("cuda"):
            memory = torch.cuda.memory_stats(torch.device(args.device))
            terminal_predicates["no_allocator_retry_oom"] = all(
                int(value) == 0 for key, value in memory.items() if "retry" in key or "oom" in key
            )
    except BaseException as exc:
        complete = False
        error = error or {"type": type(exc).__name__, "message": str(exc)}
    complete = (
        complete
        and bool(terminal_predicates)
        and all(terminal_predicates.values())
        and not pkl
        and guard["test_access_attempts"] == 0
        and guard["validation_constructor_count"] == 1
        and [row["completed_epochs"] for row in boundaries]
        == list(range(start, args.stop_before_epoch + 1))
    )
    payload = {
        "schema_version": "bounded-e3-epochs-v1",
        "scientific_completion": False,
        "status": "complete" if complete else "failed",
        "error": error,
        "source": source.payload(),
        "implementation": implementation,
        "config_sha256": sha256_file(args.config),
        "frozen_identity": frozen_identity,
        "start_epoch": start,
        "stop_before_epoch": args.stop_before_epoch,
        "schedule_epochs": 50,
        "guard": guard,
        "persistent_pkl_count": len(pkl),
        "boundaries": boundaries,
        "preflight_predicates": predicates,
        "gpu": gpu,
        "started_host": snapshot,
        "completed_host": completed_host,
        "terminal_predicates": terminal_predicates,
    }
    profile._atomic_json(output, payload)
    print(str(output), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
