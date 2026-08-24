#!/usr/bin/env python3
"""Plan or deliberately execute the frozen five-dataset server matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from gradpert.data._io import atomic_json
from gradpert.execution.matrix import (
    ExperimentTask,
    MatrixRuntime,
    build_experiment_tasks,
    require_completed_task,
    require_learned_smoke_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "nonlearned", "full"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--native-python", type=Path, required=True)
    parser.add_argument("--gears-python", type=Path, required=True)
    parser.add_argument("--gears-checkout", type=Path, required=True)
    parser.add_argument("--gears-data-root", type=Path, required=True)
    parser.add_argument("--txpert-python", type=Path, required=True)
    parser.add_argument("--txpert-checkout", type=Path, required=True)
    parser.add_argument("--device", action="append", dest="devices", required=True)
    parser.add_argument("--namespace", default="formal-v1")
    parser.add_argument("--expected-commit", required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--formal", action="store_true")
    identity.add_argument("--development", action="store_true")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute-task")
    execution.add_argument("--execute-all", action="store_true")
    parser.add_argument("--resume-native-full", action="store_true")
    parser.add_argument("--receipt-root", type=Path)
    return parser


def _executable_path(path: Path) -> Path:
    """Validate an interpreter path without resolving its virtualenv symlink."""

    candidate = path.expanduser().absolute()
    if not candidate.is_file():
        raise FileNotFoundError(f"Python executable does not exist: {candidate}")
    if not os.access(candidate, os.X_OK):
        raise PermissionError(f"Python executable is not executable: {candidate}")
    return candidate


def _runtime(args: argparse.Namespace) -> MatrixRuntime:
    project_root = args.project_root.resolve(strict=True)
    return MatrixRuntime(
        project_root=project_root,
        config_root=(args.config_root or project_root / "configs" / "experiments").resolve(
            strict=True
        ),
        data_root=args.data_root.resolve(strict=True),
        runs_root=args.runs_root.resolve(),
        native_python=_executable_path(args.native_python),
        gears_python=_executable_path(args.gears_python),
        gears_checkout=args.gears_checkout.resolve(strict=True),
        gears_data_root=args.gears_data_root.resolve(strict=True),
        txpert_python=_executable_path(args.txpert_python),
        txpert_checkout=args.txpert_checkout.resolve(strict=True),
        devices=tuple(args.devices),
    )


def _execute(task: ExperimentTask, *, project_root: Path, receipt_root: Path) -> int:
    manifest = task.run_root / "small_results" / "run_manifest.json"
    if manifest.is_file():
        require_completed_task(task)
        result = {"task": task.payload(), "returncode": 0, "state": "already_materialized"}
        atomic_json(receipt_root / f"{task.run_id}.json", result)
        return 0
    if task.run_root.exists() and not task.command[-1] == "--resume":
        raise FileExistsError(
            f"incomplete run root requires explicit supported resume: {task.run_root}"
        )
    environment = os.environ.copy()
    environment.update(dict(task.environment))
    completed = subprocess.run(
        list(task.command),
        cwd=project_root,
        env=environment,
        check=False,
    )
    atomic_json(
        receipt_root / f"{task.run_id}.json",
        {
            "task": task.payload(),
            "returncode": completed.returncode,
            "state": "completed" if completed.returncode == 0 else "failed",
        },
    )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume_native_full and args.phase != "full":
        raise ValueError("--resume-native-full is valid only for the full phase")
    runtime = _runtime(args)
    formal = bool(args.formal)
    if args.phase == "full":
        if not formal:
            raise ValueError("full phase is formal-only")
        require_learned_smoke_gate(
            runtime=runtime,
            expected_commit=args.expected_commit,
            require_formal=True,
        )
    tasks = build_experiment_tasks(
        phase=args.phase,
        runtime=runtime,
        namespace=args.namespace,
        expected_commit=args.expected_commit,
        formal=formal,
        resume_native_full=args.resume_native_full,
    )
    if args.execute_task:
        selected = tuple(task for task in tasks if task.task_id == args.execute_task)
        if len(selected) != 1:
            raise ValueError(f"unknown task ID: {args.execute_task}")
    elif args.execute_all:
        selected = tasks
    else:
        selected = ()
    payload = {
        "schema_version": "experiment-matrix-plan-v1",
        "phase": args.phase,
        "formal": formal,
        "expected_commit": args.expected_commit,
        "task_count": len(tasks),
        "selected_for_execution": [task.task_id for task in selected],
        "tasks": [task.payload() for task in tasks],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not selected:
        return 0
    if args.receipt_root is None:
        raise ValueError("execution requires --receipt-root")
    receipt_root = args.receipt_root.resolve()
    receipt_root.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_root / f"{args.namespace}.{args.phase}.plan.json", payload)
    for task in selected:
        returncode = _execute(task, project_root=runtime.project_root, receipt_root=receipt_root)
        if returncode != 0:
            return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
