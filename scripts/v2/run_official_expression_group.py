"""Sealed dependent official queue; preflight then official fit and last-only evaluation."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from run_br400_group import validate_complete
from run_group import lock
from shared_gpu import admit, snapshot, validate_policy

from benchmarks.expression_holdout.config import (
    load_config,
    validate_cpu_gate,
    validate_preflight,
    validate_terminal,
)
from gradpert.data._io import atomic_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def dependency_ready(spec: dict) -> bool:
    directory = Path(spec["root"])
    if sha256_file(directory / "queue.json") != spec["queue_sha256"]:
        raise ValueError("preceding immutable queue seal changed")
    if (directory / "FAILURE.json").exists():
        raise RuntimeError("preceding queue failed; requires main repair, no baseline launch")
    terminal = directory / "COMPLETE.json"
    if not terminal.exists():
        return False
    result = json.loads(terminal.read_text())
    previous = json.loads((directory / "queue.json").read_text())
    if result.get("scientific_complete") is not True or result.get("status") != "complete":
        raise ValueError("preceding process exit is not scientific completion")
    if previous["source_commit"] != spec["source_commit"]:
        raise ValueError("preceding queue source identity drift")
    for row in previous["rows"]:
        actual = validate_complete(row["plan"])
        if result["completed"].get(row["name"]) != actual:
            raise ValueError("preceding full last evaluation receipt differs")
    return True


def prepare(runtime_path: Path, output: Path) -> dict:
    runtime = json.loads(runtime_path.read_text())
    if output.exists() or not output.resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("fresh server queue ID required")
    source = inspect_source_identity(
        ROOT,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=runtime["publication_receipt"],
        expected_publication_receipt_sha256=runtime["publication_sha256"],
    )
    policy = json.loads(Path(runtime["policy"]).read_text())
    validate_policy(policy)
    rows = []
    priority = runtime.get("priority", ["scouter", "scdfm", "gears"])
    if (
        not priority
        or len(set(priority)) != len(priority)
        or not set(priority) <= {"scouter", "scouter_project", "scouter_official", "scdfm", "gears"}
    ):
        raise ValueError("only explicitly authorized unique baseline names allowed")
    for name in priority:
        config = ROOT / "configs/v2/official_expression_20261013" / f"{name}_jurkat.yaml"
        cfg = load_config(config)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_id = f"{name}-jurkat-seed1-{stamp}-{uuid.uuid4().hex}"
        output_root = Path(runtime["runs_root"]) / run_id
        individual = {
            **runtime,
            **runtime["models"][name],
            "source_commit": source.commit,
            "device": "cuda:0",
            "seed": 1,
        }
        individual.pop("models")
        individual["excluded_conditions"] = cfg["data"].get("omit_condition_ids", [])
        validate_cpu_gate(individual, config, source.commit)
        path = output / f"{name}.runtime.json"
        rows.append(
            {
                "name": name,
                "config": str(config),
                "config_sha256": sha256_file(config),
                "runtime": str(path),
                "runtime_sha256": None,
                "run_root": str(output_root),
                "preflight_root": str(output_root) + "-preflight",
                "gpu": "0",
            }
        )
        individual["cuda_memory_fraction"] = policy["max_own_memory_fraction"]
        individual["_runtime_destination"] = str(path)
        rows[-1]["_runtime_content"] = individual
    output.mkdir(parents=True)
    for row in rows:
        individual = row.pop("_runtime_content")
        individual.pop("_runtime_destination")
        atomic_json(Path(row["runtime"]), individual)
        row["runtime_sha256"] = sha256_file(row["runtime"])
    queue = {
        "schema": "official-expression-holdout-queue-v1",
        "source": str(ROOT),
        "source_commit": source.commit,
        "dependency": runtime["dependency"],
        "shared_gpu_policy": policy,
        "rows": rows,
        "repair_budget": 0,
        "runtime_sha256": sha256_file(runtime_path),
    }
    atomic_json(output / "queue.json", queue)
    return queue


def execute(path: Path) -> None:
    queue = json.loads(path.read_text())
    seal = sha256_file(path)
    root = path.parent
    if queue["schema"] != "official-expression-holdout-queue-v1":
        raise ValueError("queue schema mismatch")
    source = inspect_source_identity(
        ROOT, formal=False, expected_repository="https://github.com/elan6666/GraD-Pert.git"
    )
    if source.commit != queue["source_commit"] or source.dirty or str(ROOT) != queue["source"]:
        raise ValueError("queue controller source changed")
    if (root / "state.json").exists():
        raise FileExistsError("do not restart or overwrite an existing official queue attempt")
    state = {
        "schema": queue["schema"],
        "queue_sha256": seal,
        "source_commit": queue["source_commit"],
        "status": "running",
        "phase": "waiting_dependency",
        "pid": os.getpid(),
        "completed": {},
        "scientific_complete": False,
    }
    atomic_json(root / "state.json", state)
    try:
        with lock(root / "controller.lock"):
            while not dependency_ready(queue["dependency"]):
                if sha256_file(path) != seal:
                    raise ValueError("immutable dependent queue changed")
                time.sleep(60)
            for row in queue["rows"]:
                config = Path(row["config"])
                runtime = Path(row["runtime"])
                if (
                    sha256_file(config) != row["config_sha256"]
                    or sha256_file(runtime) != row["runtime_sha256"]
                ):
                    raise ValueError("sealed official config/runtime changed")
                run = Path(row["run_root"])
                probe = Path(row["preflight_root"])
                if run.exists() or probe.exists():
                    raise FileExistsError("new official IDs cannot replace existing artifacts")
                with contextlib.ExitStack() as stack:
                    for gpu in ["0", "1"]:
                        stack.enter_context(
                            lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{gpu}.lock")
                        )

                    def guard():
                        if sha256_file(path) != seal:
                            raise ValueError("immutable official queue changed")
                        admit(queue["shared_gpu_policy"], snapshot(os.getpid()))

                    guard()
                    environment = {
                        **os.environ,
                        "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
                        "CUDA_VISIBLE_DEVICES": row["gpu"],
                        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2",
                    }
                    python = json.loads(runtime.read_text())["python"]
                    for mode, destination in [("preflight", probe), ("full", run)]:
                        state.update(
                            phase=f"{row['name']}_{mode}",
                            active={
                                "name": row["name"],
                                "mode": mode,
                                "run_root": str(destination),
                            },
                        )
                        atomic_json(root / "state.json", state)
                        argv = [
                            python,
                            "-m",
                            "benchmarks.expression_holdout.runner",
                            "--config",
                            str(config),
                            "--runtime",
                            str(runtime),
                            "--output",
                            str(destination),
                            "--mode",
                            mode,
                        ]
                        if mode == "full":
                            argv += ["--preflight", str(probe / "receipt.json")]
                        with (root / f"{row['name']}-{mode}.log").open("x") as log:
                            proc = subprocess.Popen(
                                argv,
                                cwd=ROOT,
                                env=environment,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                            state["active"]["pid"] = proc.pid
                            atomic_json(root / "state.json", state)
                            try:
                                while proc.poll() is None:
                                    guard()
                                    time.sleep(10)
                            except BaseException:
                                import signal

                                os.killpg(proc.pid, signal.SIGTERM)
                                proc.wait(timeout=30)
                                raise
                            if proc.returncode:
                                raise RuntimeError(
                                    f"{row['name']} {mode} failed exit={proc.returncode}"
                                )
                        receipt = json.loads((destination / "receipt.json").read_text())
                        if (
                            receipt["source_commit"] != queue["source_commit"]
                            or receipt["zero_pkl"] is not True
                        ):
                            raise ValueError(
                                "official receipt source/artifact postcondition differs"
                            )
                        if mode == "full":
                            receipt = validate_terminal(destination, row, queue["source_commit"])
                            state["completed"][row["name"]] = {
                                "root": str(destination),
                                "receipt_sha256": sha256_file(destination / "receipt.json"),
                                "last_test_sha256": receipt["last_test_sha256"],
                            }
                            atomic_json(root / "state.json", state)
                        else:
                            validate_preflight(
                                receipt,
                                config=config,
                                source_commit=queue["source_commit"],
                                official_commit=load_config(config)["source_code"]["commit"],
                                runtime_sha256=row["runtime_sha256"],
                            )
            state.update(status="complete", phase="complete", active=None, scientific_complete=True)
            atomic_json(root / "state.json", state)
            atomic_json(root / "COMPLETE.json", state)
    except BaseException as error:
        state.update(
            status="failed", phase="failed", error_type=type(error).__name__, error=str(error)
        )
        atomic_json(root / "state.json", state)
        atomic_json(root / "FAILURE.json", state)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--runtime", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--queue", type=Path)
    a = p.parse_args()
    if a.prepare:
        if not a.runtime or not a.output:
            raise ValueError("prepare requires runtime/output")
        print(json.dumps(prepare(a.runtime, a.output), indent=2))
    elif a.queue:
        execute(a.queue)
    else:
        raise ValueError("choose prepare or execute sealed queue")


if __name__ == "__main__":
    main()
