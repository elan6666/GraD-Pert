"""Guard one original capped BR last evaluation without redispatching training."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
from run_group import lock
from run_unified_ddp import _capture_process_identity
from run_unified_group import ALLOCATOR, LEASE_ROOT, child_environment, validate_completed
from shared_gpu import (
    DYNAMIC_SCHEMA,
    admit,
    descendant,
    parent_map,
    process_info,
    snapshot,
    terminate_tree,
    validate_policy,
)

from gradpert.data._io import atomic_json, read_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file, sha256_json

SCHEMA = "gradpert-v2-completed-br-eval-1"
SERVER_ROOT = Path("/data/yilangliu")
ORIGINAL_PLAN = (
    SERVER_ROOT
    / "GraD-Pert/development/v2-ffn-recovery-10ad0aa-20261011T163726Z/BR.launch-plan.json"
)
SCIENTIFIC_COMMIT = "b37b171fd68cb29e5c8142c66778aac6da9a6f06"
BR_CONFIG = "configs/v2/ffn_four_20261011/BR/gradpert_v2/nadig_jurkat.yaml"
POLL_SECONDS = 5


def seal(path: Path) -> dict:
    path = path.resolve(strict=True)
    if not path.is_relative_to(SERVER_ROOT):
        raise ValueError("evaluation evidence must remain on server")
    return {"path": str(path), "sha256": sha256_file(path)}


def check_seal(entry: dict) -> Path:
    path = Path(entry["path"])
    if (
        not path.is_absolute()
        or not path.resolve().is_relative_to(SERVER_ROOT)
        or sha256_file(path) != entry["sha256"]
    ):
        raise ValueError("sealed BR evaluation input changed")
    return path


def absent_outputs(plan: dict) -> None:
    root = Path(plan["run_root"])
    if any(
        path.exists()
        for path in (
            root / "postfit-last-attempt1",
            Path(plan["test_root"]),
            root / "fit/last-test.json",
            root / "COMPLETE.json",
            root / "POSTFIT_ATTEMPT1_FAILURE.json",
        )
    ):
        raise FileExistsError("original BR evaluation attempt must be fresh; no retry/overwrite")


def scientific_contract(plan: dict, checkpoint_sha256: str) -> dict:
    if not isinstance(checkpoint_sha256, str) or not re.fullmatch(
        "[0-9a-f]{64}", checkpoint_sha256
    ):
        raise ValueError("full checkpoint SHA256 required")
    source = Path(plan["repository_root"]).resolve()
    config = plan["resolved_config"]
    if (
        plan["source_commit"] != SCIENTIFIC_COMMIT
        or plan["postfit_policy"] != "deferred"
        or plan["gpu"] != "0,1"
        or plan["seed"] != 1
        or Path(plan["config"]).resolve() != source / BR_CONFIG
        or plan["resolved_config"] != yaml.safe_load(Path(plan["config"]).read_text())
        or config["training"]["max_epochs"]["value"] != 4
        or config["model"]["parameters"]["microbatch"]["value"] != 48
        or config["model"]["parameters"]["world_size"]["value"] != 2
        or config["model"]["parameters"]["accumulation"]["value"] != 2
        or config["training"]["train_batch_size"]["value"] != 192
        or plan["resource_schedule"]["capped_eval_fraction"] != 0.25
    ):
        raise ValueError("original BR scientific protocol changed")
    for field in ("config", "runtime", "publication"):
        if sha256_file(Path(plan[field])) != plan[field + "_sha256"]:
            raise ValueError("scientific source/config/runtime publication changed")
    runtime = read_json(Path(plan["runtime"]))
    if (
        Path(runtime["data_root"]).resolve() != Path(plan["data_root"]).resolve()
        or Path(runtime["publication_receipt"]).resolve() != Path(plan["publication"]).resolve()
        or runtime["publication_sha256"] != plan["publication_sha256"]
        or Path(plan["run_root"]).resolve() != Path(runtime["runs_root"]).resolve() / plan["run_id"]
        or Path(plan["test_root"]).resolve()
        != Path(runtime["runs_root"]).resolve() / (plan["run_id"] + "-test")
    ):
        raise ValueError("original native runtime/data/run/publication binding changed")
    if "repository_root" in runtime and Path(runtime["repository_root"]).resolve() != source:
        raise ValueError("scientific runtime source changed")
    if any(
        not Path(plan[key]).resolve().is_relative_to(SERVER_ROOT)
        for key in ("repository_root", "run_root", "test_root", "data_root")
    ):
        raise ValueError("scientific artifacts stay on server")
    identity = inspect_source_identity(
        source,
        formal=True,
        expected_repository=config["source_code"]["repository"],
        publication_receipt=plan["publication"],
        expected_publication_receipt_sha256=plan["publication_sha256"],
    )
    if identity.dirty or identity.commit != SCIENTIFIC_COMMIT:
        raise ValueError("original scientific source must be clean and published")
    training = validate_training_stage(plan)
    selected = training["last"]
    root = Path(plan["run_root"])
    selected_path = (root / "fit" / selected["file"]).resolve(strict=True)
    if (
        training["epoch"] != 4
        or selected["epoch"] != 4
        or training["test_roles"] != ["last"]
        or selected["sha256"] != checkpoint_sha256
        or not selected_path.is_relative_to((root / "fit").resolve())
        or sha256_file(selected_path) != checkpoint_sha256
        or sha256_file(root / "fit/last.pt") != checkpoint_sha256
    ):
        raise ValueError("BR epoch4 last checkpoint changed")
    return training


def child_command(queue: dict) -> list[str]:
    return [
        sys.executable,
        str(Path(queue["plan"]["repository_root"]) / "scripts/v2/run_unified_ddp.py"),
        "--evaluate",
        queue["launch"]["path"],
        "--attempt",
        "1",
    ]


def prepare(
    launch: Path, output: Path, policy: dict, publication: Path, checkpoint_sha256: str
) -> dict:
    launch, output = launch.resolve(), output.resolve()
    if launch != ORIGINAL_PLAN.resolve():
        raise ValueError("only the exact retired BR launch is authorized")
    if not output.is_relative_to(SERVER_ROOT) or output.exists():
        raise ValueError("fresh server controller root required")
    plan = read_json(launch)
    if any(
        output.is_relative_to(Path(plan[key]).resolve())
        for key in ("repository_root", "run_root", "test_root")
    ) or output.is_relative_to(launch.parent):
        raise ValueError("controller root cannot modify original source/run/queue")
    training = scientific_contract(plan, checkpoint_sha256)
    absent_outputs(plan)
    validate_policy(policy)
    if policy["schema"] != DYNAMIC_SCHEMA:
        raise ValueError("evaluation requires approved schema3 policy")
    source = Path(__file__).resolve().parents[2]
    published = seal(publication)
    identity = inspect_source_identity(
        source,
        formal=True,
        expected_repository=plan["resolved_config"]["source_code"]["repository"],
        publication_receipt=published["path"],
        expected_publication_receipt_sha256=published["sha256"],
    )
    if identity.dirty:
        raise ValueError("controller source must be clean and published")
    root = Path(plan["run_root"])
    paths = [
        root / p
        for p in (
            "TRAIN_COMPLETE.json",
            "launch.json",
            "resolved_config.json",
            "run_manifest.json",
            "fit/epoch_state.json",
            "fit/history.json",
            "fit/last.pt",
        )
    ]
    paths.append(root / "fit" / training["last"]["file"])
    queue = {
        "schema": SCHEMA,
        "output": str(output),
        "launch": seal(launch),
        "plan": plan,
        "training": training,
        "training_files": [seal(path) for path in paths],
        "checkpoint_sha256": checkpoint_sha256,
        "shared_gpu_policy": copy.deepcopy(policy),
        "policy_sha256": sha256_json(policy),
        "controller": {
            "repository_root": str(source),
            "commit": identity.commit,
            "publication": published,
        },
        "attempt": 1,
        "eval_fraction": 0.25,
    }
    verify(queue, fresh=True)
    output.mkdir()
    atomic_json(output / "queue.json", queue)
    return queue


def verify(queue: dict, *, fresh: bool) -> dict:
    if (
        queue.get("schema") != SCHEMA
        or queue.get("attempt") != 1
        or queue.get("eval_fraction") != 0.25
        or check_seal(queue["launch"]).resolve() != ORIGINAL_PLAN.resolve()
        or read_json(Path(queue["launch"]["path"])) != queue["plan"]
    ):
        raise ValueError("exact original BR evaluation queue changed")
    policy = queue["shared_gpu_policy"]
    validate_policy(policy)
    if policy["schema"] != DYNAMIC_SCHEMA or sha256_json(policy) != queue["policy_sha256"]:
        raise ValueError("sealed evaluation policy changed")
    for entry in queue["training_files"]:
        check_seal(entry)
    plan = queue["plan"]
    training = scientific_contract(plan, queue["checkpoint_sha256"])
    if training != queue["training"]:
        raise ValueError("sealed training receipt changed")
    controller = queue["controller"]
    output = Path(queue["output"])
    protected = [Path(plan[key]).resolve() for key in ("repository_root", "run_root", "test_root")]
    protected.extend(
        (
            Path(queue["launch"]["path"]).parent.resolve(),
            Path(controller["repository_root"]).resolve(),
        )
    )
    if (
        not output.is_absolute()
        or not output.resolve().is_relative_to(SERVER_ROOT)
        or any(output.resolve().is_relative_to(root) for root in protected)
    ):
        raise ValueError("controller root must preserve original source/run/queue")
    if Path(controller["repository_root"]).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("executing controller source differs from queue")
    publication = check_seal(controller["publication"])
    identity = inspect_source_identity(
        controller["repository_root"],
        formal=True,
        expected_repository=plan["resolved_config"]["source_code"]["repository"],
        publication_receipt=publication,
        expected_publication_receipt_sha256=controller["publication"]["sha256"],
    )
    if identity.dirty or identity.commit != controller["commit"]:
        raise ValueError("controller source must match clean published identity")
    if fresh:
        absent_outputs(plan)
    return plan


def execute(queue: dict, directory: Path) -> None:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR
    ):
        raise ValueError("CPU visibility and allocator contract required")
    if (
        directory.resolve() != Path(queue["output"]).resolve()
        or not directory.resolve().is_relative_to(SERVER_ROOT)
        or (directory / "state.json").exists()
    ):
        raise ValueError("sealed fresh controller root required")
    if read_json(directory / "queue.json") != queue:
        raise ValueError("evaluation queue differs from file")
    plan = verify(queue, fresh=True)
    queue_sha = sha256_file(directory / "queue.json")
    state = {
        "status": "running",
        "pid": os.getpid(),
        "scientific_complete": False,
        "queue_sha256": queue_sha,
        "command": child_command(queue),
        "active": None,
    }
    child = None
    identity = None
    known = {}

    def guard():
        if sha256_file(directory / "queue.json") != queue_sha:
            raise ValueError("evaluation queue file changed")
        observed = snapshot(os.getpid())
        with (directory / "telemetry.jsonl").open("a") as stream:
            stream.write(json.dumps(observed) + "\n")
        admit(queue["shared_gpu_policy"], observed)
        return observed

    def remember():
        parents = parent_map()
        for pid in parents:
            if child is not None and descendant(pid, child.pid, parents):
                info = process_info(pid)
                if info is not None:
                    if info["uid"] != os.getuid():
                        raise RuntimeError("evaluation descendant UID changed")
                    known.setdefault(pid, info)
        for pid, expected in known.items():
            current = process_info(pid)
            if current is not None and any(
                current[k] != expected[k] for k in ("pid", "uid", "start_ticks", "command_sha256")
            ):
                raise RuntimeError("evaluation process identity changed")

    def stop():
        errors = []
        for pid, expected in reversed(list(known.items())):
            try:
                if process_info(pid) is not None:
                    terminate_tree(pid, expected)
            except BaseException as error:
                errors.append(str(error))
        if errors:
            state["cleanup_errors"] = errors

    def interrupted(_sig, _frame):
        raise KeyboardInterrupt("BR evaluation controller interrupted")

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    stack = contextlib.ExitStack()
    try:
        stack.enter_context(lock(directory / "controller.lock"))
        for gpu in ("0", "1"):
            stack.enter_context(lock(LEASE_ROOT / f"v2-gpu-{gpu}.lock"))
        if any(p["owned"] for p in guard()["processes"]):
            raise RuntimeError("other owned CUDA work exists")
        verify(queue, fresh=True)
        with (directory / "evaluation.log").open("wb") as stream:
            child = subprocess.Popen(
                child_command(queue),
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=child_environment(plan["repository_root"], "0"),
                start_new_session=True,
            )
            identity = _capture_process_identity(child)
            known[child.pid] = {**identity, "uid": os.getuid()}
            info = process_info(child.pid)
            if (
                info is None
                or info["uid"] != os.getuid()
                or any(info[k] != identity[k] for k in ("pid", "start_ticks", "command_sha256"))
            ):
                raise RuntimeError("evaluation child identity missing or changed")
            state["active"] = identity
            atomic_json(directory / "state.json", state)
            while child.poll() is None:
                remember()
                guard()
                time.sleep(POLL_SECONDS)
            code = child.wait()
        state.update(exit_code=code, active=None)
        remember()
        deadline = time.monotonic() + 30
        while any(process_info(pid) is not None for pid in known):
            guard()
            if time.monotonic() >= deadline:
                raise RuntimeError("evaluation descendants remain after native exit")
            time.sleep(POLL_SECONDS)
            remember()
        if any(p["owned"] for p in guard()["processes"]):
            raise RuntimeError("evaluation CUDA work remains after exit")
        if code != 0:
            raise RuntimeError(f"native capped BR evaluation failed with exit {code}; no retry")
        verify(queue, fresh=False)
        completed = validate_completed(plan)
        state.update(status="complete", scientific_complete=True, completed=completed)
        atomic_json(directory / "state.json", state)
        atomic_json(directory / "COMPLETE.json", state)
    except BaseException as error:
        if identity is not None:
            stop()
        elif child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(directory / "state.json", state)
        atomic_json(directory / "FAILURE.json", state)
        raise
    finally:
        stack.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shared-policy", type=Path)
    parser.add_argument("--publication", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--queue", type=Path)
    args = parser.parse_args()
    fields = ("plan", "output", "shared_policy", "publication", "checkpoint_sha256")
    if args.prepare and not args.queue and all(getattr(args, key) is not None for key in fields):
        prepare(
            args.plan,
            args.output,
            read_json(args.shared_policy),
            args.publication,
            args.checkpoint_sha256,
        )
    elif args.queue and not args.prepare and all(getattr(args, key) is None for key in fields):
        execute(read_json(args.queue), args.queue.resolve().parent)
    else:
        parser.error(
            "choose --prepare with plan/output/shared-policy/publication/checkpoint-sha256, "
            "or --queue"
        )


if __name__ == "__main__":
    main()
