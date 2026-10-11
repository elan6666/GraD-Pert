"""Bounded BR ten-update capacity evidence, never a scientific training queue."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml
from run_group import lock, validate_preflight
from run_unified_ddp import _capture_process_identity, probe
from run_unified_group import ALLOCATOR, LEASE_ROOT, child_environment
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
from gradpert.execution.train_entry import resolve_plan
from gradpert.hashing import sha256_file

SERVER_ROOT = Path("/data/yilangliu")
MODE = "br-capacity-sweep-1"
COARSE = [48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256]
POLL_SECONDS = 5


def manifest_contract(source: Path, manifest: Path) -> dict:
    matrix = read_json(manifest)
    if (
        matrix.get("schema") != MODE
        or matrix.get("steps") != 10
        or matrix.get("world_size") != 2
        or matrix.get("accumulation") != 2
        or matrix.get("refinement_grid") != 4
        or matrix.get("coarse_candidates") != COARSE[1:]
        or matrix.get("training_authorized") is not False
        or matrix.get("only_changed_entries")
        != ["model.parameters.microbatch", "training.train_batch_size"]
        or matrix.get("min_free_mib") != 2048
        or matrix.get("max_own_memory_fraction") != 0.85
    ):
        raise ValueError("capacity manifest protocol changed")
    parent = source / matrix["parent"]
    if sha256_file(parent) != matrix["parent_sha256"]:
        raise ValueError("parent BR config changed")
    baseline = yaml.safe_load(parent.read_text())
    expected = list(range(48, 257, 4))
    if [r["microbatch"] for r in matrix["rows"]] != expected:
        raise ValueError("capacity grid changed")
    for row in matrix["rows"]:
        path = source / row["config"]
        if (
            not path.resolve().is_relative_to(source.resolve())
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError("capacity config hash/path changed")
        config = yaml.safe_load(path.read_text())
        micro = row["microbatch"]
        restored = copy.deepcopy(config)
        restored["model"]["parameters"]["microbatch"] = baseline["model"]["parameters"][
            "microbatch"
        ]
        restored["training"]["train_batch_size"] = baseline["training"]["train_batch_size"]
        if (
            row["name"] != f"m{micro}"
            or row["global_batch"] != micro * 4
            or config["model"]["parameters"]["microbatch"]["value"] != micro
            or config["training"]["train_batch_size"]["value"] != micro * 4
            or restored != baseline
        ):
            raise ValueError("capacity profiles may vary only micro/global batch")
        params = config["model"]["parameters"]
        if params["world_size"]["value"] != 2 or params["accumulation"]["value"] != 2:
            raise ValueError("BR topology changed")
    return matrix


def prepare(source: Path, manifest: Path, runtime: Path, output: Path, policy: dict) -> dict:
    source, manifest, runtime, output = (p.resolve() for p in (source, manifest, runtime, output))
    if not output.is_relative_to(SERVER_ROOT) or output.exists():
        raise ValueError("capacity output must be a fresh server directory")
    validate_policy(policy)
    if policy["schema"] != DYNAMIC_SCHEMA:
        raise ValueError("capacity requires the authorized dynamic policy")
    matrix = manifest_contract(source, manifest)
    rows = []
    for row in matrix["rows"]:
        plan = resolve_plan(
            argparse.Namespace(
                config=source / row["config"],
                runtime=runtime,
                data_root=None,
                gpu="0,1",
                seed=1,
                defer_test=True,
            )
        )
        if Path(plan["run_root"]).exists() or Path(plan["test_root"]).exists():
            raise FileExistsError("capacity plan must have fresh native IDs")
        rows.append({"name": row["name"], "microbatch": row["microbatch"], "plan": plan})
    queue = {
        "schema": "unified-ddp-queue-1",
        "mode": MODE,
        "source": str(source),
        "source_commit": rows[0]["plan"]["source_commit"],
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "runtime": str(runtime),
        "runtime_sha256": sha256_file(runtime),
        "rows": rows,
        "steps": 10,
        "shared_gpu_policy": copy.deepcopy(policy),
        "output": str(output),
        "training_authorized": False,
    }
    verify(queue)
    output.mkdir(parents=False)
    atomic_json(output / "queue.json", queue)
    return queue


def verify(queue: dict) -> None:
    if (
        queue.get("schema") != "unified-ddp-queue-1"
        or queue.get("mode") != MODE
        or queue.get("steps") != 10
        or queue.get("training_authorized") is not False
    ):
        raise ValueError("unknown capacity queue")
    validate_policy(queue["shared_gpu_policy"])
    if queue["shared_gpu_policy"]["schema"] != DYNAMIC_SCHEMA:
        raise ValueError("capacity policy changed")
    source = Path(queue["source"])
    for key in ("manifest", "runtime"):
        if sha256_file(Path(queue[key])) != queue[key + "_sha256"]:
            raise ValueError("sealed capacity input changed")
    runtime_path = Path(queue["runtime"]).resolve()
    runtime = read_json(runtime_path)
    if Path(runtime["source_root"]).resolve() != source.resolve():
        raise ValueError("capacity runtime source binding changed")
    matrix = manifest_contract(source, Path(queue["manifest"]))
    if len(queue["rows"]) != len(matrix["rows"]):
        raise ValueError("capacity queue scope changed")
    ids = set()
    for row, spec in zip(queue["rows"], matrix["rows"], strict=True):
        plan = row["plan"]
        if (
            row["name"] != spec["name"]
            or row["microbatch"] != spec["microbatch"]
            or plan["config"] != str(source / spec["config"])
            or plan["config_sha256"] != spec["sha256"]
            or plan["source_commit"] != queue["source_commit"]
            or plan["repository_root"] != str(source)
            or Path(plan["runtime"]).resolve() != runtime_path
            or plan["runtime_sha256"] != queue["runtime_sha256"]
            or Path(plan["data_root"]).resolve() != Path(runtime["data_root"]).resolve()
            or Path(plan["publication"]).resolve() != Path(runtime["publication_receipt"]).resolve()
            or plan["publication_sha256"] != runtime["publication_sha256"]
            or plan["gpu"] != "0,1"
            or plan["seed"] != 1
            or plan["postfit_policy"] != "deferred"
            or plan["resolved_config"] != yaml.safe_load(Path(plan["config"]).read_text())
            or plan["run_id"] in ids
        ):
            raise ValueError("native capacity plan binding changed")
        ids.add(plan["run_id"])
        for field in ("config", "publication"):
            if sha256_file(Path(plan[field])) != plan[field + "_sha256"]:
                raise ValueError("capacity launch dependency changed")
        if any(
            not Path(plan[key]).resolve().is_relative_to(SERVER_ROOT)
            for key in ("data_root", "run_root", "test_root")
        ):
            raise ValueError("capacity artifacts must remain on server")
        identity = inspect_source_identity(
            source,
            formal=True,
            expected_repository=plan["resolved_config"]["source_code"]["repository"],
            publication_receipt=plan["publication"],
            expected_publication_receipt_sha256=plan["publication_sha256"],
        )
        if identity.dirty or identity.commit != queue["source_commit"]:
            raise ValueError("capacity source is not clean and published")


def next_candidate(results: list[dict]) -> int | None:
    tested = {r["microbatch"]: r["status"] for r in results}
    if len(tested) != len(results) or any(s not in ("passed", "oom") for s in tested.values()):
        raise ValueError("invalid capacity outcomes")
    failures = [m for m, status in tested.items() if status == "oom"]
    if not failures:
        return next((m for m in COARSE if m not in tested), None)
    ceiling = min(failures)
    floor = max((m for m, s in tested.items() if s == "passed" and m < ceiling), default=0)
    if not floor or ceiling - floor <= 4:
        return None
    candidate = floor + ((ceiling - floor) // 8) * 4
    if candidate in tested or not floor < candidate < ceiling:
        raise ValueError("capacity refinement would repeat a candidate")
    return candidate


def validate_receipt(queue: dict, row: dict, entry: dict) -> dict:
    validate_preflight(row["plan"], entry)
    receipt = read_json(Path(entry["receipt"]))
    if (
        receipt.get("kind") != "preflight_only"
        or receipt.get("steps_requested") != 10
        or receipt.get("steps_completed") != 10
        or receipt.get("inference_shape") != [300, 5000]
        or receipt.get("shared_gpu_policy") != queue["shared_gpu_policy"]
        or receipt.get("shared_queue_sha256") != sha256_file(Path(queue["output"]) / "queue.json")
        or receipt.get("source", {}).get("publication_receipt_sha256")
        != row["plan"]["publication_sha256"]
        or not receipt.get("last_terms")
        or any(
            not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in receipt["last_terms"].values()
        )
    ):
        raise ValueError("ten-update checkpoint/inference/finite evidence changed")
    return receipt


def is_oom(row: dict, directory: Path) -> bool:
    failures = list(directory.glob("rank-*-failure.json"))
    if not failures:
        return False
    observed = []
    for path in failures:
        receipt = read_json(path)
        if (
            receipt.get("status") != "failed"
            or receipt.get("config_sha256") != row["plan"]["config_sha256"]
            or receipt.get("source", {}).get("commit") != row["plan"]["source_commit"]
            or receipt.get("source", {}).get("dirty") is not False
        ):
            raise ValueError("failed probe identity changed")
        observed.append(receipt)

    def oom(receipt):
        return receipt.get(
            "error_type"
        ) == "OutOfMemoryError" or "CUDA out of memory" in receipt.get("error", "")

    for receipt in observed:
        if not oom(receipt) and not (
            receipt.get("error_type") in ("RuntimeError", "DistBackendError")
            and any(
                text in receipt.get("error", "")
                for text in (
                    "Connection closed by peer",
                    "Connection reset by peer",
                    "NCCL communicator was aborted",
                )
            )
        ):
            return False
    return any(oom(receipt) for receipt in observed)


def run_candidate(queue: dict, row: dict, directory: Path, guard) -> dict:
    output = directory / f"{row['name']}-preflight"
    if output.exists():
        raise FileExistsError("candidate probe already exists")
    guard()
    started = time.monotonic()
    known = {}
    child = None
    identity = None

    def remember():
        parents = parent_map()
        for pid in parents:
            if child is not None and descendant(pid, child.pid, parents):
                info = process_info(pid)
                if info is not None:
                    if info["uid"] != os.getuid():
                        raise RuntimeError("probe descendant ownership changed")
                    known.setdefault(pid, info)
        for pid, expected in known.items():
            current = process_info(pid)
            if current is not None and any(
                current[k] != expected[k] for k in ("uid", "start_ticks", "command_sha256")
            ):
                raise RuntimeError("probe process identity changed")

    def stop():
        errors = []
        for pid, expected in reversed(list(known.items())):
            try:
                if process_info(pid) is not None:
                    terminate_tree(pid, expected)
            except BaseException as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("owned probe cleanup failed: " + "; ".join(errors))

    try:
        with (directory / f"{row['name']}.log").open("wb") as stream:
            child = subprocess.Popen(
                probe(row, directory, shared=True),
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=child_environment(queue["source"], "0,1"),
                start_new_session=True,
            )
            identity = _capture_process_identity(child)
            known[child.pid] = {**identity, "uid": os.getuid()}
            info = process_info(child.pid)
            if (
                info is None
                or info["uid"] != os.getuid()
                or any(info[k] != identity[k] for k in ("start_ticks", "command_sha256"))
            ):
                raise RuntimeError("capacity child identity missing")
            known[child.pid] = {**info, **identity}
            atomic_json(directory / f"{row['name']}.identity.json", identity)
            while child.poll() is None:
                remember()
                guard()
                time.sleep(POLL_SECONDS)
            code = child.wait()
        remember()
        deadline = time.monotonic() + 30
        while any(process_info(pid) is not None for pid in known):
            guard()
            if time.monotonic() >= deadline:
                raise RuntimeError("probe ranks did not release after launcher exit")
            time.sleep(POLL_SECONDS)
            remember()
        observed = guard()
        if any(p["owned"] for p in observed["processes"]):
            raise RuntimeError("probe GPU ownership remains after exit")
        verify(queue)
        result = {
            "microbatch": row["microbatch"],
            "global_batch": row["microbatch"] * 4,
            "exit_code": code,
            "identity": identity,
            "wall_seconds": time.monotonic() - started,
            "log": str(directory / f"{row['name']}.log"),
        }
        if code:
            if not is_oom(row, output):
                raise RuntimeError("non-OOM native preflight failure")
            result["status"] = "oom"
        else:
            path = output / "receipt.json"
            entry = {"receipt": str(path), "sha256": sha256_file(path)}
            receipt = validate_receipt(queue, row, entry)
            if any(
                not math.isfinite(receipt[key]) or receipt[key] <= 0
                for key in (
                    "end_to_end_training_cells_per_second",
                    "startup_seconds_until_training",
                    "peak_reserved_bytes",
                )
            ):
                raise ValueError("capacity comparison measurements missing or nonfinite")
            result.update(
                status="passed",
                receipt=entry,
                throughput=receipt["end_to_end_training_cells_per_second"],
                startup_seconds=receipt["startup_seconds_until_training"],
                peak_reserved_bytes=receipt["peak_reserved_bytes"],
            )
        atomic_json(directory / f"{row['name']}.exit.json", result)
        return result
    except BaseException:
        if identity is not None:
            stop()
        elif child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        raise


def execute(queue: dict, directory: Path) -> None:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR
    ):
        raise ValueError("CPU controller visibility/allocator contract missing")
    if Path(queue["source"]).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("controller differs from sealed published source")
    if directory.resolve() != Path(queue["output"]) or (directory / "state.json").exists():
        raise ValueError("fresh sealed capacity execution required")
    verify(queue)
    if read_json(directory / "queue.json") != queue:
        raise ValueError("capacity queue file changed")
    queue_sha = sha256_file(directory / "queue.json")
    state = {
        "queue_sha256": queue_sha,
        "status": "running",
        "scientific_complete": False,
        "pid": os.getpid(),
        "results": [],
    }

    def guard():
        if sha256_file(directory / "queue.json") != queue_sha:
            raise ValueError("capacity queue hash changed")
        observed = snapshot(os.getpid())
        with (directory / "telemetry.jsonl").open("a") as stream:
            stream.write(json.dumps(observed) + "\n")
        admit(queue["shared_gpu_policy"], observed)
        return observed

    def interrupted(_sig, _frame):
        raise KeyboardInterrupt("capacity controller interrupted")

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(lock(directory / "controller.lock"))
            for gpu in ("0", "1"):
                stack.enter_context(lock(LEASE_ROOT / f"v2-gpu-{gpu}.lock"))
            observed = guard()
            if any(p["owned"] for p in observed["processes"]):
                raise RuntimeError("capacity controller already owns CUDA work")
            while (micro := next_candidate(state["results"])) is not None:
                verify(queue)
                row = next(r for r in queue["rows"] if r["microbatch"] == micro)
                state["active"] = micro
                atomic_json(directory / "state.json", state)
                state["results"].append(run_candidate(queue, row, directory, guard))
                state["active"] = None
                atomic_json(directory / "state.json", state)
            passed = [r for r in state["results"] if r["status"] == "passed"]
            failed = [r["microbatch"] for r in state["results"] if r["status"] == "oom"]
            state.update(
                status="complete",
                highest_passed=max((r["microbatch"] for r in passed), default=None),
                fastest=max(passed, key=lambda r: r["throughput"], default=None),
                first_oom=min(failed, default=None),
                bounded_ceiling_not_maximum=not failed,
                scope="ten-update capacity only; no sustained maximum or scientific completion",
            )
            atomic_json(directory / "state.json", state)
            atomic_json(directory / "COMPLETE.json", state)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(directory / "state.json", state)
        atomic_json(directory / "FAILURE.json", state)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shared-policy", type=Path)
    parser.add_argument("--queue", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.queue or any(
            getattr(args, k) is None
            for k in ("source", "manifest", "runtime", "output", "shared_policy")
        ):
            parser.error(
                "--prepare requires source/manifest/runtime/output/shared-policy and no queue"
            )
        queue = prepare(
            args.source, args.manifest, args.runtime, args.output, read_json(args.shared_policy)
        )
        print(
            json.dumps(
                {"queue": str(args.output / "queue.json"), "source_commit": queue["source_commit"]}
            )
        )
    elif args.queue and all(
        getattr(args, k) is None
        for k in ("source", "manifest", "runtime", "output", "shared_policy")
    ):
        execute(read_json(args.queue), args.queue.resolve().parent)
    else:
        parser.error("choose --prepare or --queue")


if __name__ == "__main__":
    main()
