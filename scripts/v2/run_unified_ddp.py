"""One dual-rank fit at a time, with bounded last-test overlap when admitted."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from generate_unified_ddp import PRIORITY, verify_manifest
from run_group import lock, validate_preflight
from run_unified_group import (
    ALLOCATOR,
    LEASE_ROOT,
    child_environment,
    gpu_idle,
    probe_command,
    validate_completed,
)
from run_unified_overlap import evaluate_attempt, process_identity
from shared_gpu import admit, snapshot, terminate_tree, validate_policy

from gradpert.data._io import atomic_json, read_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.execution.train_entry import resolve_plan
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file

SCHEMA = "unified-ddp-queue-1"


def prepare(
    source: Path, manifest: Path, runtime: Path, output: Path, shared_policy: dict | None = None
) -> dict:
    if not output.resolve().is_relative_to("/data/yilangliu") or output.exists():
        raise ValueError("queue must be a fresh server directory")
    matrix = verify_manifest(source, manifest)
    indexed = {r["name"]: r for r in matrix["rows"]}
    rows = []
    for name in PRIORITY:
        row = indexed[name]
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
        if plan["config_sha256"] != row["sha256"]:
            raise ValueError("launch config differs from matrix")
        if Path(plan["run_root"]).exists() or Path(plan["test_root"]).exists():
            raise FileExistsError("fresh run must not replace old artifacts")
        rows.append({"name": name, "plan": plan})
    queue = {
        "schema": SCHEMA,
        "source": str(source),
        "source_commit": rows[0]["plan"]["source_commit"],
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "runtime": str(runtime),
        "runtime_sha256": sha256_file(runtime),
        "rows": rows,
        "initial_preflight": ["CG1", "N0"],
        "steps": 10,
        "deferred_arms": [r["name"] for r in matrix["rows"] if r["name"] not in PRIORITY],
    }
    if shared_policy is not None:
        validate_policy(shared_policy)
        queue["shared_gpu_policy"] = copy.deepcopy(shared_policy)
    output.mkdir()
    atomic_json(output / "queue.json", queue)
    verify(queue)
    return queue


def verify(queue: dict) -> None:
    if queue.get("shared_gpu_policy") is not None:
        validate_policy(queue["shared_gpu_policy"])
    if queue["schema"] != SCHEMA or [r["name"] for r in queue["rows"]] != list(PRIORITY):
        raise ValueError("queue scope/order changed")
    if queue["initial_preflight"] != ["CG1", "N0"] or queue["steps"] != 10:
        raise ValueError("bounded preflight protocol changed")
    for field in ("manifest", "runtime"):
        if sha256_file(Path(queue[field])) != queue[field + "_sha256"]:
            raise ValueError("queue file changed")
    manifest = verify_manifest(Path(queue["source"]), Path(queue["manifest"]))
    indexed = {r["name"]: r for r in manifest["rows"]}
    for row in queue["rows"]:
        plan = row["plan"]
        config = plan["resolved_config"]
        params, training = config["model"]["parameters"], config["training"]
        if (
            plan["source_commit"] != queue["source_commit"]
            or plan["gpu"] != "0,1"
            or plan["seed"] != 1
            or plan["postfit_policy"] != "deferred"
            or plan["config_sha256"] != indexed[row["name"]]["sha256"]
            or params["world_size"]["value"] != 2
            or params["accumulation"]["value"] != 2
            or training["train_batch_size"]["value"] != 4 * params["microbatch"]["value"]
            or params["validation_mode"]["value"] != "disabled"
            or training["max_epochs"]["value"] != 6
            or config.get("continuation") is not None
        ):
            raise ValueError("DDP launch differs from the six-epoch fresh protocol")
        for field in ("config", "publication"):
            if sha256_file(Path(plan[field])) != plan[field + "_sha256"]:
                raise ValueError("sealed launch dependency changed")
    plan = queue["rows"][0]["plan"]
    identity = inspect_source_identity(
        queue["source"],
        formal=True,
        expected_repository=plan["resolved_config"]["source_code"]["repository"],
        publication_receipt=plan["publication"],
        expected_publication_receipt_sha256=plan["publication_sha256"],
    )
    if identity.commit != queue["source_commit"] or identity.dirty:
        raise ValueError("queue source is not clean and published")


def probe(row: dict, directory: Path, shared: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        *probe_command({**row, "lane": "0,1"}, directory)[1:],
    ]
    if shared:
        command.extend(
            [
                "--shared-queue",
                str(directory / "queue.json"),
                "--resource-owner-pid",
                str(os.getpid()),
            ]
        )
    return command


def fit_budget(peak_reserved: int, total_bytes: int) -> float:
    """A 15% probe margin inside 60% VRAM admits a capped 65% fit +25% eval."""
    if min(peak_reserved, total_bytes) <= 0:
        raise ValueError("positive measured reserved and total memory are required")
    return 0.65 if peak_reserved * 1.15 <= total_bytes * 0.60 else 1.0


def gpu_processes_owned() -> bool:
    """Only descendants of this controller may occupy the leased GPUs."""
    parents = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            continue
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    for value in output.splitlines():
        pid = int(value.strip())
        visited = set()
        while pid != os.getpid() and pid in parents and pid not in visited:
            visited.add(pid)
            pid = parents[pid]
        if pid != os.getpid():
            return False
    return True


def finalized_plan(row: dict, entry: dict, total_bytes: int, shared_policy=None) -> dict:
    validate_preflight(row["plan"], entry)
    receipt = read_json(entry["receipt"])
    if receipt["steps_completed"] != 10 or receipt["kind"] != "preflight_only":
        raise ValueError("exact ten complete updates required")
    fraction = fit_budget(receipt["peak_reserved_bytes"], total_bytes)
    if shared_policy is not None:
        validate_policy(shared_policy)
        if receipt.get("shared_gpu_policy") != shared_policy:
            raise ValueError("probe did not exercise the sealed shared-GPU policy")
        fraction = min(fraction, shared_policy["max_own_memory_fraction"])
    plan = {
        **copy.deepcopy(row["plan"]),
        "cuda_memory_fraction": fraction,
        "resource_schedule": {
            "preflight_sha256": entry["sha256"],
            "fit_fraction": fraction,
            "capped_eval_fraction": 0.25,
            "probe_margin": 1.15,
            "measured_peak_reserved_bytes": receipt["peak_reserved_bytes"],
            "physical_total_bytes": total_bytes,
        },
    }
    if shared_policy is not None:
        plan["resource_schedule"]["shared_gpu_policy"] = copy.deepcopy(shared_policy)
    return plan


def execute(queue: dict, directory: Path) -> None:
    if os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR:
        raise ValueError("allocator contract missing")
    if (
        not directory.resolve().is_relative_to("/data/yilangliu")
        or (directory / "state.json").exists()
    ):
        raise ValueError("execution must be a fresh server queue; no implicit restart")
    verify(queue)
    state = {
        "schema": SCHEMA,
        "pid": os.getpid(),
        "status": "running",
        "phase": "preflight",
        "queue_sha256": sha256_file(directory / "queue.json"),
        "active": {},
        "preflights": {},
        "plans": {},
        "completed": {},
        "events": [],
    }
    pending_evals: list[dict] = []
    active_eval = None
    active_fit = None
    shared_policy = queue.get("shared_gpu_policy")

    def shared_check(tag):
        observed = snapshot(os.getpid())
        # Append evidence even when admission fails; never touch foreign processes.
        with (directory / "shared-gpu-telemetry.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": tag, **observed}) + "\n")
        admit(shared_policy, observed)

    def record():
        state["updated_unix"] = time.time()
        atomic_json(directory / "state.json", state)

    def start(row, stage, command, gpu, stack, attempt=1):
        nonlocal active_eval, active_fit
        verify(queue)
        if shared_policy is not None:
            shared_check(f"{row['name']}-{stage}-start")
        elif not gpu_processes_owned():
            raise RuntimeError("unrelated GPU process appeared; preserve it and stop new dispatch")
        tag = f"{row['name']}-{stage}-attempt{attempt}"
        log = stack.enter_context((directory / (tag + ".log")).open("x"))
        child = subprocess.Popen(
            command,
            cwd=queue["source"],
            env=child_environment(queue["source"], gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        item = {
            "name": row["name"],
            "stage": stage,
            "pid": child.pid,
            "identity": process_identity(child.pid),
            "log": str(directory / (tag + ".log")),
            "started_unix": time.time(),
            "attempt": attempt,
            "run_id": row["plan"]["run_id"],
        }
        state["active"][stage] = item
        state["events"].append({"event": "start", **item})
        record()
        return child, row, item

    def finish(active):
        child, row, item = active
        if shared_policy is None:
            code = child.wait()
        else:
            try:
                while True:
                    try:
                        code = child.wait(timeout=5)
                        shared_check(f"{row['name']}-{item['stage']}-finish")
                        break
                    except subprocess.TimeoutExpired:
                        shared_check(f"{row['name']}-{item['stage']}-running")
            except BaseException:
                terminate_tree(child.pid, item["identity"])
                child.wait(timeout=10)
                raise
        receipt = {**item, "exit_code": code, "finished_unix": time.time()}
        atomic_json(
            directory / f"{row['name']}-{item['stage']}-attempt{item['attempt']}.exit.json", receipt
        )
        state["active"].pop(item["stage"], None)
        state["events"].append({"event": "exit", **receipt})
        record()
        return code

    def drain_eval(stack):
        nonlocal active_eval
        if active_eval is None:
            return
        _child, row, item = active_eval
        code = finish(active_eval)
        active_eval = None
        if code:
            failure = Path(row["plan"]["run_root"]) / "POSTFIT_ATTEMPT1_FAILURE.json"
            if item["attempt"] == 1 and failure.exists() and read_json(failure)["oom"]:
                # The next fit is not launched until this exact eval retry drains.
                start_eval(row, stack, attempt=2)
                drain_eval(stack)
                return
            raise RuntimeError(f"{row['name']} eval failed with exit {code}")
        state["completed"][row["name"]] = validate_completed(row["plan"])
        record()

    def start_eval(row, stack, attempt=1, uncapped=False):
        nonlocal active_eval
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--evaluate",
            str(directory / f"{row['name']}.launch-plan.json"),
            "--attempt",
            str(attempt),
        ]
        if uncapped:
            command.append("--uncapped")
        active_eval = start(row, "eval", command, "0", stack, attempt)

    def idle():
        if shared_policy is not None:
            shared_check("admission")
            return
        while True:
            usage = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            if all(gpu_idle(usage, gpu) for gpu in ("0", "1")):
                return
            state["phase"] = "waiting_idle_dual_gpu"
            record()
            time.sleep(20)

    def preflight(row, stack):
        name = row["name"]
        if name in state["preflights"]:
            return
        drain_eval(stack)
        idle()
        state["phase"] = "preflight"
        if (directory / f"{name}-preflight").exists():
            raise FileExistsError("preflight artifacts cannot be overwritten")
        active = start(
            row, "preflight", probe(row, directory, shared_policy is not None), "0,1", stack
        )
        if finish(active):
            raise RuntimeError(f"{name} DDP preflight failed; preserve evidence, do not train")
        receipt = directory / f"{name}-preflight/receipt.json"
        entry = {"receipt": str(receipt), "sha256": sha256_file(receipt)}
        validate_preflight(row["plan"], entry)
        total = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
        )
        plan = finalized_plan(
            row, entry, min(int(n) for n in total.splitlines()) * 1024**2, shared_policy
        )
        path = directory / f"{name}.launch-plan.json"
        if path.exists():
            raise FileExistsError("final launch plan already exists")
        atomic_json(path, plan)
        state["preflights"][name] = entry
        state["plans"][name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "fit_fraction": plan["cuda_memory_fraction"],
        }
        record()

    with contextlib.ExitStack() as stack:
        stack.enter_context(lock(directory / "controller.lock"))
        stack.enter_context(lock(LEASE_ROOT / "v2-gpu-0.lock"))
        stack.enter_context(lock(LEASE_ROOT / "v2-gpu-1.lock"))
        record()
        try:
            indexed = {r["name"]: r for r in queue["rows"]}
            # Expensive CG1 is checked before any formal fit at the common batch.
            for name in queue["initial_preflight"]:
                preflight(indexed[name], stack)
            for row in queue["rows"]:
                preflight(row, stack)
                path = directory / f"{row['name']}.launch-plan.json"
                actual = {**row, "plan": read_json(path)}
                budget = actual["plan"]["cuda_memory_fraction"]
                if budget > 0.65:
                    drain_eval(stack)
                    while pending_evals:
                        start_eval(pending_evals.pop(0), stack, uncapped=True)
                        drain_eval(stack)
                if active_eval is not None and active_eval[0].poll() is not None:
                    drain_eval(stack)
                if (
                    Path(actual["plan"]["run_root"]).exists()
                    or Path(actual["plan"]["test_root"]).exists()
                ):
                    raise FileExistsError("fresh fit cannot replace prior run")
                state["phase"] = "dual_gpu_fit"
                command = [
                    sys.executable,
                    "-c",
                    "import json,sys;from gradpert.execution.train_entry import execute_plan;"
                    "execute_plan(json.load(open(sys.argv[1])))",
                    str(path),
                ]
                active_fit = start(actual, "fit", command, "0,1", stack)
                if budget == 0.65 and active_eval is None and pending_evals:
                    start_eval(pending_evals.pop(0), stack)
                    state["phase"] = "dual_gpu_fit_with_capped_eval"
                    record()
                if finish(active_fit):
                    active_fit = None
                    raise RuntimeError(f"{row['name']} fit failed; no later fit dispatched")
                active_fit = None
                validate_training_stage(actual["plan"])
                pending_evals.append(actual)
            drain_eval(stack)
            while pending_evals:
                start_eval(pending_evals.pop(0), stack, uncapped=True)
                drain_eval(stack)
            state.update(status="complete", phase="complete")
            record()
            atomic_json(directory / "COMPLETE.json", state)
        except BaseException as error:
            if shared_policy is not None:
                for active in (active_fit, active_eval):
                    if active is not None:
                        terminate_tree(active[0].pid, active[2]["identity"])
            # Drain only the already-authorized evaluator, never launch peers.
            state.update(
                status="failed", phase="failed", error_type=type(error).__name__, error=str(error)
            )
            record()
            atomic_json(directory / "FAILURE.json", state)
            if active_eval is not None:
                with contextlib.suppress(BaseException):
                    drain_eval(stack)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--uncapped", action="store_true")
    parser.add_argument("--shared-policy", type=Path)
    args = parser.parse_args()
    if args.evaluate:
        evaluate_attempt(
            read_json(args.evaluate),
            args.attempt,
            0.25 if args.attempt == 1 and not args.uncapped else 1.0,
            gpu="0",
            full_first=args.uncapped,
        )
    elif args.queue:
        execute(read_json(args.queue), args.queue.parent)
    else:
        if not all((args.manifest, args.runtime, args.output)):
            parser.error("preparation requires manifest/runtime/output")
        prepare(
            Path(__file__).resolve().parents[2],
            args.manifest,
            args.runtime,
            args.output,
            read_json(args.shared_policy) if args.shared_policy else None,
        )


if __name__ == "__main__":
    main()
