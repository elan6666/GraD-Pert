"""Sealed first-batch queue: two independent GPUs, all ten preflights before fitting."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from collect_results import collect_run
from run_group import lock, validate_preflight

from gradpert.data._io import atomic_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.execution.train_entry import resolve_plan
from gradpert.hashing import sha256_file

ARMS = ("N0", "U24", "MR1", "P1", "C1", "O1", "VH", "S1-L4", "CG1", "S12-L4")
# Capacity evidence precedes formal work; put the expensive conditional/crop
# recipes first in each lane to reject an infeasible common batch promptly.
PREFLIGHT_PRIORITY = (8, 6, 2, 0, 4, 9, 7, 3, 1, 5)
SCHEMA = "gradpert-v2-unified-first-queue-1"
SERVER_ROOT = Path("/data/yilangliu")
LEASE_ROOT = SERVER_ROOT / "GraD-Pert/runtime"
ALLOCATOR = "expandable_segments:True"
METRICS = {
    f"{family}_{scope}"
    for family in ("txpert_macro_pearson_delta", "trishift_pearson_delta", "systema_pearson")
    for scope in ("all", "deg")
}


def _manifest(source: Path, path: Path) -> dict:
    from generate_unified_group import verify_manifest

    return verify_manifest(source, path)


def _server_path(path: Path) -> None:
    if not path.resolve().is_relative_to(SERVER_ROOT.resolve()):
        raise ValueError("scientific work and queue evidence stay under /data/yilangliu")


def parse_gpus(value: str) -> tuple[str, str]:
    devices = tuple(value.split(","))
    if len(devices) != 2 or set(devices) != {"0", "1"}:
        raise ValueError("first batch requires two independent physical GPU lanes 0 and 1")
    return devices  # type: ignore[return-value]


def prepare_queue(
    manifest: Path,
    runtime: Path,
    source: Path,
    gpus: str,
    *,
    resolver=None,
    verifier=None,
) -> dict:
    """Resolve fresh run IDs once; no scientific process or directory is started."""
    resolver = resolver or resolve_plan
    verifier = verifier or _manifest
    devices = parse_gpus(gpus)
    verified = verifier(source, manifest)
    if (
        verified.get("schema") != "unified-first-1"
        or [r["name"] for r in verified["rows"]] != list(ARMS)
        or verified.get("epochs") != 6
        or verified.get("validation") != "disabled"
        or verified.get("test_roles") != ["last"]
        or verified.get("world_size") != 1
        or verified.get("accumulation") != 1
        or verified.get("seed") != 1
    ):
        raise ValueError("manifest differs from the ten authorized fresh single-GPU arms")
    rows = []
    for index, row in enumerate(verified["rows"]):
        plan = resolver(
            argparse.Namespace(
                config=(source / row["config"]).resolve(),
                runtime=runtime,
                data_root=None,
                gpu=devices[index % 2],
                seed=1,
            )
        )
        if plan["config_sha256"] != row["sha256"]:
            raise ValueError("resolved plan differs from the manifest configuration")
        _validate_training_plan(plan)
        if Path(plan["run_root"]).exists() or Path(plan["test_root"]).exists():
            raise FileExistsError("fresh queue never replaces an existing run or test root")
        rows.append(
            {
                "index": index,
                "name": row["name"],
                "experiment_id": row["experiment_id"],
                "lane": devices[index % 2],
                "plan": plan,
            }
        )
    if len({r["plan"]["run_id"] for r in rows}) != len(rows):
        raise ValueError("queue run IDs must be distinct")
    if len({r["plan"]["source_commit"] for r in rows}) != 1:
        raise ValueError("all arms must use the same published source")
    return {
        "schema": SCHEMA,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "runtime": str(runtime.resolve()),
        "runtime_sha256": sha256_file(runtime),
        "source": str(source.resolve()),
        "source_commit": rows[0]["plan"]["source_commit"],
        "gpus": list(devices),
        "preflight_steps": 10,
        "preflight_priority": list(PREFLIGHT_PRIORITY),
        "rows": rows,
    }


def _validate_training_plan(plan: dict) -> None:
    config = plan["resolved_config"]
    training, params = config["training"], config["model"]["parameters"]
    if (
        config["model"]["model_id"] != "gradpert_v2"
        or params["world_size"]["value"] != 1
        or params["accumulation"]["value"] != 1
        or params["microbatch"]["value"] != training["train_batch_size"]["value"]
        or params["validation_mode"]["value"] != "disabled"
        or training["formal_run_policy"] != "v2_fixed_6"
        or training["max_epochs"]["value"] != 6
        or training["monitor"] != "none"
        or training["monitor_mode"] != "none"
        or training["run_seeds"] != [1]
        or plan["seed"] != 1
        or plan["gpu"] not in ("0", "1")
        or config.get("continuation") is not None
        or plan.get("postfit_policy", "inline") != "inline"
    ):
        raise ValueError("queue requires fresh six epochs, inline last test, world1/accum1")
    for key in ("data_root", "run_root", "test_root"):
        _server_path(Path(plan[key]))


def verify_seal(queue: dict, *, verifier=None, source_inspector=None) -> None:
    """Validate immutable files and current clean published source before every child."""
    verifier = verifier or _manifest
    source_inspector = source_inspector or inspect_source_identity
    if (
        queue.get("schema") != SCHEMA
        or queue.get("preflight_steps") != 10
        or queue.get("preflight_priority") != list(PREFLIGHT_PRIORITY)
    ):
        raise ValueError("unknown queue contract")
    devices = parse_gpus(",".join(queue["gpus"]))
    for field in ("manifest", "runtime"):
        if sha256_file(Path(queue[field])) != queue[field + "_sha256"]:
            raise ValueError(f"sealed {field} changed")
    verified = verifier(Path(queue["source"]), Path(queue["manifest"]))
    if [r["name"] for r in queue["rows"]] != list(ARMS):
        raise ValueError("queue arm order changed")
    if len(verified["rows"]) != len(queue["rows"]):
        raise ValueError("queue arm count differs from manifest")
    for index, (row, original) in enumerate(zip(queue["rows"], verified["rows"], strict=True)):
        plan = row["plan"]
        _validate_training_plan(plan)
        if (
            row["index"] != index
            or row["name"] != original["name"]
            or row["experiment_id"] != original["experiment_id"]
            or row["lane"] != devices[index % 2]
            or plan["gpu"] != row["lane"]
            or plan["source_commit"] != queue["source_commit"]
            or Path(plan["repository_root"]).resolve() != Path(queue["source"]).resolve()
            or Path(plan["config"]).resolve()
            != (Path(queue["source"]) / original["config"]).resolve()
            or plan["config_sha256"] != original["sha256"]
            or plan["runtime_sha256"] != queue["runtime_sha256"]
            or Path(plan["runtime"]).resolve() != Path(queue["runtime"]).resolve()
        ):
            raise ValueError("saved row identity differs from the sealed queue")
        for field in ("config", "publication"):
            if sha256_file(Path(plan[field])) != plan[field + "_sha256"]:
                raise ValueError(f"sealed {field} changed")
        if plan.get("genept_receipt") and (
            sha256_file(Path(plan["genept_receipt"])) != plan["genept_sha256"]
        ):
            raise ValueError("sealed GenePT receipt changed")
    plan = queue["rows"][0]["plan"]
    identity = source_inspector(
        queue["source"],
        formal=True,
        expected_repository=plan["resolved_config"]["source_code"]["repository"],
        publication_receipt=plan["publication"],
        expected_publication_receipt_sha256=plan["publication_sha256"],
    )
    if identity.commit != queue["source_commit"] or identity.dirty:
        raise ValueError("active source differs from the clean published queue source")


def validate_short_preflight(plan: dict, entry: dict) -> None:
    validate_preflight(plan, entry)
    receipt = json.loads(Path(entry["receipt"]).read_text())
    if (
        receipt["kind"] != "preflight_only"
        or receipt.get("steps_requested") != 10
        or receipt["steps_completed"] != 10
        or receipt.get("gpu") != plan["gpu"]
        or receipt.get("world_size") != 1
        or receipt["source"].get("publication_receipt_sha256") != plan["publication_sha256"]
        or any(
            receipt.get(key, False)
            for key in (
                "profile_last_update",
                "cpu_prefetch_diagnostic_only",
                "fused_sinkhorn_diagnostic_only",
                "save_no_grad_sinkhorn_diagnostic_only",
                "fused_gram_diagnostic_only",
                "sequence_checkpoint_disabled_diagnostic_only",
            )
        )
    ):
        raise ValueError("first batch requires exact ten-update single-GPU full-path preflight")


def validate_completed(plan: dict, *, collector=None) -> dict:
    """Require last-only epoch6, frozen identity, eighteen metrics and zero PKL."""
    collector = collector or collect_run
    root = Path(plan["run_root"])
    if (root / "FAILURE.json").exists() or not (root / "COMPLETE.json").is_file():
        raise ValueError("formal process has no successful terminal receipt")
    rows = collector(root)
    if (
        len(rows) != 1
        or rows[0]["role"] != "last"
        or rows[0]["status"] != "complete"
        or rows[0]["checkpoint_epoch"] != 6
        or rows[0]["training_sha"] != plan["source_commit"]
        or rows[0]["evaluation_sha"] != plan["source_commit"]
        or rows[0]["config_sha256"] != plan["config_sha256"]
        or json.loads((root / "launch.json").read_text()) != plan
        or json.loads((root / "resolved_config.json").read_text()) != plan["resolved_config"]
    ):
        raise ValueError("completed run differs from the saved last-only six-epoch plan")
    result = json.loads((root / "fit/last-test.json").read_text())["result"]
    groups = result.get("metric_gene_groups", {})
    if set(groups) != {"seen_expression", "unseen_expression"}:
        raise ValueError("last test lacks the two expression-exposure groups")
    for metrics in [result["metrics"], *(g["metrics"] for g in groups.values())]:
        if len(metrics) != 6 or {m["metric_id"] for m in metrics} != METRICS:
            raise ValueError("last test requires all six Pearson variants for every gene group")
        for metric in metrics:
            finite, total = metric["finite_condition_count"], metric["total_condition_count"]
            if type(finite) is not int or type(total) is not int or not 0 <= finite <= total:
                raise ValueError("metric effective condition counts are missing or invalid")
            value = metric["macro_mean"]
            if (finite == 0 and value is not None) or (
                finite > 0 and (value is None or not math.isfinite(value))
            ):
                raise ValueError("metric availability differs from effective condition counts")
    if any(root.rglob("*.pkl")) or any(Path(plan["test_root"]).rglob("*.pkl")):
        raise ValueError("successful queue row contains PKL artifacts")
    return {
        "run_id": plan["run_id"],
        "complete_sha256": sha256_file(root / "COMPLETE.json"),
        "last_test_sha256": sha256_file(root / "fit/last-test.json"),
        "metrics": result["metrics"],
        "metric_gene_groups": groups,
        "collected": rows[0],
    }


def gpu_idle(output: str, gpu: str) -> bool:
    usage = {}
    for line in output.splitlines():
        if line.strip():
            fields = line.split(",")
            if len(fields) != 2:
                return False
            try:
                usage[fields[0].strip()] = int(fields[1].strip())
            except ValueError:
                return False
    return gpu in usage and 0 <= usage[gpu] <= 512


def child_environment(source: str, gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    # Each lane is a new independent world, even if a controller was called from DDP.
    for key in (
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "GROUP_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(key, None)
    environment["PYTORCH_ALLOC_CONF"] = ALLOCATOR
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["GRADPERT_SPARSE_UNION_IMPL"] = "cpu_array"
    environment["PYTHONPATH"] = str(Path(source) / "src")
    return environment


def probe_command(row: dict, directory: Path) -> list[str]:
    plan = row["plan"]
    return [
        sys.executable,
        str(Path(plan["repository_root"]) / "scripts/v2/capacity_probe.py"),
        "--config",
        plan["config"],
        "--data-root",
        plan["data_root"],
        "--output",
        str(directory / (row["name"] + "-preflight")),
        "--gpu",
        row["lane"],
        "--publication",
        plan["publication"],
        "--publication-sha256",
        plan["publication_sha256"],
        "--steps",
        "10",
    ]


def formal_command(row: dict, directory: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        "import json,sys;from gradpert.execution.train_entry import execute_plan;"
        "execute_plan(json.load(open(sys.argv[1])))",
        str(directory / (row["name"] + ".launch-plan.json")),
    ]


class QueueState:
    """One atomic state file shared by the two threads; no unsynchronized JSON writes."""

    def __init__(self, directory: Path, queue: dict):
        self.directory = directory
        self.mutex = threading.RLock()
        self.values: dict[str, Any] = {
            "schema": SCHEMA,
            "pid": os.getpid(),
            "queue_sha256": sha256_file(directory / "queue.json"),
            "source_commit": queue["source_commit"],
            "status": "running",
            "phase": "starting",
            "lanes": {gpu: {"phase": "idle", "child_pid": None} for gpu in queue["gpus"]},
            "preflights": {},
            "completed": {},
            "failures": [],
        }

    def record(self, **values) -> None:
        with self.mutex:
            self.values.update(values, updated_unix=time.time())
            atomic_json(self.directory / "state.json", self.values)

    def lane(self, gpu: str, **values) -> None:
        with self.mutex:
            self.values["lanes"][gpu].update(values)
            self.record()

    def entry(self, group: str, name: str, value: dict) -> None:
        with self.mutex:
            self.values[group][name] = value
            self.record()

    def failed(self, gpu: str, row: dict, stage: str, error: BaseException) -> None:
        with self.mutex:
            self.values["failures"].append(
                {
                    "gpu": gpu,
                    "row": row["name"],
                    "stage": stage,
                    "run_id": row["plan"]["run_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            self.lane(gpu, phase="failed", error=str(error), child_pid=None)
            self.record(status="failed", phase="draining_existing_peer")
            atomic_json(self.directory / "FAILURE.json", self.values)


def run_child(
    row: dict, directory: Path, stage: str, state: QueueState, cancel: threading.Event
) -> bool:
    """Wait for capacity, lease one GPU and one row, then start exactly one process."""
    plan, gpu = row["plan"], row["lane"]
    state.lane(
        gpu, phase="waiting_for_idle_gpu", row=row["name"], run_id=plan["run_id"], child_pid=None
    )
    command = (
        probe_command(row, directory) if stage == "preflight" else formal_command(row, directory)
    )
    while not cancel.is_set():
        usage = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        state.lane(gpu, memory_snapshot=usage.strip())
        if gpu_idle(usage, gpu):
            break
        cancel.wait(60)
    if cancel.is_set():
        return False
    with contextlib.ExitStack() as stack:
        leases = [stack.enter_context(lock(LEASE_ROOT / f"v2-gpu-{gpu}.lock"))]
        leases.append(
            stack.enter_context(
                lock(
                    LEASE_ROOT
                    / f"v2-row-{plan['config_sha256']}-{plan['source_commit']}-{plan['seed']}.lock"
                )
            )
        )
        if cancel.is_set():
            return False
        saved_queue = json.loads((directory / "queue.json").read_text())
        if sha256_file(directory / "queue.json") != state.values["queue_sha256"]:
            raise ValueError("queue plan changed after controller admission")
        verify_seal(saved_queue)
        usage = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        if not gpu_idle(usage, gpu):
            raise RuntimeError("selected GPU changed after leasing; existing work preserved")
        if stage == "formal":
            if Path(plan["run_root"]).exists() or Path(plan["test_root"]).exists():
                raise FileExistsError("fresh queue cannot rerun, resume or replace an existing row")
            saved_plan = directory / (row["name"] + ".launch-plan.json")
            if json.loads(saved_plan.read_text()) != plan:
                raise ValueError("row launch plan changed after the queue was sealed")
            entries = _preflight_entries(saved_queue, directory)
            validate_short_preflight(plan, entries[row["name"]])
        elif (directory / (row["name"] + "-preflight")).exists():
            raise FileExistsError("preflight evidence cannot be overwritten or implicitly retried")
        log_path = directory / f"{row['name']}-{stage}.log"
        started = time.time()
        with log_path.open("x") as log:
            # Admission and failure cancellation share a mutex: once either lane
            # records a failure, its peer may finish but cannot launch another row.
            with state.mutex:
                if cancel.is_set():
                    return False
                child = subprocess.Popen(
                    command,
                    cwd=plan["repository_root"],
                    env=child_environment(plan["repository_root"], gpu),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=tuple(leases),
                )
                state.lane(
                    gpu, phase=stage, child_pid=child.pid, log=str(log_path), started_unix=started
                )
            code = child.wait()
        exit_receipt = {
            "exit_code": code,
            "wall_seconds": time.time() - started,
            "finished_unix": time.time(),
            "run_id": plan["run_id"],
            "row": row["name"],
            "stage": stage,
            "gpu": gpu,
        }
        atomic_json(directory / f"{row['name']}-{stage}.exit.json", exit_receipt)
        state.lane(gpu, child_pid=None, exit_code=code, wall_seconds=exit_receipt["wall_seconds"])
        if code:
            raise RuntimeError(f"{row['name']} {stage} exited {code}; explicit repair required")
    return True


def _preflight_entries(queue: dict, directory: Path) -> dict[str, dict]:
    sealed = json.loads((directory / "PREFLIGHT_COMPLETE.json").read_text())
    if sealed["queue_sha256"] != sha256_file(directory / "queue.json") or sealed[
        "preflight_index_sha256"
    ] != sha256_file(directory / "preflight-index.json"):
        raise ValueError("preflight stage seal changed")
    indexed = json.loads((directory / "preflight-index.json").read_text())
    if [r["name"] for r in indexed["rows"]] != list(ARMS):
        raise ValueError("all ten preflight entries are required before any formal training")
    result = {}
    for row, entry in zip(queue["rows"], indexed["rows"], strict=True):
        if entry["name"] != row["name"] or entry["gpu"] != row["lane"]:
            raise ValueError("preflight lane or arm identity changed")
        validate_short_preflight(row["plan"], entry["preflight"])
        result[row["name"]] = entry["preflight"]
    return result


def execute_queue(
    queue: dict,
    directory: Path,
    phase: str = "all",
    *,
    child_runner=None,
    seal_validator=None,
    preflight_validator=None,
    completion_validator=None,
) -> None:
    """Two lane workers; failures drain the active peer and forbid all further launches."""
    if phase not in ("preflight", "formal", "all"):
        raise ValueError("unknown queue phase")
    _server_path(directory)
    if os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR:
        raise ValueError("required allocator contract is missing")
    if (directory / "FAILURE.json").exists() or (directory / "COMPLETE.json").exists():
        raise ValueError("terminal queue is immutable; repair requires a new explicit queue")
    child_runner = child_runner or run_child
    seal_validator = seal_validator or verify_seal
    preflight_validator = preflight_validator or validate_short_preflight
    completion_validator = completion_validator or validate_completed
    if json.loads((directory / "queue.json").read_text()) != queue:
        raise ValueError("queue differs from its saved immutable plan")
    cancel = threading.Event()
    state = QueueState(directory, queue)
    failures: list[BaseException] = []
    with lock(directory / "queue.lock"):
        try:
            seal_validator(queue)
            if phase == "formal":
                entries = _preflight_entries(queue, directory)
                state.values["preflights"].update(entries)
                if any((directory / f"{r['name']}-formal.log").exists() for r in queue["rows"]):
                    raise ValueError("formal stage was already attempted; no implicit retries")
            elif (directory / "PREFLIGHT_COMPLETE.json").exists():
                raise ValueError("preflight stage already complete; use --phase formal")

            def lane_worker(gpu: str, stage: str) -> None:
                scheduled_rows = (
                    [queue["rows"][index] for index in PREFLIGHT_PRIORITY]
                    if stage == "preflight"
                    else queue["rows"]
                )
                for row in scheduled_rows:
                    if row["lane"] != gpu or cancel.is_set():
                        continue
                    try:
                        seal_validator(queue)
                        if not child_runner(row, directory, stage, state, cancel):
                            break
                        if stage == "preflight":
                            receipt = directory / (row["name"] + "-preflight/receipt.json")
                            entry = {"receipt": str(receipt), "sha256": sha256_file(receipt)}
                            preflight_validator(row["plan"], entry)
                            state.entry("preflights", row["name"], entry)
                        else:
                            state.entry("completed", row["name"], completion_validator(row["plan"]))
                        state.lane(gpu, phase="row_complete", row=row["name"], child_pid=None)
                    except BaseException as error:
                        with state.mutex:
                            cancel.set()
                            failures.append(error)
                            state.failed(gpu, row, stage, error)
                        break

            def run_stage(stage: str) -> None:
                state.record(phase=stage)
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(lane_worker, gpu, stage) for gpu in queue["gpus"]]
                    for future in futures:
                        future.result()  # Existing peers finish; no kill or next row after failure.
                if failures:
                    raise RuntimeError(
                        "queue stage failed; evidence retained, no automatic retry"
                    ) from failures[0]

            if phase in ("preflight", "all"):
                run_stage("preflight")
                index = {
                    "schema": SCHEMA,
                    "rows": [
                        {
                            "name": r["name"],
                            "gpu": r["lane"],
                            "preflight": state.values["preflights"][r["name"]],
                        }
                        for r in queue["rows"]
                    ],
                }
                atomic_json(directory / "preflight-index.json", index)
                atomic_json(
                    directory / "PREFLIGHT_COMPLETE.json",
                    {
                        "schema": SCHEMA,
                        "queue_sha256": sha256_file(directory / "queue.json"),
                        "preflight_index_sha256": sha256_file(directory / "preflight-index.json"),
                        "rows": len(index["rows"]),
                        "source_commit": queue["source_commit"],
                        "finished_unix": time.time(),
                    },
                )
                state.record(status="preflight_complete", phase="preflight_complete")
            if phase in ("formal", "all"):
                # Validation is repeated at the barrier: no first fit before the tenth gate.
                _preflight_entries(queue, directory)
                state.record(status="running")
                run_stage("formal")
                if set(state.values["completed"]) != set(ARMS):
                    raise ValueError("scientific completion requires all ten successful last tests")
                state.record(status="complete", phase="complete")
                atomic_json(directory / "COMPLETE.json", state.values)
        except BaseException as error:
            state.record(
                status="failed", phase="failed", error_type=type(error).__name__, error=str(error)
            )
            atomic_json(directory / "FAILURE.json", state.values)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--phase", choices=("preflight", "formal", "all"), default="all")
    parser.add_argument(
        "--execute", action="store_true", help="default only seals and prints plans"
    )
    args = parser.parse_args()
    _server_path(args.queue_root)
    if os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR:
        parser.error("launch requires PYTORCH_ALLOC_CONF=expandable_segments:True")
    source = Path(__file__).resolve().parents[2]
    if args.phase == "formal":
        queue = json.loads((args.queue_root / "queue.json").read_text())
        if (
            Path(queue["manifest"]).resolve() != args.manifest.resolve()
            or Path(queue["runtime"]).resolve() != args.runtime.resolve()
            or queue["gpus"] != list(parse_gpus(args.gpus))
        ):
            parser.error("formal phase must reuse the exact saved manifest/runtime/GPU lanes")
        verify_seal(queue)
        _preflight_entries(queue, args.queue_root)
    else:
        queue = prepare_queue(args.manifest, args.runtime, source, args.gpus)
        args.queue_root.mkdir(parents=True, exist_ok=False)
        atomic_json(args.queue_root / "queue.json", queue)
        for row in queue["rows"]:
            atomic_json(args.queue_root / f"{row['name']}.launch-plan.json", row["plan"])
    if args.execute:
        execute_queue(queue, args.queue_root, args.phase)
    else:
        print(
            json.dumps(
                {
                    "status": "planned_not_launched",
                    "queue_root": str(args.queue_root),
                    "phase": args.phase,
                    "rows": len(queue["rows"]),
                    "lanes": {
                        gpu: [r["name"] for r in queue["rows"] if r["lane"] == gpu]
                        for gpu in queue["gpus"]
                    },
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
