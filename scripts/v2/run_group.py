"""Sealed sequential group queue; separate processes release CUDA between rows."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import subprocess
import sys
from pathlib import Path

from collect_results import collect_run
from generate_group import verify_group
from prepare_followup import validate_dependency

from gradpert.data._io import atomic_json
from gradpert.execution.train_entry import resolve_plan
from gradpert.hashing import sha256_file


def validate_preflight(plan: dict, entry: dict) -> None:
    path = Path(entry["receipt"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("preflight receipt checksum mismatch")
    receipt = json.loads(path.read_text())
    kind = receipt.get("kind")
    if kind not in ("integration_only", "capacity_only") or receipt.get("status") != "passed":
        raise ValueError("preflight must be a passed integration or capacity probe")
    minimum = 128 if kind == "capacity_only" else 1
    if receipt.get("steps_completed", 0) < minimum or not receipt.get("resume_checkpoint_sha256"):
        raise ValueError("preflight update/checkpoint evidence missing")
    if kind == "integration_only" and receipt["steps_completed"] != 1:
        raise ValueError("integration preflight must contain exactly one update")
    source = receipt["source"]
    if (
        source["dirty"]
        or source["commit"] != plan["source_commit"]
        or source["published_commit"] != plan["source_commit"]
    ):
        raise ValueError("preflight source differs from the clean published launch source")
    if (
        receipt["config_sha256"] != plan["config_sha256"]
        or receipt["data_root"] != plan["data_root"]
        or receipt["data"]["run_seed"] != plan["seed"]
        or receipt.get("world_size", 1) != len(plan["gpu"].split(","))
    ):
        raise ValueError("preflight config, data, seed or topology differs from launch")
    if receipt.get("peak_allocated_bytes", 0) <= 0 or not math.isfinite(
        receipt.get("last_terms", {}).get("gradient_norm", float("nan"))
    ):
        raise ValueError("preflight memory or finite-gradient evidence missing")


def prepare_queue(
    manifest: Path,
    parent: Path,
    runtime: Path,
    gpu: str,
    preflight_index: Path,
    seed: int | None,
    *,
    resolver=resolve_plan,
) -> dict:
    verified = verify_group(manifest, parent)
    validate_dependency(manifest, parent)
    entries = json.loads(preflight_index.read_text())["rows"]
    indexed = {(r["config_sha256"], r["seed"]): r for r in entries}
    if len(indexed) != len(entries):
        raise ValueError("duplicate preflight entries")
    items = []
    for row in verified["rows"]:
        plan = resolver(
            argparse.Namespace(
                config=Path(row["config"]), runtime=runtime, data_root=None, gpu=gpu, seed=seed
            )
        )
        entry = indexed.get((plan["config_sha256"], plan["seed"]))
        if entry is None:
            raise ValueError(f"missing exact config/seed preflight: {row['name']}")
        validate_preflight(plan, entry)
        items.append({"name": row["name"], "plan": plan, "preflight": entry})
    return {
        "schema_version": "gradpert-v2-group-queue-1",
        "group": verified["group"],
        "manifest": str(manifest.resolve()),
        "manifest_sha256": verified["manifest_sha256"],
        "parent": str(parent.resolve()),
        "parent_sha256": sha256_file(parent),
        "preflight_index_sha256": sha256_file(preflight_index),
        "items": items,
    }


def next_action(plan: dict) -> str:
    root = Path(plan["run_root"])
    if not root.exists():
        return "launch"
    launch = root / "launch.json"
    if not launch.is_file() or json.loads(launch.read_text()) != plan:
        raise ValueError("existing run root is not owned by the saved launch plan")
    if (root / "COMPLETE.json").exists():
        rows = collect_run(root)
        if any(
            r["status"] != "complete"
            or r["config_sha256"] != plan["config_sha256"]
            or r["training_sha"] != plan["source_commit"]
            for r in rows
        ):
            raise ValueError("completion does not match the saved queue plan")
        return "skip_complete"
    if not (root / "fit/epoch_state.json").is_file():
        raise ValueError("startup failed before an epoch commit; repair explicitly before retry")
    return "resume"


@contextlib.contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process owns lease {path}") from error
        yield handle.fileno()


def execute_queue(queue: dict, directory: Path) -> None:
    if not directory.resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("experiment queues and scientific work stay on the server")
    if (
        sha256_file(Path(queue["manifest"])) != queue["manifest_sha256"]
        or sha256_file(Path(queue["parent"])) != queue["parent_sha256"]
    ):
        raise ValueError("group manifest or parent changed after queue sealing")
    with lock(directory / "queue.lock") as queue_fd:
        for index, item in enumerate(queue["items"]):
            plan = item["plan"]
            validate_preflight(plan, item["preflight"])
            for field in ("config", "runtime"):
                if sha256_file(Path(plan[field])) != plan[field + "_sha256"]:
                    raise ValueError(f"sealed {field} changed before queue execution")
            action = next_action(plan)
            if action == "skip_complete":
                continue
            devices = plan["gpu"].split(",")
            if not set(devices) <= {"0", "1"}:
                raise ValueError("group runner expects the two named physical GPU indices")
            with contextlib.ExitStack() as stack:
                leases = [
                    stack.enter_context(
                        lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{gpu}.lock")
                    )
                    for gpu in sorted(devices)
                ]
                lease_name = (
                    f"v2-row-{plan['config_sha256']}-{plan['source_commit']}-{plan['seed']}.lock"
                )
                row_lease = stack.enter_context(
                    lock(Path("/data/yilangliu/GraD-Pert/runtime") / lease_name)
                )
                leases.append(row_lease)
                usage = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                occupied = {
                    r.split(",")[0].strip(): int(r.split(",")[1]) for r in usage.splitlines()
                }
                if any(occupied[gpu] > 512 for gpu in devices):
                    raise RuntimeError(
                        "group launch requires idle selected GPUs; existing work preserved"
                    )
                plan_path = directory / f"{index:03d}-{item['name']}.launch-plan.json"
                if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
                    raise ValueError("persisted row launch plan changed")
                atomic_json(plan_path, plan)
                if action == "resume":
                    command = [
                        sys.executable,
                        str(Path(__file__).with_name("resume.py")),
                        "--launch",
                        str(Path(plan["run_root"]) / "launch.json"),
                    ]
                else:
                    command = [
                        sys.executable,
                        "-c",
                        "import json,sys; from gradpert.execution.train_entry import execute_plan; "
                        "execute_plan(json.load(open(sys.argv[1])))",
                        str(plan_path),
                    ]
                with (directory / f"{index:03d}-{item['name']}.log").open("a") as log:
                    subprocess.run(
                        command,
                        cwd=plan["repository_root"],
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        pass_fds=(queue_fd, *leases),
                    )
                if next_action(plan) != "skip_complete":
                    raise RuntimeError("row process exited without verified best/last completion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preflight-index", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume:
        queue = json.loads((args.queue_root / "queue.json").read_text())
        verified = verify_group(args.manifest, args.parent)
        validate_dependency(args.manifest, args.parent)
        if (
            verified["manifest_sha256"] != queue["manifest_sha256"]
            or sha256_file(args.preflight_index) != queue["preflight_index_sha256"]
            or any(p["plan"]["gpu"] != args.gpu for p in queue["items"])
            or any(
                Path(p["plan"]["runtime"]).resolve() != args.runtime.resolve()
                for p in queue["items"]
            )
        ):
            raise ValueError("resume arguments differ from the sealed queue")
        if [i["name"] for i in queue["items"]] != [r["name"] for r in verified["rows"]]:
            raise ValueError("saved queue rows differ from the verified group")
        for item, row in zip(queue["items"], verified["rows"], strict=True):
            if (
                item["plan"]["config_sha256"] != row["sha256"]
                or Path(item["plan"]["config"]).resolve() != Path(row["config"]).resolve()
            ):
                raise ValueError("saved queue config differs from the verified group")
        if args.seed is not None and any(p["plan"]["seed"] != args.seed for p in queue["items"]):
            raise ValueError("resume seed differs from the sealed queue")
    else:
        queue = prepare_queue(
            args.manifest, args.parent, args.runtime, args.gpu, args.preflight_index, args.seed
        )
    if not args.execute:
        print(json.dumps(queue, indent=2))
        return
    if not args.queue_root.resolve().is_relative_to("/data/yilangliu"):
        parser.error("queue must stay on the server")
    if not args.resume:
        args.queue_root.mkdir(parents=True, exist_ok=False)
        atomic_json(args.queue_root / "queue.json", queue)
    execute_queue(queue, args.queue_root)


if __name__ == "__main__":
    main()
