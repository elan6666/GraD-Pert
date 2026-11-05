"""Run one native dataset-transfer stage with immutable inputs and existing resource gates."""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.execution.train_entry import resolve_plan
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file
from scripts.v2.run_group import lock, validate_preflight
from scripts.v2.shared_gpu import admit, process_info, snapshot, terminate_tree

SOURCE = Path(__file__).resolve().parents[2]
DATASETS = ("replogle_k562_essential", "replogle_rpe1_essential", "nadig_hepg2", "norman")


def validate_family_preflight(plan: dict, entry: dict) -> None:
    """Require this family's exact ten-update and 300-control native probe."""
    validate_preflight(plan, entry)
    receipt = read_json(Path(entry["receipt"]))
    shape = receipt.get("inference_shape", [])
    if (
        receipt.get("kind") != "preflight_only"
        or receipt.get("steps_requested") != 10
        or receipt.get("steps_completed") != 10
        or receipt.get("inference_exercised") is not True
        or len(shape) != 2
        or shape[0] != 300
        or shape[1] < 1
    ):
        raise ValueError("family requires exact10 native updates and 300-control inference")


def validate_acceptance(plan: dict, ready_path: Path, acceptance: dict) -> None:
    """Bind an affirmative independent acceptance to the immutable launch inputs."""
    if (
        acceptance.get("approved") is not True
        or acceptance.get("preflight_ready_sha256") != sha256_file(ready_path)
        or acceptance.get("source_commit") != plan["source_commit"]
        or acceptance.get("config_sha256") != plan["config_sha256"]
        or acceptance.get("run_id") != plan["run_id"]
        or acceptance.get("optimizer_updates") != 300
    ):
        raise ValueError("independent main acceptance missing or changed")


def native_run(
    command: list[str],
    log: Path,
    policy: dict,
    *,
    admission_check: Callable[[], None] | None = None,
) -> None:
    """Recheck ownership/resource identity while the exact child is alive."""
    with log.open("x") as stream:
        process = subprocess.Popen(
            command, cwd=SOURCE, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT
        )
        identity = None
        try:
            identity = process_info(process.pid)
            if identity is None:
                # /proc/cmdline can be briefly empty across exec. Never wait for
                # an unidentified live job to finish without resource checks.
                for _ in range(20):
                    if process.poll() is not None:
                        raise RuntimeError("native child exited before identity capture")
                    if admission_check is None:
                        admit(policy, snapshot(os.getpid()))
                    else:
                        admission_check()
                    time.sleep(0.05)
                    identity = process_info(process.pid)
                    if identity is not None:
                        break
                if identity is None:
                    raise RuntimeError("native child identity unavailable after bounded startup")
            atomic_json(log.with_suffix(".process.json"), identity)
            while process.poll() is None:
                if admission_check is None:
                    admit(policy, snapshot(os.getpid()))
                else:
                    admission_check()
                time.sleep(15)
            if process.returncode:
                raise RuntimeError(f"native child exited {process.returncode}; inspect {log}")
        finally:
            if process.poll() is None:
                if identity is not None:
                    terminate_tree(process.pid, identity)
                else:
                    # This is the unreaped direct Popen child, never a PID from
                    # a historical receipt. Do not guess a descendant identity.
                    process.terminate()
                process.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--mode", choices=("preflight", "fit"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (
        not root.is_relative_to("/data/yilangliu")
        or os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
    ):
        raise ValueError("server root and explicit allocator required")
    runtime = read_json(args.runtime)
    config = (
        SOURCE / "configs/v2/dataset_transfer_300_20261102/gradpert_v2" / f"{args.dataset}.yaml"
    )
    cfg = load_experiment_config(config)
    if cfg.training.formal_run_policy != "v2_updates_300":
        raise ValueError("only the explicit 300-update family may execute")
    identity = inspect_source_identity(
        SOURCE,
        formal=True,
        expected_repository=cfg.source_code.repository,
        publication_receipt=runtime["publication_receipt"],
        expected_publication_receipt_sha256=runtime["publication_sha256"],
    )
    # Source publication was independently verified before this stage. Every
    # successor still verifies the exact same sealed config/runtime/publication.
    plan_path = root / args.dataset / "PLAN.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    if plan_path.exists():
        plan = read_json(plan_path)
    else:
        plan = resolve_plan(
            SimpleNamespace(
                config=config,
                runtime=args.runtime,
                data_root=Path(runtime["data_root"]),
                gpu="0,1",
                seed=1,
                defer_test=True,
            )
        )
        plan["cuda_memory_fraction"] = 0.90
        atomic_json(plan_path, plan)
    if (
        plan["source_commit"] != identity.commit
        or plan["config_sha256"] != sha256_file(config)
        or plan["runtime_sha256"] != sha256_file(args.runtime)
        or plan["gpu"] != "0,1"
    ):
        raise ValueError("prepared source/config/runtime changed")
    output = plan_path.parent
    controller = output / f"{args.mode}-controller.json"
    if controller.exists():
        raise FileExistsError(
            "stage identity already exists; inspect it instead of duplicate launch"
        )
    atomic_json(
        controller,
        {
            "pid": os.getpid(),
            "identity": process_info(os.getpid()),
            "mode": args.mode,
            "run_id": plan["run_id"],
            "source_commit": identity.commit,
            "config_sha256": plan["config_sha256"],
        },
    )
    policy = runtime["shared_gpu_policy"]
    with contextlib.ExitStack() as stack:
        for gpu in (0, 1):
            stack.enter_context(lock(Path(f"/data/yilangliu/GraD-Pert/runtime/v2-gpu-{gpu}.lock")))
        admit(policy, snapshot(os.getpid()))
        try:
            if args.mode == "preflight":
                preflight = output / "preflight"
                if preflight.exists():
                    raise FileExistsError(
                        "preflight output exists; never rerun under the same identity"
                    )
                native_run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node",
                        "2",
                        str(SOURCE / "scripts/v2/capacity_probe.py"),
                        "--config",
                        str(config),
                        "--data-root",
                        runtime["data_root"],
                        "--output",
                        str(preflight),
                        "--gpu",
                        "0,1",
                        "--publication",
                        runtime["publication_receipt"],
                        "--publication-sha256",
                        runtime["publication_sha256"],
                        "--steps",
                        "10",
                    ],
                    output / "preflight.log",
                    policy,
                )
                receipt = preflight / "receipt.json"
                entry = {"receipt": str(receipt), "sha256": sha256_file(receipt)}
                validate_family_preflight(plan, entry)
                atomic_json(
                    output / "PREFLIGHT_READY.json",
                    {
                        "source_commit": identity.commit,
                        "config_sha256": plan["config_sha256"],
                        "run_id": plan["run_id"],
                        "preflight": entry,
                        "formal_fit_started": False,
                    },
                )
            else:
                ready = read_json(output / "PREFLIGHT_READY.json")
                validate_family_preflight(plan, ready["preflight"])
                validate_acceptance(
                    plan,
                    output / "PREFLIGHT_READY.json",
                    read_json(output / "MAIN_ACCEPTANCE.json"),
                )
                native_run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node",
                        "2",
                        str(SOURCE / "scripts/v2/distributed_train.py"),
                        "--plan",
                        str(plan_path),
                    ],
                    output / "fit.log",
                    policy,
                )
                result = validate_training_stage(plan)
                if (
                    read_json(Path(plan["run_root"]) / "fit/history.json")[-1]["optimizer_steps"]
                    != 300
                ):
                    raise ValueError("fit did not complete exact300 updates")
                atomic_json(output / "FIT_COMPLETE.json", result)
            if list(root.rglob("*.pkl")) or list(Path(runtime["runs_root"]).rglob("*.pkl")):
                raise ValueError("successful family must contain zero PKL")
        except BaseException as error:
            atomic_json(
                output / f"{args.mode}-FAILURE.json",
                {
                    "type": type(error).__name__,
                    "error": str(error),
                    "source_commit": identity.commit,
                    "run_id": plan["run_id"],
                    "scientific_complete": False,
                },
            )
            raise


if __name__ == "__main__":
    main()
