"""One sealed anchor preflight or fit, reusing native execution/resource gates."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.generate_response_anchor import configuration
from scripts.v2.run_group import lock, validate_preflight
from scripts.v2.shared_gpu import (
    admit,
    descendant,
    parent_map,
    process_info,
    snapshot,
    terminate_tree,
)


def checked(binding: dict) -> Path:
    path = Path(binding["path"])
    if (
        not path.resolve(strict=True).is_relative_to("/data/yilangliu")
        or sha256_file(path) != binding["sha256"]
    ):
        raise ValueError("sealed server input changed")
    return path


def recipe(root: Path, arm: str) -> tuple[Path, Path, str]:
    """Explicit recipe selection; historical MSE keeps its original sealed paths."""
    families = {
        "R14-FRGA": ("response_anchor_20261028", "mse"),
        "R15-DIR": ("response_anchor_direction_20261029", "cosine_direction"),
    }
    if arm not in families:
        raise ValueError("unknown fixed response anchor arm")
    family, mode = families[arm]
    base = root / "configs/v2" / family
    return (
        base / "calibration/gradpert_v2/nadig_jurkat.yaml",
        base / "CALIBRATION_DECISION.json",
        mode,
    )


def validate_direction_diagnostics(diagnostic: dict) -> None:
    """Reject undefined frozen targets and unobserved/nonfinite readout summaries."""
    if (
        diagnostic.get("response_anchor_loss") != "cosine_direction"
        or diagnostic.get("eps") != 1e-6
        or diagnostic.get("target_count") != 1322
        or diagnostic.get("target_clamped_count") != 0
        or type(diagnostic.get("head_observation_count")) is not int
        or diagnostic["head_observation_count"] < 1
        or type(diagnostic.get("head_clamped_count")) is not int
        or not 0 <= diagnostic["head_clamped_count"] <= diagnostic["head_observation_count"]
        or any(
            not isinstance(diagnostic.get(key), (int, float))
            or isinstance(diagnostic.get(key), bool)
            or not math.isfinite(diagnostic[key])
            for key in ("target_norm_min", "target_norm_max", "head_norm_min", "head_norm_max")
        )
        or not 1e-6 < diagnostic["target_norm_min"] <= diagnostic["target_norm_max"]
        or not 0 <= diagnostic["head_norm_min"] <= diagnostic["head_norm_max"]
    ):
        raise ValueError("invalid direction objective norm audit")


def validate_calibration(config_path: Path, calibration: dict, arm: str = "R14-FRGA") -> None:
    root = Path(__file__).resolve().parents[2]
    probe_path, decision_path, mode = recipe(root, arm)
    decision = json.loads(decision_path.read_text())
    raw = yaml.safe_load(config_path.read_text())
    p = raw["model"]["parameters"]
    actual_mode = p.get("response_anchor_loss", {"value": "mse"})["value"]
    if actual_mode != mode:
        raise ValueError("execution arm differs from the configured objective")
    if mode == "cosine_direction":
        diagnostic = calibration.get("direction_diagnostics", {})
        validate_direction_diagnostics(diagnostic)
        if (
            decision.get("arm") != arm
            or decision.get("response_anchor_loss") != mode
            or calibration.get("response_anchor_loss") != mode
        ):
            raise ValueError("direction fit requires its own complete initialization calibration")
    if (
        not calibration.get("calibration_complete")
        or not calibration.get("state_restored")
        or calibration.get("forward_precision") != "bfloat16_autocast"
        or calibration.get("world_size") != 2
        or calibration.get("optimizer_updates") != 0
        or calibration.get("teacher_updates") != 0
        or calibration.get("center_updates") != 0
        or len(calibration.get("batches", [])) != 8
        or calibration.get("source", {}).get("commit") != decision["calibration_source"]
        or calibration.get("source", {}).get("dirty") is not False
        or calibration.get("config_sha256") != sha256_file(probe_path)
        or calibration.get("resolved_config_sha256")
        != sha256_json(load_experiment_config(probe_path).model_dump(mode="json"))
        or calibration.get("identity", {}).get("run_seed") != 1
        or calibration.get("bank_manifest_sha256") != p["response_anchor_sha256"]["value"]
        or calibration.get("bank_arrays_sha256") != p["response_anchor_arrays_sha256"]["value"]
        or calibration.get("selected_weight") != p["response_anchor_weight"]["value"]
        or raw
        != configuration(
            Path(__file__).resolve().parents[2],
            bank_path=p["response_anchor_path"]["value"],
            manifest_sha256=p["response_anchor_sha256"]["value"],
            arrays_sha256=p["response_anchor_arrays_sha256"]["value"],
            weight=calibration["selected_weight"],
            loss_mode=mode,
        )
    ):
        raise ValueError("fit config must equal the single calibrated RI+anchor recipe")


def execute(spec_path: Path, digest: str) -> None:
    spec_path = checked({"path": str(spec_path), "sha256": digest})
    spec = json.loads(spec_path.read_text())
    if spec["schema"] != "response-anchor-execution-v1" or spec["stage"] not in {
        "preflight",
        "fit",
    }:
        raise ValueError("unknown single-stage execution spec")
    root = Path(__file__).resolve().parents[2]
    config_path, pub, policy_path, calibration_path = (
        checked(spec[name]) for name in ("config", "publication", "policy", "calibration")
    )
    if not config_path.resolve().is_relative_to(root):
        raise ValueError("anchor configuration must belong to the executing clean source")
    config = load_experiment_config(config_path)
    source = inspect_source_identity(
        root,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=pub,
        expected_publication_receipt_sha256=spec["publication"]["sha256"],
    )
    if source.commit != spec["source_commit"] or spec["gpu"] != "0,1":
        raise ValueError("anchor source/world differs")
    calibration = json.loads(calibration_path.read_text())
    arm = spec.get("arm", "R14-FRGA")
    validate_calibration(config_path, calibration, arm)
    _, decision_path, mode = recipe(root, arm)
    decision = json.loads(decision_path.read_text())
    if spec["calibration"]["sha256"] != decision["calibration_complete_sha256"]:
        raise ValueError("execution requires the exact frozen calibration endpoint")
    if calibration["resource_policy_sha256"] != spec["policy"]["sha256"]:
        raise ValueError("resource policy differs from calibrated admission")
    policy = json.loads(policy_path.read_text())
    output = Path(spec["output"]).resolve()
    if (
        not output.is_relative_to("/data/yilangliu")
        or output.is_relative_to(root)
        or output.exists()
    ):
        raise ValueError("execution output must be a new independent server directory")
    output.mkdir(parents=True)
    child = None
    known = {}

    def guard() -> None:
        checked({"path": str(spec_path), "sha256": digest})
        names = ("config", "publication", "policy", "calibration") + (
            ("plan", "preflight") if spec["stage"] == "fit" else ()
        )
        for name in names:
            checked(spec[name])
        admit(policy, snapshot(os.getpid()))

    stack = contextlib.ExitStack()
    try:
        for gpu in ("0", "1"):
            stack.enter_context(
                lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{gpu}.lock")
            )
        guard()
        if spec["stage"] == "preflight":
            resource_queue = output / "RESOURCE_QUEUE.json"
            atomic_json(
                resource_queue,
                {
                    "schema": "unified-ddp-queue-1",
                    "source_commit": source.commit,
                    "shared_gpu_policy": policy,
                    "rows": [
                        {
                            "name": arm,
                            "plan": {
                                "config": str(config_path),
                                "config_sha256": spec["config"]["sha256"],
                                "data_root": spec["data_root"],
                                "publication": str(pub),
                                "publication_sha256": spec["publication"]["sha256"],
                            },
                        }
                    ],
                },
            )
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                "2",
                str(root / "scripts/v2/capacity_probe.py"),
                "--config",
                str(config_path),
                "--data-root",
                spec["data_root"],
                "--output",
                str(output / f"{arm}-preflight"),
                "--gpu",
                "0,1",
                "--publication",
                str(pub),
                "--publication-sha256",
                spec["publication"]["sha256"],
                "--steps",
                "10",
                "--shared-queue",
                str(resource_queue),
                "--resource-owner-pid",
                str(os.getpid()),
            ]
        else:
            plan_path = checked(spec["plan"])
            plan = json.loads(plan_path.read_text())
            if (
                plan["source_commit"] != source.commit
                or plan["config_sha256"] != spec["config"]["sha256"]
                or plan["postfit_policy"] != "deferred"
                or plan["gpu"] != "0,1"
                or plan["config"] != str(config_path)
                or plan["data_root"] != spec["data_root"]
                or plan["publication"] != str(pub)
                or plan["publication_sha256"] != spec["publication"]["sha256"]
                or plan.get("cuda_memory_fraction") != policy["max_own_memory_fraction"]
            ):
                raise ValueError("native fit plan differs from sealed calibrated preflight")
            validate_preflight(
                plan,
                {
                    "receipt": str(checked(spec["preflight"])),
                    "sha256": spec["preflight"]["sha256"],
                },
            )
            probe = json.loads(Path(spec["preflight"]["path"]).read_text())
            if not probe.get("response_anchor_gate", {}).get("passed"):
                raise ValueError("fit requires the native complete40/full-state anchor preflight")
            if mode == "cosine_direction":
                validate_direction_diagnostics(
                    probe["response_anchor_gate"].get("direction_diagnostics", {})
                )
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                "2",
                str(root / "scripts/v2/distributed_train.py"),
                "--plan",
                str(plan_path),
            ]
        env = {
            **os.environ,
            "PYTHONPATH": f"{root / 'src'}:{root / 'scripts/v2'}:{root}",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with (output / "child.log").open("xb") as log:
            child = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            identity = process_info(child.pid)
            if identity is None:
                raise RuntimeError("native child exited before identity capture")
            known[child.pid] = identity
            atomic_json(
                output / "RUNNING.json",
                {
                    "pid": child.pid,
                    "command": command,
                    "spec_sha256": digest,
                    "source_commit": source.commit,
                    "scientific_complete": False,
                },
            )
            while child.poll() is None:
                parents = parent_map()
                for pid in parents:
                    if (
                        descendant(pid, child.pid, parents)
                        and (info := process_info(pid)) is not None
                    ):
                        known.setdefault(pid, info)
                guard()
                time.sleep(5)
            code = child.wait()
        if code:
            raise RuntimeError(f"native {spec['stage']} failed with exit {code}")
        deadline = time.monotonic() + 30
        while any(process_info(pid) is not None for pid in known):
            guard()
            if time.monotonic() >= deadline:
                raise RuntimeError("owned native ranks remain after launcher exit")
            time.sleep(1)
        guard()
        current_source = inspect_source_identity(
            root,
            formal=True,
            expected_repository=config.source_code.repository,
            publication_receipt=pub,
            expected_publication_receipt_sha256=spec["publication"]["sha256"],
        )
        native = (
            (output / f"{arm}-preflight/receipt.json")
            if spec["stage"] == "preflight"
            else (Path(plan["run_root"]) / "TRAIN_COMPLETE.json")
        )
        if current_source != source or not native.is_file() or list(output.rglob("*.pkl")):
            raise ValueError("native source/receipt/zero-PKL closure failed")
        if spec["stage"] == "fit" and list(Path(plan["run_root"]).rglob("*.pkl")):
            raise ValueError("fit must preserve whole-run zero-PKL")
        atomic_json(
            output / "COMPLETE.json",
            {
                "stage": spec["stage"],
                "exit_code": code,
                "spec_sha256": digest,
                "source_commit": source.commit,
                "native_receipt": str(native),
                "native_receipt_sha256": sha256_file(native),
                "scientific_complete": False,
            },
        )
    except BaseException as error:
        cleanup_errors = []
        for pid, identity in reversed(list(known.items())):
            try:
                terminate_tree(pid, identity)
            except BaseException as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        if child is not None and child.poll() is None:
            try:
                child.terminate()
                child.wait(timeout=30)
            except BaseException as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        atomic_json(
            output / "FAILURE.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "cleanup_errors": cleanup_errors,
                "source_commit": source.commit,
                "spec_sha256": digest,
                "scientific_complete": False,
            },
        )
        raise
    finally:
        stack.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    execute(args.spec, args.spec_sha256)


if __name__ == "__main__":
    main()
