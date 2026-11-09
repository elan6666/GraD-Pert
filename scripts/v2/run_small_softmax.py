"""Run a sealed two-epoch prediction-only arm through preflight and fit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE), str(SOURCE / "scripts/v2")]

from gradpert.data._io import atomic_json  # noqa: E402
from gradpert.execution.identity import inspect_source_identity  # noqa: E402
from gradpert.execution.v2_training_stage import validate_training_stage  # noqa: E402
from gradpert.hashing import sha256_file  # noqa: E402
from scripts.v2.run_dataset_transfer_300 import native_run  # noqa: E402
from scripts.v2.run_group import lock, validate_preflight  # noqa: E402
from scripts.v2.shared_gpu import process_info, snapshot  # noqa: E402


def admit_selected_idle(observed: dict, gpu: int, uuid: str) -> None:
    device = observed["gpus"].get(gpu)
    if device is None or device["uuid"] != uuid:
        raise RuntimeError("selected GPU identity changed")
    if device["free_mib"] < 2048:
        raise RuntimeError("selected GPU free-memory reserve exhausted")
    for process in observed["processes"]:
        if process["gpu_uuid"] == uuid and not process["owned"]:
            raise RuntimeError("selected GPU has foreign work; preserve it")
        if process["owned"] and process["gpu_uuid"] != uuid:
            raise RuntimeError("owned process escaped selected GPU")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if sha256_file(args.plan) != args.plan_sha256:
        raise ValueError("sealed plan changed")
    plan = json.loads(args.plan.read_text())
    root = args.plan.resolve().parent
    if not root.is_relative_to("/data/yilangliu/"):
        raise ValueError("server-only scientific execution")
    gpu = int(plan["gpu"])
    for key, value in (
        ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
        ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        ("CUDA_VISIBLE_DEVICES", str(gpu)),
    ):
        if os.environ.get(key) != value:
            raise ValueError("required execution environment differs: " + key)
    source = inspect_source_identity(
        SOURCE,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=plan["publication"],
        expected_publication_receipt_sha256=plan["publication_sha256"],
    )
    if (
        source.commit != plan["source_commit"]
        or sha256_file(Path(plan["config"])) != plan["config_sha256"]
    ):
        raise ValueError("plan source/config mismatch")
    config = plan["resolved_config"]
    params = config["model"]["parameters"]
    variant = params.get("architecture_variant", {}).get("value", "historical")
    prediction_only = variant == "small_softmax_mhc" or (
        variant == "historical" and params.get("distillation_enabled", {}).get("value") is False
    )
    if (
        not prediction_only
        or any(
            params[k]["value"]
            for k in (
                "lambda1",
                "lambda2",
                "ssl1_condition",
                "ssl1_node",
                "ssl1_spread",
                "ssl2_dino",
                "ssl2_ibot",
                "ssl2_koleo",
            )
        )
        or config["training"]["max_epochs"]["value"] != 2
        or plan["postfit_policy"] != "deferred"
    ):
        raise ValueError("single prediction-only two-epoch scope changed")
    if Path(plan["run_root"]).exists():
        raise FileExistsError("no duplicate run or implicit resume")
    marker = root / (args.stage + "-owner.json")
    if marker.exists():
        raise FileExistsError("stage already dispatched; reconcile its identity")
    owner = process_info(os.getpid())
    if owner is None:
        raise RuntimeError("controller identity unavailable")
    with lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{gpu}.lock"):

        def guard():
            state = snapshot(os.getpid())
            admit_selected_idle(state, gpu, args.gpu_uuid)
            atomic_json(root / "RESOURCE_CURRENT.json", state)

        guard()
        preflight = root / "preflight"
        if args.stage == "train":
            validate_preflight(
                plan,
                dict(
                    receipt=str(preflight / "receipt.json"),
                    sha256=sha256_file(preflight / "receipt.json"),
                ),
            )
        atomic_json(
            marker,
            dict(
                identity=owner,
                stage=args.stage,
                plan_sha256=args.plan_sha256,
                gpu_uuid=args.gpu_uuid,
            ),
        )
        try:
            if args.stage == "preflight":
                command = [
                    sys.executable,
                    str(SOURCE / "scripts/v2/capacity_probe.py"),
                    "--config",
                    plan["config"],
                    "--data-root",
                    plan["data_root"],
                    "--output",
                    str(preflight),
                    "--gpu",
                    str(gpu),
                    "--publication",
                    plan["publication"],
                    "--publication-sha256",
                    plan["publication_sha256"],
                    "--steps",
                    "10",
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node",
                    "1",
                    str(SOURCE / "scripts/v2/distributed_train.py"),
                    "--plan",
                    str(args.plan),
                ]
            native_run(command, root / (args.stage + ".log"), {}, admission_check=guard)
            if args.stage == "train":
                validate_training_stage(plan)
            else:
                validate_preflight(
                    plan,
                    dict(
                        receipt=str(preflight / "receipt.json"),
                        sha256=sha256_file(preflight / "receipt.json"),
                    ),
                )
            atomic_json(
                root / (args.stage + "-complete.json"),
                dict(
                    stage=args.stage,
                    plan_sha256=args.plan_sha256,
                    owner=owner,
                    scientific_complete=False,
                ),
            )
        except BaseException as error:
            atomic_json(
                root / (args.stage + "-failure.json"),
                dict(
                    stage=args.stage, error_type=type(error).__name__, error=str(error), owner=owner
                ),
            )
            raise


if __name__ == "__main__":
    main()
