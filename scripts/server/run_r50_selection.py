"""Run a fresh R50 coordinate: one validation-only integration, then fifty epochs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: argparse.Namespace, phase: str) -> list[str]:
    config = args.source / f"configs/r50/{args.row}/gradpert_b2/nadig_jurkat.yaml"
    return [
        sys.executable,
        str(args.source / "scripts/server/run_shared_gpu_native.py"),
        "--reserve-mib",
        "4096",
        *(
            ["--memory-fraction", str(args.memory_fraction)]
            if getattr(args, "memory_fraction", None) is not None
            else []
        ),
        "model",
        phase,
        "--config",
        str(config),
        "--repository-root",
        str(args.source),
        "--data-root",
        str(args.data_root),
        "--run-root",
        str(args.root / phase),
        "--run-id",
        f"{args.root.parent.name}/{args.row}/{phase}/seed-1",
        "--run-seed",
        "1",
        "--device",
        "cuda:0",
        "--formal",
        "--source-publication-receipt",
        str(args.publication),
        "--source-publication-receipt-sha256",
        args.publication_sha,
        "--source-publication-remote-ref",
        "refs/heads/main",
        "--genept-preflight-receipt",
        str(args.genept_receipt),
        "--genept-preflight-receipt-sha256",
        args.genept_sha,
        "--json",
    ]


def validate(root: Path, *, epochs: int, commit: str, config_sha: str) -> dict:
    small = root / "small_results"
    manifest = json.loads((small / "run_manifest.json").read_text())
    selection = json.loads((small / "selection_receipt.json").read_text())
    meta = json.loads((small / "run_meta.json").read_text())
    expected_steps = epochs * meta["steps_per_epoch"]
    if not (
        manifest["status"] == "trained"
        and manifest["test_evaluations"] == 0
        and manifest["source_commit"] == commit
        and not manifest["source_dirty"]
        and manifest["formal_eligible"]
        and manifest["config_sha256"] == config_sha
        and selection["status"] == "complete"
        and not selection["scientific_completion"]
        and selection["epochs_completed"] == epochs
        and selection["optimizer_steps"] == expected_steps
        and selection["test_evaluations"] == 0
        and selection["test_deferred"]
        and selection["control_manifest_scope"] == "validation"
    ):
        raise ValueError("R50 terminal identity/budget/test boundary mismatch")
    validations = sorted(small.glob("validation.epoch-*.json"))
    if len(validations) != epochs or [
        json.loads(p.read_text())["epoch"] for p in validations
    ] != list(range(epochs)):
        raise ValueError("R50 must validate exactly once per epoch")
    with (small / "train_steps.csv").open() as f:
        steps = list(csv.DictReader(f))
    if len(steps) != expected_steps or any(
        int(row["global_step"]) != i or int(row["epoch"]) != i // meta["steps_per_epoch"]
        for i, row in enumerate(steps)
    ):
        raise ValueError("R50 step order differs")
    checkpoints = list(root.rglob("*.pt"))
    retention_path = small / "checkpoint_retention.json"
    retention = json.loads(retention_path.read_text()) if retention_path.exists() else {}
    expected_names = {"best.pt"}
    if retention.get("policy") == "best_and_last_for_postfit_test":
        expected_names.add("last.pt")
        if sha(root / "checkpoints/last.pt") != retention["last_checkpoint_sha256"]:
            raise ValueError("R50 last checkpoint hash differs")
    if len(checkpoints) != len(expected_names) or {p.name for p in checkpoints} != expected_names:
        raise ValueError("R50 checkpoint retention differs")
    if sha(root / "checkpoints/best.pt") != manifest["best_checkpoint_sha256"]:
        raise ValueError("R50 checkpoint hash differs")
    if list(root.rglob("*.pkl")) or (root / "work").exists():
        raise ValueError("R50 requires zero persistent PKL and no evaluation work")
    for name in ("metrics_summary.json", "prediction_manifest.json", "evaluation_manifest.json"):
        if (small / name).exists():
            raise ValueError("R50 selection must not materialize test outputs")
    return {
        "status": "complete",
        "epochs": epochs,
        "run_manifest_sha256": sha(small / "run_manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row", choices=("ref", "lr_low", "lr_mid", "batch128", "batch512"), required=True
    )
    parser.add_argument("--memory-fraction", type=float)
    for flag in ("source", "data-root", "publication", "genept-receipt", "root"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("commit", "publication-sha", "genept-sha", "config-sha"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve(strict=True)
    args.root = args.root.resolve()
    config = args.source / f"configs/r50/{args.row}/gradpert_b2/nadig_jurkat.yaml"

    def identity() -> None:
        head = subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(args.source), "status", "--porcelain"], text=True
        ).strip()
        if head != args.commit or dirty:
            raise RuntimeError("R50 source identity changed")
        if (
            sha(config) != args.config_sha
            or sha(args.publication) != args.publication_sha
            or sha(args.genept_receipt) != args.genept_sha
        ):
            raise RuntimeError("R50 immutable input changed")

    identity()
    if args.root.exists():
        raise FileExistsError("R50 never overwrites or relaunches a run root")
    if not args.root.is_relative_to("/data/yilangliu/GraD-Pert/runs"):
        raise ValueError("R50 formal outputs belong under the server runs directory")
    if args.dry_run:
        print(
            json.dumps({"row": args.row, "commands": [command(args, p) for p in ("smoke", "full")]})
        )
        return
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise RuntimeError("R50 requires expandable_segments:True")
    args.root.mkdir(parents=True, exist_ok=False)
    phase_receipts = []
    for phase, epochs in (("smoke", 1), ("full", 50)):
        identity()
        print(f"START {args.row} {phase}", flush=True)
        with (args.root / f"{phase}.log").open("x") as log:
            result = subprocess.run(
                command(args, phase), cwd=args.source, stdout=log, stderr=subprocess.STDOUT
            )
        print(f"RC {args.row} {phase} {result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError("R50 failed; preserve evidence, never auto-relaunch")
        phase_receipts.append(
            validate(
                args.root / phase, epochs=epochs, commit=args.commit, config_sha=args.config_sha
            )
        )
        identity()
    with (args.root / "COMPLETE.json").open("x") as f:
        json.dump(
            {
                "row": args.row,
                "source_commit": args.commit,
                "phases": phase_receipts,
                "test_evaluations": 0,
            },
            f,
        )
    print(f"R50_COMPLETE {args.row}", flush=True)
    # Full training is sealed before test; the smoke remains validation-only.
    from gradpert.execution.postfit import evaluate_best_last

    evaluate_best_last(
        training_root=args.root / "full",
        output_root=args.root.parent.parent / (args.root.parent.name + "-test") / args.row,
        data_root=args.data_root,
        repository_root=args.source,
        publication=args.publication,
        publication_sha256=args.publication_sha,
        device_name="cuda:0",
        memory_fraction=args.memory_fraction,
    )
    print(f"R50_TEST_COMPLETE {args.row}", flush=True)


if __name__ == "__main__":
    main()
