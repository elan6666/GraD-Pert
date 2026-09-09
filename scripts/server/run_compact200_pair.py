"""Run one user-authorized compact B row: integration smoke, then fresh full fit."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", choices=("a0", "b0", "b1", "c1", "c2", "c3"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    config_name = (
        f"{args.row}_b0_capacity_step_batch1024_epoch200"
        if args.row.startswith("c")
        else f"{args.row}_compact128_step_batch1024_epoch200"
    )
    if args.row == "a0":
        config_name = "a0_step_batch1024_epoch100"
    config = source / f"configs/combinations/{config_name}" / "gradpert_b2/nadig_jurkat.yaml"

    def identity():
        assert (
            subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            == args.commit
        )
        assert not subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
        assert hashlib.sha256(args.publication.read_bytes()).hexdigest() == args.publication_sha

    identity()
    if args.root.exists():
        raise FileExistsError(args.root)
    args.root.mkdir(parents=True)
    assert os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True"
    for phase in ("smoke", "full"):
        identity()
        root = args.root / phase
        command = [
            sys.executable,
            str(source / "scripts/server/run_shared_gpu_native.py"),
            "--reserve-mib",
            "2048",
            "model",
            phase,
            "--config",
            str(config),
            "--repository-root",
            str(source),
            "--data-root",
            "/data/yilangliu/GraD-Pert/data-vnext-a942114",
            "--run-root",
            str(root),
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
            "--json",
        ]
        print(f"START {args.row} {phase}", flush=True)
        with (args.root / f"{phase}.log").open("x") as log:
            result = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT)
        print(f"RC {args.row} {phase} {result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"{phase} failed; preserve evidence, do not relaunch")
        manifest = json.loads((root / "small_results/run_manifest.json").read_text())
        assert manifest["status"] == "evaluated" and manifest["test_evaluations"] == 1
        assert manifest["source_commit"] == args.commit
        assert manifest["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
        assert manifest["formal_eligible"]
        assert not list(root.rglob("*.pkl"))
        receipt = json.loads((root / "small_results/performance_receipt.json").read_text())
        assert (
            receipt["epochs_completed"] == 1
            if phase == "smoke"
            else 1 <= receipt["epochs_completed"] <= (100 if args.row == "a0" else 200)
        )
        assert [p.name for p in root.rglob("*.pt")] == ["best.pt"]
        identity()
    (args.root / "COMPLETE.json").write_text(
        json.dumps({"row": args.row, "source_commit": args.commit, "status": "complete"})
    )


if __name__ == "__main__":
    main()
