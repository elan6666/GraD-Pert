#!/usr/bin/env python3
"""Launch one isolated official-package smoke with an explicit native adapter path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

DATASETS = (
    "replogle_k562_essential",
    "replogle_rpe1_essential",
    "nadig_jurkat",
    "nadig_hepg2",
    "norman",
)
RUNNERS = {
    "gears": ("benchmarks.gears.runner", "gears"),
    "txpert_public": ("benchmarks.txpert.runner", "txpert_public"),
}


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    project_root = args.project_root.resolve(strict=True)
    module, config_directory = RUNNERS[args.model]
    config = project_root / "configs" / "experiments" / config_directory / f"{args.dataset}.yaml"
    config.resolve(strict=True)
    destination = args.run_root.resolve() / args.model / args.dataset / "smoke-seed-1"
    command = [
        str(args.official_python.resolve(strict=True)),
        "-m",
        module,
        "--config",
        str(config),
        "--official-checkout",
        str(args.official_checkout.resolve(strict=True)),
        "--data-root",
        str(args.data_root.resolve(strict=True)),
        "--run-root",
        str(destination),
        "--run-id",
        f"{args.run_namespace}__{args.model}__{args.dataset}__smoke__seed1",
        "--device",
        args.device,
        "--repository-root",
        str(project_root),
    ]
    if args.model == "gears":
        if args.official_data_root is None:
            raise ValueError("GEARS requires --official-data-root")
        command.extend(["--official-data-root", str(args.official_data_root.resolve(strict=True))])
    if args.formal:
        command.append("--formal")
    else:
        if args.development_commit is None:
            raise ValueError("development launch requires --development-commit")
        command.extend(["--development-commit", args.development_commit])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(project_root / "src"), str(project_root)))
    return command, environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(RUNNERS), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--official-python", type=Path, required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--official-data-root", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-namespace", default="development-v2")
    parser.add_argument("--device", required=True)
    launch = parser.add_mutually_exclusive_group(required=True)
    launch.add_argument("--formal", action="store_true")
    launch.add_argument("--development-commit")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    command, environment = build_command(args)
    receipt = {
        "command": command,
        "cwd": str(args.project_root.resolve()),
        "pythonpath": environment["PYTHONPATH"],
        "execute": args.execute,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    return subprocess.run(command, cwd=args.project_root, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
