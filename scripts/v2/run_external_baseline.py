"""Plan or dispatch existing isolated R50 runners against a v2 dataset anchor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from gradpert.config import load_experiment_config
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("gears", "txpert_public", "scouter_genept_seed")


def command(args: argparse.Namespace, runtime: dict) -> list[str]:
    config = load_experiment_config(args.config)
    anchor = load_experiment_config(args.anchor_config)
    if config.model_id not in MODELS or anchor.model_id != "gradpert_v2":
        raise ValueError("requires an official baseline and a v2 anchor")
    if config.data != anchor.data or config.evaluation != anchor.evaluation:
        raise ValueError("baseline canonical data/evaluation contract differs from v2")
    if (
        config.training.formal_run_policy != "external_fixed_50"
        or config.training.max_epochs.value != 50
        or config.training.early_stopping
        or config.artifacts.result_mode != "metrics_only"
        or config.training.run_seeds != [1]
    ):
        raise ValueError("existing external runners require fixed50, metrics_only, seed1")
    if args.mode == "full" and args.smoke_run_root is None:
        raise ValueError("full execution requires the exact-source smoke run root")
    if args.mode == "step-smoke" and args.smoke_run_root is not None:
        raise ValueError("a fresh step probe must not reuse a smoke root")
    argv = [
        sys.executable,
        "-m",
        "gradpert",
        "benchmark",
        "--model",
        config.model_id,
        "--python",
        str(args.python),
        "--repository-root",
        str(ROOT),
        "--",
        "--config",
        str(args.config.resolve()),
        "--official-checkout",
        str(args.official_checkout),
        "--data-root",
        str(runtime["data_root"]),
        "--run-root",
        str(args.run_root),
        "--run-id",
        args.run_id,
        "--repository-root",
        str(ROOT),
        "--device",
        "cuda:0",
    ]
    if config.model_id == "scouter_genept_seed":
        if args.genept_seed is None or args.environment_lock is None:
            raise ValueError("Scouter requires explicit GenePT seed and environment lock")
        argv += [
            "--genept-seed",
            str(args.genept_seed),
            "--environment-lock",
            str(args.environment_lock),
            "--publication-receipt",
            str(runtime["publication_receipt"]),
            "--publication-receipt-sha256",
            runtime["publication_sha256"],
        ]
    else:
        argv += [
            "--formal",
            "--source-publication-receipt",
            str(runtime["publication_receipt"]),
            "--source-publication-receipt-sha256",
            runtime["publication_sha256"],
        ]
        if config.model_id == "gears":
            if args.official_data_root is None:
                raise ValueError("GEARS requires an explicit official data root")
            argv += ["--official-data-root", str(args.official_data_root)]
    if args.mode == "step-smoke":
        argv.append("--step-smoke")
    else:
        argv += ["--smoke-run-root", str(args.smoke_run_root)]
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "anchor-config", "runtime", "python", "official-checkout", "run-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("smoke-run-root", "official-data-root", "genept-seed", "environment-lock"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("step-smoke", "full"), default="step-smoke")
    parser.add_argument("--gpu", choices=("0", "1"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    runtime = json.loads(args.runtime.read_text())
    for path in (
        Path(runtime["data_root"]),
        args.run_root,
        args.official_checkout,
        args.official_data_root,
        args.smoke_run_root,
        args.genept_seed,
    ):
        if path is not None and not path.resolve().is_relative_to("/data/yilangliu"):
            raise ValueError("baseline data, checkouts and outputs must remain on the server")
    if args.run_root.exists():
        raise FileExistsError("baseline run root must be new; retain failed runs unchanged")
    if not args.python.is_file():
        raise FileNotFoundError("isolated runner Python does not exist")
    source = inspect_source_identity(
        ROOT,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=Path(runtime["publication_receipt"]),
        expected_publication_receipt_sha256=runtime["publication_sha256"],
    )
    argv = command(args, runtime)
    plan = {
        "kind": "external_baseline_dispatch_plan",
        "status": "planned_not_launched",
        "source": source.payload(),
        "config_sha256": sha256_file(args.config),
        "anchor_sha256": sha256_file(args.anchor_config),
        "runtime_sha256": sha256_file(args.runtime),
        "mode": args.mode,
        "gpu": args.gpu,
        "command": argv,
        "cwd": str(ROOT),
        "acceptance": (
            "Runner must validate exact config/source/environment/canonical hashes in smoke gate; "
            "full run retains and tests best/last with zero persistent PKL. "
            "A zero exit code alone is not a scientific completion receipt."
        ),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "1",
    }
    from run_group import lock

    locks = Path("/data/yilangliu/GraD-Pert/runtime")
    row_name = f"v2-row-{plan['config_sha256']}-{source.commit}-1.lock"
    with lock(locks / f"v2-gpu-{args.gpu}.lock") as gpu_lease, lock(locks / row_name) as row_lease:
        observed = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                args.gpu,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if int(observed.strip()) > 512:
            raise RuntimeError("baseline launch requires an idle GPU; preserve existing jobs")
        subprocess.run(argv, cwd=ROOT, env=environment, check=True, pass_fds=(gpu_lease, row_lease))


if __name__ == "__main__":
    main()
