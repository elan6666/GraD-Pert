"""Small interactive front door to the existing sealed R50 lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

DEFAULT_CONFIG = "configs/r50/batch1024/gradpert_b2/nadig_jurkat.yaml"
DEFAULT_RUNTIME = Path("/data/yilangliu/GraD-Pert/runtime/train.json")
SERVER_ROOT = Path("/data/yilangliu")


def repository_root() -> Path:
    """Resolve the executing package, not an unrelated current directory."""
    for root in Path(__file__).resolve().parents:
        if (root / "pyproject.toml").is_file() and (root / ".git").exists():
            return root
    raise ValueError("train requires an installed source checkout with .git")


def resolve_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Read and validate inputs without creating directories or importing CUDA."""
    from gradpert.execution.identity import inspect_source_identity

    root = repository_root()
    config_path = (args.config or root / DEFAULT_CONFIG).resolve(strict=True)
    config = load_experiment_config(config_path)
    if config.training.formal_run_policy != "r50_selection":
        raise ValueError("train currently supports the complete R50 best/last lifecycle only")
    if config.training.max_epochs.value != 50 or config.training.early_stopping:
        raise ValueError("train requires 50 epochs without early stopping")
    if config.artifacts.result_mode != "metrics_only":
        raise ValueError("train currently requires metrics_only results")
    seed = config.training.run_seeds[0] if args.seed is None else args.seed
    if seed not in config.training.run_seeds:
        raise ValueError(f"seed {seed} is not allowed by config: {config.training.run_seeds}")
    if not re.fullmatch(r"[0-9]+|GPU-[a-fA-F0-9-]+", args.gpu):
        raise ValueError("--gpu requires one physical GPU index or UUID")
    runtime_path = (
        args.runtime or Path(os.environ.get("GRADPERT_RUNTIME", str(DEFAULT_RUNTIME)))
    ).resolve(strict=True)
    runtime = json.loads(runtime_path.read_text())
    data = Path(args.data_root or runtime["data_root"]).resolve(strict=True)
    runs = Path(runtime["runs_root"]).resolve()
    if not data.is_relative_to(SERVER_ROOT) or not runs.is_relative_to(SERVER_ROOT):
        raise ValueError("data and results must stay under /data/yilangliu")
    publication = Path(runtime["publication_receipt"]).resolve(strict=True)
    publication_sha = runtime["publication_sha256"]
    if sha256_file(publication) != publication_sha:
        raise ValueError("publication receipt hash mismatch")
    identity = inspect_source_identity(
        root,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=publication,
        expected_publication_receipt_sha256=publication_sha,
    )
    prior = runtime.get("genept_receipt")
    prior_sha = runtime.get("genept_sha256")
    # The native runner additionally checks topology, gene order and coverage.
    # A supplied receipt must be valid even when this config does not need it.
    if bool(prior) != bool(prior_sha):
        raise ValueError("GenePT receipt and SHA256 must be configured together")
    if prior and sha256_file(Path(prior)) != prior_sha:
        raise ValueError("GenePT receipt hash mismatch")
    parameters = config.model.parameters
    artifact = parameters.get("genept_artifact_path")
    if artifact and str(artifact.value).endswith(".npz") and not prior:
        raise ValueError("GenePT configuration requires runtime genept_receipt and genept_sha256")
    token = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    run_id = f"{config.dataset_id}-seed{seed}-{token}"
    run_root = runs / run_id
    if run_root.exists():
        raise FileExistsError(run_root)
    return {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "resolved_config": config.model_dump(mode="json"),
        "runtime": str(runtime_path),
        "runtime_sha256": sha256_file(runtime_path),
        "source_commit": identity.commit,
        "repository_root": str(root),
        "data_root": str(data),
        "run_id": run_id,
        "run_root": str(run_root),
        "test_root": str(runs / (run_id + "-test")),
        "gpu": args.gpu,
        "seed": seed,
        "publication": str(publication),
        "publication_sha256": publication_sha,
        "genept_receipt": prior,
        "genept_sha256": prior_sha,
        "stages": ["fit", "per-epoch validation", "curves", "best/last test"],
        "report_all_validation_metrics": True,
    }


def execute_plan(plan: dict[str, Any]) -> None:
    """Use the same fit/postfit implementations as automated experiments."""
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    available = {value.strip() for line in devices.splitlines() for value in line.split(",")}
    if plan["gpu"] not in available:
        raise ValueError("requested physical GPU is not present")
    for key, digest in (("config", "config_sha256"), ("runtime", "runtime_sha256")):
        if sha256_file(Path(plan[key])) != plan[digest]:
            raise ValueError(f"{key} changed after planning")
    os.environ["CUDA_VISIBLE_DEVICES"] = plan["gpu"]
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["GRADPERT_SPARSE_UNION_IMPL"] = "cpu_array"
    from gradpert.data._io import atomic_json
    from gradpert.evaluation.state import prepare_evaluation_state
    from gradpert.execution.native import run_native_experiment
    from gradpert.execution.postfit import evaluate_best_last

    config = load_experiment_config(plan["config"])
    prepare_evaluation_state(
        dataset_id=config.dataset_id,
        protocol_id=config.data.protocol_id,
        data_root=plan["data_root"],
        validation_only=True,
    )
    root = Path(plan["run_root"])
    root.mkdir(parents=True, exist_ok=False)
    # Keep intent outside the native root, which must initially be empty.
    atomic_json(root / "launch.json", plan)
    fit_root = root / "fit"
    run_native_experiment(
        config_path=plan["config"],
        data_root=plan["data_root"],
        run_root=fit_root,
        run_id=plan["run_id"],
        run_seed=plan["seed"],
        mode="full",
        device_name="cuda:0",
        repository_root=plan["repository_root"],
        formal=True,
        source_publication_receipt=plan["publication"],
        source_publication_receipt_sha256=plan["publication_sha256"],
        genept_preflight_receipt=plan["genept_receipt"],
        genept_preflight_receipt_sha256=plan["genept_sha256"],
        report_all_validation_metrics=True,
    )
    evaluate_best_last(
        training_root=fit_root,
        output_root=Path(plan["test_root"]),
        data_root=Path(plan["data_root"]),
        repository_root=Path(plan["repository_root"]),
        publication=Path(plan["publication"]),
        publication_sha256=plan["publication_sha256"],
        device_name="cuda:0",
    )
    atomic_json(
        root / "COMPLETE.json", {"run_id": plan["run_id"], "source_commit": plan["source_commit"]}
    )


def train_entry(args: argparse.Namespace) -> int:
    plan = resolve_plan(args)
    print(json.dumps({"dry_run": args.dry_run, **plan}, indent=2, ensure_ascii=False))
    if not args.dry_run:
        execute_plan(plan)
    return 0
