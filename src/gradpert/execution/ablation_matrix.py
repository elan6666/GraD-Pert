"""Hash-pinned orchestration for the Nadig Jurkat B2-vNext config matrix.

This module never implements training.  Every executable row is delegated to
the one native ``gradpert model pilot`` entrypoint after the complete matrix,
source, config, and GenePT preflight contracts have been validated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file
from gradpert.pilots import GenePTAvailabilityReceipt


@dataclass(frozen=True)
class AblationMatrixRow:
    variant_id: str
    config_path: Path
    config_sha256: str
    run_seed: int
    genept_preflight_required: bool


@dataclass(frozen=True)
class AblationLaunchPlanRow:
    variant_id: str
    config_path: str
    config_sha256: str
    run_root: str
    run_id: str
    run_seed: int
    device: str
    disposition: Literal["run", "skip_genept_missing_target"]


def _require_git_source(repository_root: Path, expected_commit: str) -> None:
    if len(expected_commit) != 40:
        raise ValueError("expected source commit must contain 40 hexadecimal characters")
    try:
        int(expected_commit, 16)
    except ValueError as error:
        raise ValueError("expected source commit must contain 40 hexadecimal characters") from error
    observed = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != expected_commit:
        raise ValueError("ablation launcher source commit differs from the frozen contract")
    dirty = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("ablation launcher requires a clean source worktree")


def load_ablation_matrix(
    matrix_path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[AblationMatrixRow, ...]:
    """Validate the frozen one-dataset/one-split/one-seed/10-epoch matrix."""

    matrix_file = Path(matrix_path).resolve(strict=True)
    root = Path(repository_root).resolve(strict=True)
    payload = json.loads(matrix_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ablation matrix must be a JSON object")
    if (
        payload.get("schema_version") != "1"
        or payload.get("dataset_id") != "nadig_jurkat"
        or payload.get("canonical_split_count") != 1
        or payload.get("run_seeds") != [1]
        or payload.get("max_epochs") != 10
    ):
        raise ValueError("ablation matrix experiment identity differs from the frozen design")
    raw_rows = payload.get("rows")
    if (
        not isinstance(raw_rows, list)
        or payload.get("row_count") != len(raw_rows)
        or len(raw_rows) != 22
    ):
        raise ValueError("ablation matrix must contain exactly 22 declared rows")

    rows: list[AblationMatrixRow] = []
    seen: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("every ablation matrix row must be a JSON object")
        variant_id = raw.get("variant_id")
        relative_config = raw.get("config_path")
        expected_hash = raw.get("config_sha256")
        requires_genept = raw.get("genept_preflight_required")
        if (
            not isinstance(variant_id, str)
            or not variant_id
            or variant_id in seen
            or not isinstance(relative_config, str)
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not isinstance(requires_genept, bool)
        ):
            raise ValueError("ablation matrix row identity is malformed or duplicated")
        config_path = (root / relative_config).resolve(strict=True)
        if not config_path.is_relative_to(root) or config_path.is_symlink():
            raise ValueError("ablation config must be a regular file inside the repository")
        if sha256_file(config_path) != expected_hash:
            raise ValueError(f"ablation config hash mismatch: {variant_id}")
        if (
            raw.get("dataset_id") != "nadig_jurkat"
            or raw.get("split_policy") != "frozen_canonical"
            or raw.get("run_seed") != 1
            or raw.get("max_epochs") != 10
            or raw.get("result_mode") != "metrics_only"
        ):
            raise ValueError(f"ablation row execution contract differs: {variant_id}")
        config = load_experiment_config(config_path)
        if (
            config.model.model_id != "gradpert_b2"
            or config.dataset_id != "nadig_jurkat"
            or config.training.max_epochs.value != 10
            or config.training.run_seeds != [1]
            or config.training.early_stopping
            or config.artifacts.result_mode != "metrics_only"
        ):
            raise ValueError(f"resolved config differs from matrix row: {variant_id}")
        seen.add(variant_id)
        rows.append(
            AblationMatrixRow(
                variant_id=variant_id,
                config_path=config_path,
                config_sha256=expected_hash,
                run_seed=1,
                genept_preflight_required=requires_genept,
            )
        )
    return tuple(rows)


def build_ablation_launch_plan(
    rows: Sequence[AblationMatrixRow],
    *,
    selected_variants: Sequence[str],
    runs_root: str | Path,
    device: str,
    genept_availability_receipt: str | Path | None,
) -> tuple[AblationLaunchPlanRow, ...]:
    """Resolve selected rows and fail closed before any training process starts."""

    if not device.startswith("cuda:"):
        raise ValueError("formal ablation execution requires an explicit cuda:N device")
    by_id = {row.variant_id: row for row in rows}
    selected = tuple(selected_variants) if selected_variants else tuple(by_id)
    if len(set(selected)) != len(selected):
        raise ValueError("selected ablation variants must be unique")
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise ValueError(f"unknown ablation variants: {unknown}")

    needs_genept = any(by_id[variant].genept_preflight_required for variant in selected)
    genept_status: str | None = None
    if needs_genept:
        if genept_availability_receipt is None:
            raise ValueError("GenePT variants require a sealed availability receipt")
        receipt = GenePTAvailabilityReceipt.model_validate_json(
            Path(genept_availability_receipt).resolve(strict=True).read_text(encoding="utf-8")
        )
        genept_status = receipt.status

    root = Path(runs_root).resolve()
    plan: list[AblationLaunchPlanRow] = []
    for variant_id in selected:
        row = by_id[variant_id]
        run_root = root / variant_id / "gradpert_b2" / "nadig_jurkat" / "seed-1"
        if run_root.exists() and any(run_root.iterdir()):
            raise ValueError(f"refusing to overwrite existing ablation run: {run_root}")
        disposition: Literal["run", "skip_genept_missing_target"] = "run"
        if row.genept_preflight_required and genept_status != "available":
            disposition = "skip_genept_missing_target"
        plan.append(
            AblationLaunchPlanRow(
                variant_id=variant_id,
                config_path=str(row.config_path),
                config_sha256=row.config_sha256,
                run_root=str(run_root),
                run_id=f"ablation/nadig_jurkat/{variant_id}/seed-1",
                run_seed=row.run_seed,
                device=device,
                disposition=disposition,
            )
        )
    return tuple(plan)


def _command(
    row: AblationLaunchPlanRow,
    *,
    python_executable: str,
    data_root: Path,
    repository_root: Path,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "gradpert",
        "model",
        "pilot",
        "--config",
        row.config_path,
        "--data-root",
        str(data_root),
        "--run-root",
        row.run_root,
        "--run-id",
        row.run_id,
        "--run-seed",
        str(row.run_seed),
        "--device",
        row.device,
        "--repository-root",
        str(repository_root),
        "--formal",
        "--json",
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--genept-availability-receipt", type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve(strict=True)
    rows = load_ablation_matrix(args.matrix, repository_root=repository_root)
    plan = build_ablation_launch_plan(
        rows,
        selected_variants=args.variant,
        runs_root=args.runs_root,
        device=args.device,
        genept_availability_receipt=args.genept_availability_receipt,
    )
    plan_payload: dict[str, Any] = {
        "schema_version": "nadig-vnext-ablation-launch-plan-v1",
        "matrix_path": str(args.matrix.resolve(strict=True)),
        "matrix_sha256": sha256_file(args.matrix.resolve(strict=True)),
        "expected_source_commit": args.expected_source_commit,
        "row_count": len(plan),
        "rows": [asdict(row) for row in plan],
    }
    if args.dry_run:
        print(json.dumps(plan_payload, sort_keys=True, separators=(",", ":")))
        return 0

    _require_git_source(repository_root, args.expected_source_commit)
    receipt_root = args.receipt_root.resolve()
    receipt_root.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_root / "launch-plan.json", plan_payload)
    environment = os.environ.copy()
    environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    for row in plan:
        started = time.time()
        if row.disposition == "skip_genept_missing_target":
            atomic_json(
                receipt_root / f"{row.variant_id}.json",
                {
                    "schema_version": "nadig-vnext-ablation-row-v1",
                    **asdict(row),
                    "status": "skipped_before_model_construction",
                    "returncode": None,
                    "started_unix": started,
                    "completed_unix": time.time(),
                },
            )
            continue
        command = _command(
            row,
            python_executable=args.python_executable,
            data_root=args.data_root.resolve(strict=True),
            repository_root=repository_root,
        )
        result = subprocess.run(command, env=environment, check=False)
        atomic_json(
            receipt_root / f"{row.variant_id}.json",
            {
                "schema_version": "nadig-vnext-ablation-row-v1",
                **asdict(row),
                "status": "complete" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "command": command,
                "started_unix": started,
                "completed_unix": time.time(),
            },
        )
        if result.returncode != 0:
            return result.returncode
    return 0


__all__ = [
    "AblationLaunchPlanRow",
    "AblationMatrixRow",
    "build_ablation_launch_plan",
    "load_ablation_matrix",
    "main",
]
