"""Bind installed upstream environments and persisted inference evidence."""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file, sha256_json


def verify_evaluation_files(config: dict, data_root: Path) -> dict:
    """Fail fast on changed frozen evaluation manifests; do not read test matrices."""
    root = Path(data_root) / config["data"]["dataset_id"] / config["data"]["protocol_id"]
    actual = {
        "control_manifest_sha256": sha256_file(root / "manifests/evaluation_controls.test.json"),
        "evaluation_state_sha256": sha256_file(root / "evaluation_v2/state_manifest.json"),
    }
    if actual != config["evaluation"]["expected_hashes"]:
        raise ValueError("frozen BR evaluation files differ before fit")
    return actual


def verify_cpu_assets(runtime: dict) -> dict:
    """Verify actual diagnostic identity assets before any pickle is deserialized."""
    prior_sha = sha256_file(runtime["prior_path"])
    if prior_sha != runtime["prior_sha256"]:
        raise ValueError("CPU integration official prior identity differs")
    graph_hashes = runtime["gears_asset_sha256"]
    if set(graph_hashes) != {"gene2go_all.pkl", "essential_all_data_pert_genes.pkl"}:
        raise ValueError("CPU integration requires both frozen official graph assets")
    actual = {
        filename: sha256_file(Path(runtime["gears_assets"]) / filename) for filename in graph_hashes
    }
    if actual != graph_hashes:
        raise ValueError("CPU integration official graph asset identity differs")
    return {"prior_sha256": prior_sha, "gears_asset_sha256": actual}


def verify_environment(runtime: dict) -> dict:
    """Check the sealed freeze, environment receipt and every declared distribution.

    Hashing a lock file alone does not show the interpreter is using that lock.
    Each receipt entry is therefore checked against this process's actual installed
    distribution metadata, before importing a model or accessing scientific data.
    """
    lock_path = Path(runtime["environment_lock"])
    receipt_path = Path(runtime["environment_receipt"])
    lock_sha = sha256_file(lock_path)
    receipt_sha = sha256_file(receipt_path)
    if lock_sha != runtime["environment_lock_sha256"]:
        raise ValueError("environment lock seal differs")
    if receipt_sha != runtime["environment_receipt_sha256"]:
        raise ValueError("environment receipt seal differs")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("freeze_sha256") != lock_sha:
        raise ValueError("environment receipt does not bind the sealed freeze")
    expected = receipt.get("distribution_versions")
    if (
        not isinstance(expected, dict)
        or not expected
        or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in expected.items()
        )
    ):
        raise ValueError("environment receipt must declare installed distribution versions")
    observed = {}
    for name, version in expected.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise ValueError(f"environment distribution is missing: {name}") from error
        if actual != version:
            raise ValueError(f"environment distribution differs: {name} {actual} != {version}")
        observed[name] = actual
    expected_python = runtime["python"]
    if Path(expected_python).absolute() != Path(sys.executable).absolute():
        raise ValueError("runtime interpreter differs from the executed environment")
    python = receipt.get("python")
    if not isinstance(python, str) or python.split()[0] != sys.version.split()[0]:
        raise ValueError("environment Python version differs")
    return {
        "environment_lock_sha256": lock_sha,
        "environment_receipt_sha256": receipt_sha,
        "verified_distribution_versions": observed,
        "verified_distribution_versions_sha256": sha256_json(observed),
        "python_executable": sys.executable,
        "python_version": sys.version,
    }


def persist_inference_receipt(output: Path, fitted, identity: dict) -> dict:
    """Save adapter-added post-inference evidence separately from the fit receipt."""
    receipt = {
        "schema": "official-expression-holdout-inference-provenance-v1",
        **identity,
        "inference_noise": fitted.receipt.get("inference_noise", []),
    }
    path = output / "inference-receipt.json"
    atomic_json(path, receipt)
    return {"inference_receipt_sha256": sha256_file(path)}
