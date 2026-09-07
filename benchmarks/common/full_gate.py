"""Fail-closed one-epoch integration prerequisite for external full training."""

import json
from pathlib import Path

from gradpert.hashing import sha256_file


def require_completed_smoke(root, *, config, config_sha256, training_data, source_commit):
    if root is None:
        raise ValueError("external100 requires --smoke-run-root")
    root = Path(root).resolve(strict=True)
    small = root / "small_results"
    manifest_path = small / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    training = json.loads((small / "training_receipt.json").read_text())
    expected = {
        "status": "evaluated",
        "formal_eligible": True,
        "source_dirty": False,
        "model_id": config.model_id,
        "dataset_id": config.dataset_id,
        "config_sha256": config_sha256,
        "source_commit": source_commit,
        "canonical_data_sha256": training_data.manifest.canonical_adata_sha256,
        "split_content_sha256": training_data.split.split_content_sha256,
        "test_evaluations": 1,
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("one-epoch smoke source/config/data/status identity mismatch")
    if training.get("epochs_requested") != 1 or training.get("epochs_completed") != 1:
        raise ValueError("external smoke must complete exactly one epoch")
    if training.get("canonical_test_truth_present_during_fit") is not False:
        raise ValueError("smoke must explicitly exclude canonical test truth during fitting")
    checkpoints = [*root.rglob("*.pt"), *root.rglob("*.ckpt")]
    if len(checkpoints) != 1 or sha256_file(checkpoints[0]) != manifest["best_checkpoint_sha256"]:
        raise ValueError("smoke best checkpoint is absent, changed or not unique")
    if (
        list(root.rglob("*.pkl"))
        or list(root.rglob("*.pickle"))
        or list(root.glob(".result-work-*"))
    ):
        raise ValueError("smoke zero-PKL/work-directory postcondition failed")
    return {
        "smoke_root": str(root),
        "run_manifest_sha256": sha256_file(manifest_path),
        "training_receipt_sha256": sha256_file(small / "training_receipt.json"),
    }
