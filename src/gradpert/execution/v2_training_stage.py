"""A training terminal receipt is distinct from a scientific run completion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file


def training_stage_evidence(root: Path, identity: dict[str, Any], epochs: int) -> dict[str, Any]:
    journal = read_json(root / "fit/epoch_state.json")
    history = read_json(root / "fit/history.json")
    if (
        read_json(root / "run_manifest.json") != identity
        or journal["identity"] != identity
        or journal["epoch"] != epochs
        or journal["budget"][0] != epochs
        or [row["epoch"] for row in history] != list(range(1, epochs + 1))
        or journal["last"]["epoch"] != epochs
    ):
        raise ValueError("training stage lacks the complete sealed epoch budget")
    source = identity["source"]
    if source["dirty"] or source["commit"] != source["published_commit"]:
        raise ValueError("training stage requires clean published source")
    roles = {role: journal[role] for role in ("best", "last") if journal[role] is not None}
    for selected in roles.values():
        checkpoint = (root / "fit" / selected["file"]).resolve(strict=True)
        if (
            not checkpoint.is_relative_to((root / "fit").resolve())
            or sha256_file(checkpoint) != selected["sha256"]
        ):
            raise ValueError("training stage checkpoint escaped or changed")
    return {
        "schema": "gradpert-v2-training-complete-1",
        "stage": "training_complete",
        "scientific_complete": False,
        "identity": identity,
        "epoch": epochs,
        "best": journal["best"],
        "last": journal["last"],
        "test_roles": list(roles),
        "history_sha256": sha256_file(root / "fit/history.json"),
        "epoch_state_sha256": sha256_file(root / "fit/epoch_state.json"),
    }


def seal_training_stage(root: Path, identity: dict[str, Any], epochs: int) -> dict[str, Any]:
    receipt = training_stage_evidence(root, identity, epochs)
    path = root / "TRAIN_COMPLETE.json"
    if path.exists() and read_json(path) != receipt:
        raise ValueError("existing training receipt belongs to a different stage")
    if not path.exists():
        atomic_json(path, receipt)
    return receipt


def validate_training_stage(plan: dict[str, Any]) -> dict[str, Any]:
    root = Path(plan["run_root"])
    if (root / "FAILURE.json").exists():
        raise ValueError("failed training cannot advance the queue")
    if read_json(root / "launch.json") != plan:
        raise ValueError("training stage launch differs from the sealed queue")
    manifest = read_json(root / "run_manifest.json")
    config = load_experiment_config(plan["config"])
    if (
        sha256_file(Path(plan["config"])) != plan["config_sha256"]
        or manifest["config_sha256"] != plan["config_sha256"]
        or manifest["source"]["commit"] != plan["source_commit"]
    ):
        raise ValueError("training stage source or config changed")
    receipt = training_stage_evidence(root, manifest, int(config.training.max_epochs.value))
    if read_json(root / "TRAIN_COMPLETE.json") != receipt:
        raise ValueError("training terminal receipt is missing or changed")
    return receipt
