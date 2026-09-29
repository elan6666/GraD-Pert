"""Verify and restore an immutable parent run for a new v2 training stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradpert.config.schema import ExperimentConfig
from gradpert.data._io import read_json
from gradpert.hashing import sha256_file, sha256_json
from gradpert.modeling.v2.lora import parent_to_lora_state

from .checkpoint import load_checkpoint
from .objective import JointObjective
from .optimizer import V2Optimizer


def restore_parent(
    config: ExperimentConfig,
    *,
    data_identity: dict[str, Any],
    objective: JointObjective,
    optimizer: V2Optimizer,
    generator: np.random.Generator,
    steps_per_epoch: int,
    server_root: Path = Path("/data/yilangliu"),
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], Path] | None, dict[str, Any]]:
    """Fail closed before a child run is committed or consumes a training batch."""
    stage = config.continuation
    if stage is None:
        raise ValueError("parent restore requires a continuation configuration")
    parent = Path(stage.parent_run_root).resolve(strict=True)
    checkpoint = Path(stage.parent_checkpoint).resolve(strict=True)
    if not parent.is_relative_to(server_root.resolve()) or not checkpoint.is_relative_to(parent):
        raise ValueError("parent checkpoint must remain inside a server run")
    manifest = read_json(parent / "run_manifest.json")
    complete = read_json(parent / "COMPLETE.json")
    journal = read_json(parent / "fit/epoch_state.json")
    prior_config = read_json(parent / "resolved_config.json")
    if (
        complete.get("identity") != manifest
        or complete.get("epoch") != stage.parent_epoch
        or set(complete.get("test_roles", ())) != {"best", "last"}
        or journal.get("identity") != manifest
        or journal.get("epoch") != stage.parent_epoch
        or journal.get("budget", [None])[0] != stage.parent_epoch
        or checkpoint != (parent / "fit" / journal["last"]["file"]).resolve(strict=True)
        or journal["last"]["epoch"] != stage.parent_epoch
        or journal["last"]["sha256"] != stage.parent_checkpoint_sha256
        or sha256_file(checkpoint) != stage.parent_checkpoint_sha256
    ):
        raise ValueError("parent run is incomplete or last checkpoint differs")
    if manifest.get("data") != data_identity or manifest.get(
        "resolved_config_sha256"
    ) != sha256_json(prior_config):
        raise ValueError("parent data or resolved configuration identity differs")
    current = config.model_dump(mode="json")
    for section in ("data", "evaluation", "artifacts"):
        if current[section] != prior_config[section]:
            raise ValueError(f"parent {section} protocol differs")
    if current["model"] != prior_config["model"]:
        raise ValueError("parent model or loss protocol differs")
    previous_training = dict(prior_config["training"])
    current_training = dict(current["training"])
    for name in ("formal_run_policy", "max_epochs"):
        previous_training.pop(name)
        current_training.pop(name)
    if current_training != previous_training:
        raise ValueError("parent training protocol differs outside the new budget")
    if journal["budget"][1] != steps_per_epoch:
        raise ValueError("checkpoint continuation must keep the original batch topology")
    if (
        objective.student.options.payload()
        != torch.load(checkpoint, map_location="cpu", weights_only=False)["architecture"]
    ):
        raise ValueError("parent architecture differs")
    if stage.mode == "full_state":
        progress = load_checkpoint(
            checkpoint,
            objective,
            optimizer,
            identity=manifest,
            generator=generator,
        )
        if optimizer.steps != stage.parent_epoch * steps_per_epoch:
            raise ValueError("parent optimizer step count differs")
        best = dict(journal["best"])
        best["training_identity"] = manifest
        parent_best = (best, parent / "fit" / best["file"])
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        objective.load_state_dict(parent_to_lora_state(payload["objective"], objective))
        objective.pending.clear()
        # Only the global progress counter continues. Adapter optimizer moments
        # are new and empty; the child receipt records that distinction.
        optimizer.steps = stage.parent_epoch * steps_per_epoch
        progress = payload["progress"]
        parent_best = None
    history = progress.get("history", [])
    if len(history) != stage.parent_epoch or history[-1].get("epoch") != stage.parent_epoch:
        raise ValueError("parent checkpoint progress differs from the journal")
    provenance = {
        "mode": stage.mode,
        "parent_run_id": manifest["run_id"],
        "parent_training_source": manifest["source"],
        "parent_config_sha256": manifest["config_sha256"],
        "parent_checkpoint_sha256": stage.parent_checkpoint_sha256,
        "parent_epoch": stage.parent_epoch,
        "optimizer_restored": stage.mode == "full_state",
        "learning_rate_after_parent": stage.learning_rate,
    }
    return history, parent_best, provenance
