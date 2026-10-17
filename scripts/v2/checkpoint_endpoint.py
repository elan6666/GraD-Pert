"""Bind read-only diagnostics to a real, retained checkpoint and training phase."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def select_checkpoint(root: Path, role: str = "last") -> tuple[dict, dict]:
    root = Path(root).resolve()
    config = json.loads((root / "resolved_config.json").read_text())
    journal = json.loads((root / "fit/epoch_state.json").read_text())
    parameters = config.get("model", {}).get("parameters", {})
    staged = parameters.get("training_mode", {}).get("value") == "ssl_then_mse"
    total = config["training"]["max_epochs"]["value"]
    if role == "pretrain":
        if not staged:
            raise ValueError("pretrain diagnostics require an explicitly staged model")
        epoch = parameters["ssl_pretrain_epochs"]["value"]
        if type(epoch) is not int or not 0 < epoch < total:
            raise ValueError("invalid pretraining checkpoint boundary")
        stage = "ssl_pretrain"
    elif role == "last":
        epoch, stage = total, "mse_finetune" if staged else "joint"
    else:
        raise ValueError("only last and retained pretrain checkpoints are diagnostic endpoints")
    cp = journal.get(role)
    if not isinstance(cp, dict) or cp.get("epoch") != epoch:
        raise ValueError("selected checkpoint does not match the declared endpoint epoch")
    path = (root / "fit" / cp["file"]).resolve()
    if not path.is_relative_to(root / "fit") or not path.is_file():
        raise ValueError("checkpoint must be a retained file inside this run")
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != cp["sha256"]:
        raise ValueError("selected checkpoint content changed")
    return cp, dict(role=role, epoch=epoch, training_stage=stage, full_training_epochs=total)
