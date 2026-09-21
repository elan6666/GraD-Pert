"""Resumable epoch lifecycle with an atomic checkpoint selection journal.

Data/source/resource checks belong to the execution adapter. This module also
supports small synthetic lifecycles so the identical state machine can be tested
without accessing scientific data or shortening a formal experiment.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradpert.config.step_schedule import EndpointLRWarmupCosine
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file
from gradpert.training.epoch import execute_epoch
from gradpert.training.selection import EarlyStoppingState

from .checkpoint import load_checkpoint, load_evaluation_checkpoint, save_checkpoint
from .distributed import primary_call
from .engine import optimizer_step, slice_cells
from .objective import JointObjective, TrainingBatch
from .optimizer import V2Optimizer


def _role_links(root: Path, journal: dict[str, Any]) -> None:
    for role in ("best", "last"):
        if journal[role] is None:
            continue
        target = root / journal[role]["file"]
        if sha256_file(target) != journal[role]["sha256"]:
            raise ValueError(f"{role} checkpoint differs from epoch journal")
        temporary = root / f".{role}.tmp"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target.name)
        os.replace(temporary, root / f"{role}.pt")


def fit(
    objective: JointObjective,
    optimizer: V2Optimizer,
    *,
    root: Path,
    identity: dict[str, Any],
    generator: np.random.Generator,
    epochs: int,
    steps_per_epoch: int,
    batches: Callable[[int], Iterable[TrainingBatch]],
    validate: Callable[[], dict[str, Any]],
    schedule: EndpointLRWarmupCosine,
    teacher_start: float,
    teacher_end: float,
    microbatch: int,
    bf16: bool,
    resume: bool = False,
) -> dict[str, Any]:
    """Commit only complete epochs; resume replays an interrupted epoch exactly.

    The caller's epoch sampler must be deterministic in (run seed, epoch).
    The checkpoint restores all model/optimizer/view RNG state at the preceding
    epoch boundary. Validation callbacks must return only the validation split.
    """
    if epochs < 1 or steps_per_epoch < 1 or not 0 < teacher_start <= teacher_end <= 1:
        raise ValueError("invalid lifecycle budget or teacher momentum")
    contract = {
        "schedule": asdict(schedule),
        "teacher_start": teacher_start,
        "teacher_end": teacher_end,
        "microbatch": microbatch,
        "bf16": bf16,
    }
    if not resume and optimizer.steps:
        raise ValueError("new lifecycle requires an unstepped optimizer")
    total = epochs * steps_per_epoch
    schedule.at_step(0, total)  # Reject invalid timing before writing run state.
    distributed = torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0
    primary_call(lambda: root.mkdir(parents=True, exist_ok=True))
    journal_path = root / "epoch_state.json"
    history: list[dict[str, Any]] = []
    journal: dict[str, Any] = {}
    if resume:
        journal = read_json(journal_path)
        if journal["identity"] != identity or journal["budget"] != [epochs, steps_per_epoch]:
            raise ValueError("resume identity or training budget differs")
        if journal.get("contract") != contract:
            raise ValueError("resume schedule or execution contract differs")
        primary_call(partial(_role_links, root, journal))
        progress = load_checkpoint(
            root / "last.pt", objective, optimizer, identity=identity, generator=generator
        )
        history = progress["history"]
        if len(history) != journal["epoch"] or optimizer.steps != len(history) * steps_per_epoch:
            raise ValueError("checkpoint progress differs from committed epoch journal")
    elif journal_path.exists() or any(root.iterdir()):
        raise FileExistsError("new training requires an empty lifecycle directory")
    if not resume:
        initial = root / "epoch-0000.pt"
        save_checkpoint(
            initial,
            objective,
            optimizer,
            identity=identity,
            progress={"history": []},
            generator=generator,
        )
        journal = {
            "identity": identity,
            "budget": [epochs, steps_per_epoch],
            "contract": contract,
            "epoch": 0,
            "best": None,
            "last": {
                "file": initial.name,
                "sha256": sha256_file(initial),
                "epoch": 0,
                "prediction_loss": None,
            },
        }
        primary_call(partial(atomic_json, journal_path, journal))
    primary_call(lambda: atomic_json(root / "history.json", history))
    # Reuse native strict-improvement selection. Fixed-budget v2 deliberately
    # ignores the early-stop signal, just like native R50 selection runs.
    selection = EarlyStoppingState(mode="min")
    for record in history:
        selection.update(
            epoch=record["epoch"], validation_metric=float(record["validation"]["prediction_loss"])
        )
    for epoch in range(len(history), epochs):
        sums: dict[str, float] = {}

        def update(
            batch: TrainingBatch,
            count: int,
            *,
            current_epoch: int = epoch,
            totals: dict[str, float] = sums,
        ) -> None:
            step = current_epoch * steps_per_epoch + count
            momentum = (
                teacher_end
                - (teacher_end - teacher_start)
                * (1 + math.cos(math.pi * step / max(1, total - 1)))
                / 2
            )
            global_conditions = batch.condition_index if distributed else None
            if distributed:
                cells = len(batch.control)
                if cells < world:
                    raise ValueError("global batch must provide a row to every rank")
                batch = slice_cells(batch, cells * rank // world, cells * (rank + 1) // world)
            terms = optimizer_step(
                objective,
                optimizer,
                batch,
                microbatch=microbatch,
                lr=schedule.at_step(step, total)["learning_rate"],
                momentum=momentum,
                bf16=bf16,
                global_condition_index=global_conditions,
            )
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + value

        count = execute_epoch(batches(epoch), update, maximum_steps=steps_per_epoch)
        if count != steps_per_epoch:
            raise ValueError("epoch iterator shorter than sealed step budget")
        validation = primary_call(validate)
        if validation.get("split") != "val":
            raise ValueError("checkpoint selection requires validation-only results")
        loss = float(validation["prediction_loss"])
        if not math.isfinite(loss):
            raise FloatingPointError("nonfinite validation selection loss")
        history.append(
            {
                "epoch": epoch + 1,
                "optimizer_steps": optimizer.steps,
                "training": {k: v / count for k, v in sums.items()},
                "validation": validation,
            }
        )
        checkpoint = root / f"epoch-{epoch + 1:04d}.pt"
        save_checkpoint(
            checkpoint,
            objective,
            optimizer,
            identity=identity,
            progress={"history": history},
            generator=generator,
        )
        selected = {
            "file": checkpoint.name,
            "sha256": sha256_file(checkpoint),
            "epoch": epoch + 1,
            "prediction_loss": loss,
        }
        best = journal.get("best")
        improved, _ = selection.update(epoch=epoch + 1, validation_metric=loss)
        if improved:
            best = selected
        if best is None:  # The first finite validation must select a checkpoint.
            raise AssertionError("native selection did not select an initial checkpoint")
        journal = {
            "identity": identity,
            "budget": [epochs, steps_per_epoch],
            "contract": contract,
            "epoch": epoch + 1,
            "best": best,
            "last": selected,
        }
        # This rename is the commit point. A crash before it leaves an orphan,
        # which is ignored on resume; after it role links are safely recoverable.
        primary_call(partial(atomic_json, journal_path, journal))
        primary_call(partial(_role_links, root, journal))
        primary_call(lambda: atomic_json(root / "history.json", history))
        retained = {str(best["file"]), str(selected["file"])}

        def prune(retained_files: set[str] = retained) -> None:
            for old in root.glob("epoch-*.pt"):
                if old.name not in retained_files:
                    old.unlink()

        primary_call(prune)
    return journal


def _test_selected(
    objective: JointObjective,
    *,
    root: Path,
    evaluation_identity: dict[str, Any],
    test: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate both selected roles after the full sealed epoch budget.

    Existing receipts may be reused only for identical checkpoint and evaluation
    identities. The execution adapter supplies source/environment/data identities
    separately for training and evaluation. No optimizer or RNG is restored here.
    """
    journal = read_json(root / "epoch_state.json")
    if journal["epoch"] != journal["budget"][0]:
        raise ValueError("best/last test requires the complete training budget")
    _role_links(root, journal)
    receipts = {}
    for role in ("best", "last"):
        selected = journal[role]
        identity = {
            "training": journal["identity"],
            "evaluation": evaluation_identity,
            "checkpoint": selected,
            "role": role,
        }
        output = root / f"{role}-test.json"
        if output.exists():
            saved = read_json(output)
            if saved["identity"] != identity:
                raise ValueError("existing test receipt belongs to another evaluation")
            receipts[role] = saved
            continue
        load_evaluation_checkpoint(
            root / selected["file"],
            objective,
            training_identity=journal["identity"],
            checkpoint_sha256=selected["sha256"],
        )
        result = test()
        if result.get("split") != "test":
            raise ValueError("postfit requires the frozen test split")
        receipts[role] = {"identity": identity, "result": result}
        atomic_json(output, receipts[role])
    return receipts


def test_selected(
    objective: JointObjective,
    *,
    root: Path,
    evaluation_identity: dict[str, Any],
    test: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    return primary_call(
        lambda: _test_selected(
            objective, root=root, evaluation_identity=evaluation_identity, test=test
        )
    )
