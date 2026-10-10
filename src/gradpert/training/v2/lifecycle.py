"""Resumable epoch lifecycle with an atomic checkpoint selection journal.

Data/source/resource checks belong to the execution adapter. This module also
supports small synthetic lifecycles so the identical state machine can be tested
without accessing scientific data or shortening a formal experiment.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class ConstantStageLR:
    """Explicit post-checkpoint LR; the completed parent cosine is not replayed."""

    learning_rate: float

    def at_step(self, step: int, total_steps: int) -> dict[str, float]:
        if step < 0 or total_steps < 1 or self.learning_rate <= 0:
            raise ValueError("invalid continuation learning-rate contract")
        return {"learning_rate": self.learning_rate}


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
    validate: Callable[[], dict[str, Any]] | None,
    schedule: EndpointLRWarmupCosine | ConstantStageLR,
    teacher_start: float,
    teacher_end: float,
    microbatch: int,
    bf16: bool,
    resume: bool = False,
    bootstrap_history: list[dict[str, Any]] | None = None,
    bootstrap_best: tuple[dict[str, Any], Path] | None = None,
    diagnose: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Commit only complete epochs; resume replays an interrupted epoch exactly.

    The caller's epoch sampler must be deterministic in (run seed, epoch).
    The checkpoint restores all model/optimizer/view RNG state at the preceding
    epoch boundary. Validation callbacks must return only the validation split.
    """
    if epochs < 1 or steps_per_epoch < 1 or not 0 < teacher_start <= teacher_end <= 1:
        raise ValueError("invalid lifecycle budget or teacher momentum")
    selection_metric = "joint_loss" if validate is not None else "final_epoch"
    if validate is None and (bootstrap_history is not None or bootstrap_best is not None):
        raise ValueError("train-only lifecycle cannot reinterpret validated parent history")
    contract = {
        "schedule": asdict(schedule),
        "teacher_start": teacher_start,
        "teacher_end": teacher_end,
        "microbatch": microbatch,
        "bf16": bf16,
        "selection_metric": selection_metric,
    }
    if resume and bootstrap_history is not None:
        raise ValueError("resume must use the committed child journal, not bootstrap again")
    if not resume and bootstrap_history is None and optimizer.steps:
        raise ValueError("new lifecycle requires an unstepped optimizer")
    if bootstrap_history is not None and (
        not bootstrap_history
        or len(bootstrap_history) >= epochs
        or optimizer.steps != len(bootstrap_history) * steps_per_epoch
        or any(record.get("epoch") != index for index, record in enumerate(bootstrap_history, 1))
    ):
        raise ValueError("parent progress differs from the continuation budget")
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
        history = list(bootstrap_history or [])
        initial_epoch = len(history)
        initial = root / f"epoch-{initial_epoch:04d}.pt"
        save_checkpoint(
            initial,
            objective,
            optimizer,
            identity=identity,
            progress={"history": history},
            generator=generator,
        )
        best = None
        if bootstrap_best is not None:
            if not history:
                raise ValueError("parent best requires parent history")
            selected, source_path = bootstrap_best
            if selected["epoch"] > initial_epoch or sha256_file(source_path) != selected["sha256"]:
                raise ValueError("parent best checkpoint differs from the parent journal")
            copied = root / f"parent-{selected['file']}"
            primary_call(lambda: shutil.copyfile(source_path, copied))
            best = {**selected, "file": copied.name}
        journal = {
            "identity": identity,
            "budget": [epochs, steps_per_epoch],
            "contract": contract,
            "selection_metric": selection_metric,
            "epoch": initial_epoch,
            "selection_start_epoch": 0 if best is not None else initial_epoch,
            "best": best,
            "last": {
                "file": initial.name,
                "sha256": sha256_file(initial),
                "epoch": initial_epoch,
                "prediction_loss": None,
            },
        }
        primary_call(partial(atomic_json, journal_path, journal))
    progress_started = time.monotonic()

    def live(
        phase: str,
        epoch_number: int,
        epoch_step: int,
        terms: dict[str, float] | None = None,
        throughput: dict[str, float] | None = None,
    ) -> None:
        """Publish diagnostics after committed updates without changing training collectives."""
        if rank != 0:
            return
        completed = (epoch_number - 1) * steps_per_epoch + epoch_step
        elapsed = max(0.0, time.monotonic() - progress_started)
        payload = {
            "schema_version": "gradpert-v2-live-progress-1",
            "run_id": identity.get("run_id"),
            "phase": phase,
            "epoch": epoch_number,
            "epochs_total": epochs,
            "epoch_step": epoch_step,
            "epoch_steps_total": steps_per_epoch,
            "optimizer_steps_completed": completed,
            "optimizer_steps_total": total,
            "fraction_complete": completed / total,
            "elapsed_seconds_this_process": elapsed,
            "latest_training_terms": terms,
            "throughput": throughput,
        }
        try:
            atomic_json(root / "live_progress.json", payload)
        except OSError as error:
            # The progress display is diagnostic; a transient filesystem error
            # cannot change model updates, EMA, centers, or checkpoint selection.
            print(f"v2 live-progress write failed: {error}", flush=True)

    primary_call(lambda: atomic_json(root / "history.json", history))
    # Reuse native strict-improvement selection. Fixed-budget v2 deliberately
    # ignores the early-stop signal, just like native R50 selection runs.
    selection = EarlyStoppingState(mode="min")
    for record in history[journal.get("selection_start_epoch", 0) :] if validate else []:
        selection.update(
            epoch=record["epoch"], validation_metric=float(record["validation"]["joint_loss"])
        )
    seen_expression_ids = set(history[-1]["seen_expression_gene_indices"]) if history else set()
    if diagnose is not None and not history:
        initial_diagnostic = primary_call(diagnose)
        primary_call(
            lambda: atomic_json(root / "loss-diagnostic-epoch-0000.json", initial_diagnostic)
        )
    for epoch in range(len(history), epochs):
        sums: dict[str, float] = {}
        epoch_started = time.monotonic()
        epoch_cells = 0
        condition_audit: dict[str, dict[str, float]] = {}
        device = next(objective.student.parameters()).device
        if diagnose is not None and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        live("training", epoch + 1, 0)

        def update(
            batch: TrainingBatch,
            count: int,
            *,
            current_epoch: int = epoch,
            totals: dict[str, float] = sums,
            start_epoch: float = epoch_started,
            exposure_audit: dict[str, dict[str, float]] = condition_audit,
        ) -> None:
            nonlocal epoch_cells
            step = current_epoch * steps_per_epoch + count
            momentum = (
                teacher_end
                - (teacher_end - teacher_start)
                * (1 + math.cos(math.pi * step / max(1, total - 1)))
                / 2
            )
            global_conditions = batch.condition_index if distributed else None
            global_cells = len(batch.control)
            if diagnose is not None:
                conditions, counts = torch.unique(batch.condition_index, return_counts=True)
                for condition, cells in zip(conditions.tolist(), counts.tolist(), strict=True):
                    positions = batch.graph.target_positions[condition]
                    targets = positions[batch.graph.target_valid[condition]]
                    key = ",".join(str(x) for x in batch.graph.ids[targets].tolist())
                    record = exposure_audit.setdefault(key, {"rows": 0.0, "weight": 0.0})
                    record["rows"] += cells
                    record["weight"] += (
                        1 / len(conditions)
                        if objective.prediction_strategy == "condition_mean"
                        else cells / global_cells
                    )
            if distributed:
                if global_cells < world and not objective.population_response:
                    raise ValueError("global batch must provide a row to every rank")
                batch = slice_cells(
                    batch,
                    global_cells * rank // world,
                    global_cells * (rank + 1) // world,
                )
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
            seen_expression_ids.update(
                int(gene) for gene in batch.graph.ids[batch.query_positions].tolist()
            )
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + value
            epoch_cells += global_cells
            epoch_elapsed = max(time.monotonic() - start_epoch, 1e-9)
            live(
                "training",
                current_epoch + 1,
                count + 1,
                terms,
                {
                    "cells_completed_this_epoch": epoch_cells,
                    "cells_per_second_this_epoch": epoch_cells / epoch_elapsed,
                    "optimizer_steps_per_second_this_epoch": (count + 1) / epoch_elapsed,
                },
            )
            if rank == 0 and ((count + 1) % 10 == 0 or count + 1 == steps_per_epoch):
                print(
                    f"v2 epoch {current_epoch + 1}/{epochs} step "
                    f"{count + 1}/{steps_per_epoch} joint_loss="
                    f"{terms.get('joint_loss', float('nan')):.6f}",
                    flush=True,
                )

        count = execute_epoch(batches(epoch), update, maximum_steps=steps_per_epoch)
        if count != steps_per_epoch:
            raise ValueError("epoch iterator shorter than sealed step budget")
        loss: float | None = None
        validation: dict[str, Any] = {"validation_mode": "disabled", "performed": False}
        if validate is not None:
            live("validation", epoch + 1, steps_per_epoch)
            validation = primary_call(validate)
            if validation.get("split") != "val":
                raise ValueError("checkpoint selection requires validation-only results")
            loss = float(validation["joint_loss"])
            if not math.isfinite(loss):
                raise FloatingPointError("nonfinite validation selection loss")
        training_seconds = time.monotonic() - epoch_started
        peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        diagnostic_started = time.monotonic()
        diagnostic = primary_call(diagnose) if diagnose is not None else None
        if diagnostic is not None:
            primary_call(
                partial(
                    atomic_json, root / f"loss-diagnostic-epoch-{epoch + 1:04d}.json", diagnostic
                )
            )
        history.append(
            {
                "epoch": epoch + 1,
                "optimizer_steps": optimizer.steps,
                "training": {k: v / count for k, v in sums.items()},
                "validation": validation,
                **(
                    {
                        "training_diagnostic": diagnostic,
                        "condition_prediction_weight_audit": condition_audit,
                        "training_seconds": training_seconds,
                        "diagnostic_seconds": time.monotonic() - diagnostic_started,
                        "rank0_peak_allocated_bytes": peak_bytes,
                    }
                    if diagnostic is not None
                    else {}
                ),
                "seen_expression_gene_indices": sorted(seen_expression_ids),
            }
        )
        checkpoint = root / f"epoch-{epoch + 1:04d}.pt"
        live("checkpointing", epoch + 1, steps_per_epoch)
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
            **({"joint_loss": loss} if loss is not None else {}),
            **(
                {"prediction_loss": float(validation["prediction_loss"])}
                if "prediction_loss" in validation
                else {}
            ),
        }
        best = journal.get("best")
        if loss is not None:
            improved, _ = selection.update(epoch=epoch + 1, validation_metric=loss)
            if improved:
                best = selected
            if best is None:  # The first finite validation must select a checkpoint.
                raise AssertionError("native selection did not select an initial checkpoint")
        journal = {
            "identity": identity,
            "budget": [epochs, steps_per_epoch],
            "contract": contract,
            "selection_metric": selection_metric,
            "epoch": epoch + 1,
            "selection_start_epoch": journal.get("selection_start_epoch", 0),
            "best": best,
            "last": selected,
        }
        # This rename is the commit point. A crash before it leaves an orphan,
        # which is ignored on resume; after it role links are safely recoverable.
        primary_call(partial(atomic_json, journal_path, journal))
        primary_call(partial(_role_links, root, journal))
        primary_call(lambda: atomic_json(root / "history.json", history))
        retained = {str(selected["file"])} | ({str(best["file"])} if best else set())

        def prune(retained_files: set[str] = retained) -> None:
            for old in root.glob("epoch-*.pt"):
                if old.name not in retained_files:
                    old.unlink()

        primary_call(prune)
        live("epoch_complete", epoch + 1, steps_per_epoch)
    live("training_complete", epochs, steps_per_epoch)
    return journal


def _test_selected(
    objective: JointObjective,
    *,
    root: Path,
    evaluation_identity: dict[str, Any],
    test: Callable[[], dict[str, Any]],
    on_role_start: Callable[[str], None] | None = None,
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
        if selected is None:
            if role != "best" or journal.get("selection_metric") != "final_epoch":
                raise ValueError("selected checkpoint is missing")
            continue
        identity = {
            "training": selected.get("training_identity", journal["identity"]),
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
        if on_role_start is not None:
            on_role_start(role)
        load_evaluation_checkpoint(
            root / selected["file"],
            objective,
            training_identity=selected.get("training_identity", journal["identity"]),
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
    on_role_start: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    return primary_call(
        lambda: _test_selected(
            objective,
            root=root,
            evaluation_identity=evaluation_identity,
            test=test,
            on_role_start=on_role_start,
        )
    )
