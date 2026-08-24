"""Epoch-level smoke/full orchestration without test-set model selection."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from gradpert.modeling import GraDPertJointModel
from gradpert.training.batch import GraDPertTrainingBatch
from gradpert.training.checkpoint import (
    CheckpointIdentity,
    load_training_checkpoint,
    save_training_checkpoint,
)
from gradpert.training.logging import TrainingReceiptWriter
from gradpert.training.selection import EarlyStoppingState
from gradpert.training.step import GraDPertStepEngine

TrainEpochFactory = Callable[[int], Iterable[GraDPertTrainingBatch]]
ValidationFunction = Callable[[GraDPertJointModel, int], float]
TestFunction = Callable[[GraDPertJointModel], None]


@dataclass
class GraDPertTrainingProgress:
    completed_epochs: int = 0
    global_step: int = 0
    test_evaluations: int = 0
    early_stopping: EarlyStoppingState | None = None

    def __post_init__(self) -> None:
        if self.early_stopping is None:
            self.early_stopping = EarlyStoppingState()
        if self.completed_epochs < 0 or self.global_step < 0:
            raise ValueError("training progress cannot be negative")
        if self.test_evaluations not in {0, 1}:
            raise ValueError("test evaluation count must be zero or one")

    def to_payload(self) -> dict[str, Any]:
        if self.early_stopping is None:  # pragma: no cover - closed in __post_init__
            raise AssertionError("early-stopping state is missing")
        return {
            "completed_epochs": self.completed_epochs,
            "global_step": self.global_step,
            "test_evaluations": self.test_evaluations,
            "early_stopping": asdict(self.early_stopping),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GraDPertTrainingProgress:
        return cls(
            completed_epochs=int(payload["completed_epochs"]),
            global_step=int(payload["global_step"]),
            test_evaluations=int(payload["test_evaluations"]),
            early_stopping=EarlyStoppingState(**payload["early_stopping"]),
        )


class GraDPertTrainer:
    """Run exactly one smoke epoch or the locked full validation-selected budget."""

    def __init__(
        self,
        *,
        engine: GraDPertStepEngine,
        checkpoint_identity: CheckpointIdentity,
        run_root: str | Path,
        steps_per_epoch: int,
        run_meta: Mapping[str, Any],
    ) -> None:
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if engine.total_schedule_steps != 200 * steps_per_epoch:
            raise ValueError("Teacher schedule must span the maximum 200-epoch budget")
        self.engine = engine
        self.identity = checkpoint_identity
        self.run_root = Path(run_root)
        self.steps_per_epoch = steps_per_epoch
        self.receipts = TrainingReceiptWriter(self.run_root / "small_results")
        self.receipts.write_run_meta(run_meta)
        self.progress = GraDPertTrainingProgress()

    @property
    def last_checkpoint(self) -> Path:
        return self.run_root / "checkpoints" / "last.pt"

    @property
    def best_checkpoint(self) -> Path:
        return self.run_root / "checkpoints" / "best.pt"

    def resume(self, path: str | Path | None = None) -> None:
        payload = load_training_checkpoint(
            self.last_checkpoint if path is None else path,
            model=self.engine.model,
            optimizer=self.engine.optimizer,
            centers=self.engine.centers,
            expected_identity=self.identity,
        )
        progress = GraDPertTrainingProgress.from_payload(payload)
        if progress.global_step != progress.completed_epochs * self.steps_per_epoch:
            raise ValueError("only exact epoch-boundary native checkpoints are resumable")
        self.progress = progress

    def _save(self, path: Path) -> str:
        return save_training_checkpoint(
            path,
            model=self.engine.model,
            optimizer=self.engine.optimizer,
            centers=self.engine.centers,
            progress=self.progress.to_payload(),
            identity=self.identity,
        )

    def fit(
        self,
        *,
        mode: Literal["smoke", "full"],
        train_epoch_factory: TrainEpochFactory,
        validate: ValidationFunction,
    ) -> GraDPertTrainingProgress:
        target_epochs = 1 if mode == "smoke" else 200
        if self.progress.completed_epochs > target_epochs:
            raise ValueError("checkpoint is beyond the requested run mode")
        for epoch in range(self.progress.completed_epochs, target_epochs):
            observed_steps = 0
            for batch in train_epoch_factory(epoch):
                metrics = self.engine.train_step(batch, global_step=self.progress.global_step)
                self.receipts.write_step(
                    epoch=epoch,
                    global_step=self.progress.global_step,
                    metrics=metrics,
                )
                self.progress.global_step += 1
                observed_steps += 1
            if observed_steps != self.steps_per_epoch:
                raise ValueError(
                    f"epoch {epoch} yielded {observed_steps} steps; expected {self.steps_per_epoch}"
                )
            self.progress.completed_epochs = epoch + 1
            validation_metric = float(validate(self.engine.model, epoch))
            early = self.progress.early_stopping
            if early is None:  # pragma: no cover - closed by progress
                raise AssertionError("early-stopping state is missing")
            improved, should_stop = early.update(
                epoch=epoch,
                validation_metric=validation_metric,
            )
            self.receipts.write_validation(
                epoch=epoch,
                global_step=self.progress.global_step,
                txpert_macro_pearson_delta=validation_metric,
                improved=improved,
                consecutive_non_improvements=early.consecutive_non_improvements,
            )
            if improved:
                self._save(self.best_checkpoint)
            self._save(self.last_checkpoint)
            if mode == "full" and should_stop:
                break
        return self.progress

    def test_best_once(self, test: TestFunction) -> None:
        if self.progress.test_evaluations != 0:
            raise RuntimeError("sealed test evaluation has already been consumed")
        if not self.best_checkpoint.is_file():
            raise RuntimeError("best validation checkpoint is unavailable")
        load_training_checkpoint(
            self.best_checkpoint,
            model=self.engine.model,
            optimizer=self.engine.optimizer,
            centers=self.engine.centers,
            expected_identity=self.identity,
        )
        # Claim the sealed test set before the callback can touch Truth.  A crash
        # leaves the durable gate in ``started`` and therefore fails closed on
        # every later attempt instead of silently evaluating test twice.
        self.receipts.claim_test_once()
        test(self.engine.model)
        self.receipts.complete_test_once()
        self.progress.test_evaluations = 1
        # Keep best.pt byte-stable: PredictionArtifact records this exact file
        # hash. The durable test gate plus last.pt carry the test lifecycle.
        self._save(self.last_checkpoint)
