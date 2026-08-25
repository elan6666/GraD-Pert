"""Small append-only CSV/JSON receipts for server training runs."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gradpert.training.step import GraDPertStepMetrics


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


class TrainingReceiptWriter:
    def __init__(self, root: str | Path, *, buffer_steps: int = 1) -> None:
        if not 1 <= buffer_steps <= 1024:
            raise ValueError("buffer_steps must be between 1 and 1024")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.root / "train_steps.csv"
        self.validation_path = self.root / "validation.csv"
        self.test_gate_path = self.root / "test_once.json"
        self._last_step = self._read_last_integer(self.steps_path, "global_step")
        self._last_validation_epoch = self._read_last_integer(self.validation_path, "epoch")
        self._buffer_steps = buffer_steps
        self._pending_steps: list[dict[str, Any]] = []

    @staticmethod
    def _read_last_integer(path: Path, field: str) -> int | None:
        if not path.exists():
            return None
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"training receipt must be a regular file: {path}")
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"training receipt has a header but no rows: {path}")
        try:
            values = [int(row[field]) for row in rows]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"training receipt has an invalid {field} column: {path}") from error
        if values != list(range(values[0], values[0] + len(values))):
            raise ValueError(f"training receipt {field} values are not contiguous: {path}")
        if values[0] != 0:
            raise ValueError(f"training receipt {field} values must start at zero: {path}")
        return values[-1]

    @staticmethod
    def _append(path: Path, row: Mapping[str, Any]) -> None:
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _append_many(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        exists = path.exists()
        fieldnames = list(rows[0])
        if any(list(row) != fieldnames for row in rows):
            raise ValueError("buffered receipt rows have inconsistent columns")
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def flush_steps(self) -> None:
        if not self._pending_steps:
            return
        self._append_many(self.steps_path, self._pending_steps)
        self._pending_steps.clear()

    def write_run_meta(self, payload: Mapping[str, Any]) -> None:
        if self.root.joinpath("run_meta.json").is_file():
            existing = json.loads(self.root.joinpath("run_meta.json").read_text(encoding="utf-8"))
            if existing != dict(payload):
                raise ValueError("existing run metadata differs from resumed run")
            return
        _atomic_json(self.root / "run_meta.json", payload)

    def write_step(self, *, epoch: int, global_step: int, metrics: GraDPertStepMetrics) -> None:
        expected = 0 if self._last_step is None else self._last_step + 1
        if global_step != expected:
            raise RuntimeError(
                f"refusing duplicate or discontinuous train step {global_step}; expected {expected}"
            )
        self._pending_steps.append({"epoch": epoch, "global_step": global_step, **asdict(metrics)})
        if len(self._pending_steps) >= self._buffer_steps:
            self.flush_steps()
        self._last_step = global_step

    def write_validation(
        self,
        *,
        epoch: int,
        global_step: int,
        txpert_macro_pearson_delta: float,
        improved: bool,
        consecutive_non_improvements: int,
    ) -> None:
        self.flush_steps()
        expected_epoch = (
            0 if self._last_validation_epoch is None else self._last_validation_epoch + 1
        )
        if epoch != expected_epoch:
            raise RuntimeError(
                "refusing duplicate or discontinuous validation epoch "
                f"{epoch}; expected {expected_epoch}"
            )
        self._append(
            self.validation_path,
            {
                "epoch": epoch,
                "global_step": global_step,
                "val_txpert_macro_pearson_delta": txpert_macro_pearson_delta,
                "improved": improved,
                "consecutive_non_improvements": consecutive_non_improvements,
            },
        )
        self._last_validation_epoch = epoch

    def claim_test_once(self) -> None:
        """Persist an at-most-once gate before any sealed test truth is accessed."""

        self.flush_steps()
        payload = {
            "schema_version": "gradpert-test-once-v1",
            "state": "started",
        }
        try:
            descriptor = os.open(
                self.test_gate_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            raise RuntimeError("sealed test evaluation has already been claimed") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")

    def complete_test_once(self) -> None:
        if not self.test_gate_path.is_file() or self.test_gate_path.is_symlink():
            raise RuntimeError("sealed test gate was not claimed")
        payload = json.loads(self.test_gate_path.read_text(encoding="utf-8"))
        if payload != {
            "schema_version": "gradpert-test-once-v1",
            "state": "started",
        }:
            raise RuntimeError("sealed test gate is invalid or already completed")
        _atomic_json(
            self.test_gate_path,
            {
                "schema_version": "gradpert-test-once-v1",
                "state": "completed",
            },
        )
