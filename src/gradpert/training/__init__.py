"""Native GraD-Pert B2 training, selection, pairing, and checkpoint mechanics.

Torch-backed names are imported lazily so data/config tooling remains usable in
lightweight environments.
"""

from typing import Any

from gradpert.training.controls import TrainingControlPairer, TrainingControlPairing
from gradpert.training.data import (
    CanonicalTrainingData,
    condition_limited_epoch_batches,
    write_training_data_receipt,
)
from gradpert.training.selection import EarlyStoppingState

__all__ = [
    "CanonicalTrainingData",
    "EarlyStoppingState",
    "GraDPertStepEngine",
    "GraDPertStepMetrics",
    "GraDPertTrainingBatch",
    "TrainingControlPairer",
    "TrainingControlPairing",
    "ValidationMetricResult",
    "build_native_optimizer",
    "condition_limited_epoch_batches",
    "evaluate_validation_macro_delta",
    "write_training_data_receipt",
]


def __getattr__(name: str) -> Any:
    if name == "GraDPertTrainingBatch":
        from gradpert.training.batch import GraDPertTrainingBatch

        return GraDPertTrainingBatch
    if name in {"GraDPertStepEngine", "GraDPertStepMetrics", "build_native_optimizer"}:
        from gradpert.training.step import (
            GraDPertStepEngine,
            GraDPertStepMetrics,
            build_native_optimizer,
        )

        return {
            "GraDPertStepEngine": GraDPertStepEngine,
            "GraDPertStepMetrics": GraDPertStepMetrics,
            "build_native_optimizer": build_native_optimizer,
        }[name]
    if name in {"ValidationMetricResult", "evaluate_validation_macro_delta"}:
        from gradpert.training.validation import (
            ValidationMetricResult,
            evaluate_validation_macro_delta,
        )

        return {
            "ValidationMetricResult": ValidationMetricResult,
            "evaluate_validation_macro_delta": evaluate_validation_macro_delta,
        }[name]
    raise AttributeError(name)
