"""Receipt-bound experiment execution entry points."""

from typing import Any

from gradpert.execution.identity import (
    EnvironmentIdentity,
    SourceIdentity,
    inspect_environment,
    inspect_source_identity,
)

__all__ = [
    "EnvironmentIdentity",
    "NativeRunResult",
    "NonlearnedRunResult",
    "SourceIdentity",
    "inspect_environment",
    "inspect_source_identity",
    "run_native_experiment",
    "run_nonlearned_experiment",
]


def __getattr__(name: str) -> Any:
    if name in {"NativeRunResult", "run_native_experiment"}:
        from gradpert.execution.native import NativeRunResult, run_native_experiment

        return {
            "NativeRunResult": NativeRunResult,
            "run_native_experiment": run_native_experiment,
        }[name]
    if name in {"NonlearnedRunResult", "run_nonlearned_experiment"}:
        from gradpert.execution.nonlearned import (
            NonlearnedRunResult,
            run_nonlearned_experiment,
        )

        return {
            "NonlearnedRunResult": NonlearnedRunResult,
            "run_nonlearned_experiment": run_nonlearned_experiment,
        }[name]
    raise AttributeError(name)
