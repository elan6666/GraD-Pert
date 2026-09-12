"""One native R50 update, full schedule horizon, no validation or test reader."""

import math
from dataclasses import asdict
from pathlib import Path

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.execution.step_resources import capture_step_resources
from gradpert.hashing import sha256_file
from scripts.performance.profile_native_a0 import (
    EvaluationAccessError,
    ProfileComplete,
    _checkpoint_roundtrip,
    _exact_engine_state,
    _training_only_evaluator_factory,
)


def run_native_step_smoke(**kwargs):
    """Process-scoped adapter; do not call concurrently in the same interpreter."""
    import torch

    import gradpert.execution.native as native
    from gradpert.training.step import GraDPertStepEngine
    from gradpert.training.trainer import GraDPertTrainer

    config = load_experiment_config(kwargs["config_path"])
    if config.training.formal_run_policy != "r50_selection" or not kwargs.get("formal"):
        raise ValueError("step smoke requires published R50 configuration")
    if kwargs.get("resume") or kwargs.get("mode") != "full":
        raise ValueError("step smoke requires a fresh full-horizon initialization")
    root = Path(kwargs["run_root"])
    if root.exists():
        raise FileExistsError("step smoke needs a fresh root")
    guard = {"guard_bindings": [], "truth_access_attempts": [], "validation_callback_count": 0}
    records = []
    identity = None
    original_fit = GraDPertTrainer.fit
    original_step = GraDPertStepEngine.train_step
    original_eval = native.CanonicalEvaluationData
    original_validate = native.evaluate_validation_macro_delta

    def fit(trainer, **fit_kwargs):
        nonlocal identity
        identity = trainer.identity
        if trainer.engine.total_schedule_steps != trainer.max_epochs * trainer.steps_per_epoch:
            raise ValueError("full schedule horizon changed")
        return original_fit(trainer, **fit_kwargs)

    def reject_validation(*args, **kw):
        guard["validation_callback_count"] += 1
        raise EvaluationAccessError("single-step smoke forbids validation")

    def step(engine, batch, *, global_step):
        if records or global_step != 0 or identity is None:
            raise RuntimeError("single-step boundary violated")
        metrics = original_step(engine, batch, global_step=global_step)
        payload = asdict(metrics)
        for value in payload.values():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError("nonfinite step metric")
        for parameter in engine.model.parameters():
            if not torch.isfinite(parameter).all() or (
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ):
                raise RuntimeError("nonfinite model or gradient")
        resources = capture_step_resources(next(engine.model.parameters()).device, root)
        state = _exact_engine_state(engine)
        checkpoint = root / "checkpoints/step-000001.pt"
        checkpoint_sha = _checkpoint_roundtrip(engine, checkpoint, identity, global_step)
        records.append(
            {
                "global_step": 0,
                "completed_steps": 1,
                "completed_epochs": 0,
                "metrics": payload,
                "state_sha256": state,
                "schedule_horizon_steps": engine.total_schedule_steps,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_role": "diagnostic_after_step_not_validation_best",
                "checkpoint_roundtrip_exact": True,
                "resources": resources,
            }
        )
        raise ProfileComplete("one real native update completed")

    GraDPertTrainer.fit = fit
    GraDPertStepEngine.train_step = step
    native.CanonicalEvaluationData = _training_only_evaluator_factory(guard)
    native.evaluate_validation_macro_delta = reject_validation
    try:
        try:
            native.run_native_experiment(**kwargs)
        except ProfileComplete:
            if len(records) != 1:
                raise RuntimeError("step boundary returned without evidence") from None
        else:
            raise RuntimeError("native training escaped its single-step boundary")
    finally:
        GraDPertTrainer.fit = original_fit
        GraDPertStepEngine.train_step = original_step
        native.CanonicalEvaluationData = original_eval
        native.evaluate_validation_macro_delta = original_validate
    if guard["truth_access_attempts"] or guard["validation_callback_count"]:
        raise RuntimeError("step smoke touched evaluation")
    if any(p.suffix.lower() in {".pkl", ".pickle"} for p in root.rglob("*")):
        raise RuntimeError("step smoke persisted PKL")
    receipt = {
        "schema": "native-r50-one-step-v1",
        "status": "complete",
        "scientific_completion": False,
        "test_evaluations": 0,
        "config_sha256": sha256_file(kwargs["config_path"]),
        "source_identity_sha256": sha256_file(root / "small_results/source_identity.json"),
        "steps": records,
        "evaluation_guard": guard,
    }
    atomic_json(root / "small_results/one_step_smoke.json", receipt)
    return receipt


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config-path",
        "data-root",
        "run-root",
        "repository-root",
        "source-publication-receipt",
        "genept-preflight-receipt",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("run-id", "source-publication-receipt-sha256", "genept-preflight-receipt-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device-name", default="cuda:0")
    args = vars(parser.parse_args())
    print(
        json.dumps(
            run_native_step_smoke(
                **args,
                formal=True,
                mode="full",
                run_seed=1,
                source_publication_remote_ref="refs/heads/main",
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
