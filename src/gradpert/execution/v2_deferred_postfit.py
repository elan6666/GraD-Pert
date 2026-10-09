"""Drain independent checkpoint tests without treating training as completion."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from gradpert.data._io import atomic_json, read_json
from gradpert.evaluation.state import prepare_evaluation_state
from gradpert.execution.v2_checkpoint_eval import execute_evaluation_plan, resolve_evaluation_plan
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file


def finalize_deferred_run(plan: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    root = Path(plan["run_root"])
    for role in training["test_roles"]:
        selected = training[role]
        test = read_json(root / "fit" / f"{role}-test.json")
        identity = test["identity"]
        if not test.get("independent_evaluation_receipt") or not test.get(
            "independent_evaluation_sha256"
        ):
            raise ValueError("independent evaluation receipt evidence missing")
        independent_path = Path(test["independent_evaluation_receipt"]).resolve(strict=True)
        if (
            not independent_path.is_relative_to(root.resolve())
            or sha256_file(independent_path) != test["independent_evaluation_sha256"]
        ):
            raise ValueError("independent evaluation receipt path or hash differs")
        independent = read_json(independent_path)
        evaluated = independent["plan"]
        if (
            independent["result"] != test["result"]
            or not independent["zero_pkl"]
            or evaluated["training_identity"] != identity["training"]
            or evaluated["checkpoint_sha256"] != selected["sha256"]
            or evaluated["checkpoint_role"] != role
            or evaluated["checkpoint_epoch"] != selected["epoch"]
            or evaluated["training_run_root"] != str(root.resolve())
            or evaluated["split"] != "test"
            or evaluated["output_root"] != str(independent_path.parent)
            or evaluated["evaluation_source"] != identity["evaluation"]["source"]
            or independent["evaluation_environment"] != identity["evaluation"]["environment"]
        ):
            raise ValueError("independent evaluation receipt differs from canonical test")
        if (
            identity["training"] != selected.get("training_identity", training["identity"])
            or identity["checkpoint"] != selected
            or identity["role"] != role
            or test["result"]["split"] != "test"
            or identity["evaluation"]["source"]["dirty"]
        ):
            raise ValueError("postfit test identity or role differs from training")
    if any(root.rglob("*.pkl")):
        raise ValueError("successful run must contain zero PKL")
    complete = {
        "identity": training["identity"],
        "epoch": training["epoch"],
        "best": training["best"],
        "last": training["last"],
        "test_roles": training["test_roles"],
        "zero_pkl": True,
        "training_complete_sha256": sha256_file(root / "TRAIN_COMPLETE.json"),
        "postfit_policy": "deferred",
    }
    if "continuation" in training["identity"]:
        complete["continuation"] = training["identity"]["continuation"]
    path = root / "COMPLETE.json"
    if path.exists() and read_json(path) != complete:
        raise ValueError("existing scientific completion differs")
    if not path.exists():
        atomic_json(path, complete)
    return complete


def run_deferred_postfit(plan: dict[str, Any], *, gpu: str = "0,1") -> dict[str, Any]:
    training = validate_training_stage(plan)
    root = Path(plan["run_root"])
    config = plan["resolved_config"]
    if plan.get("postfit_policy") != "deferred":
        raise ValueError("postfit drain requires an explicitly deferred launch")
    started = time.perf_counter()
    try:
        prepare_evaluation_state(
            dataset_id=config["dataset_id"],
            protocol_id=config["data"]["protocol_id"],
            data_root=Path(plan["data_root"]),
            evaluation_protocol="v2",
        )
        for role in training["test_roles"]:
            canonical = root / "fit" / f"{role}-test.json"
            if canonical.exists():
                # Finalization checks that a reused receipt belongs to this role.
                continue
            selected = training[role]
            role_config = Path(plan["config"])
            if "training_identity" in selected:
                from gradpert.config import load_experiment_config

                stage = load_experiment_config(role_config).continuation
                if stage is None:
                    raise ValueError("parent checkpoint role lacks continuation provenance")
                role_config = Path(read_json(Path(stage.parent_run_root) / "launch.json")["config"])
            args = argparse.Namespace(
                config=role_config,
                runtime=Path(plan["runtime"]),
                training_run_root=root,
                checkpoint=root / "fit" / selected["file"],
                checkpoint_sha256=selected["sha256"],
                checkpoint_role=role,
                output_root=root / f"postfit-{role}",
                gpu=gpu,
                split="test",
            )
            evaluation_plan = resolve_evaluation_plan(args)
            evaluation_plan["cpu_training_state"] = True
            atomic_json(
                root / "POSTFIT_STATE.json",
                {
                    "stage": "evaluating",
                    "checkpoint_role": role,
                    "training_complete_sha256": sha256_file(root / "TRAIN_COMPLETE.json"),
                    "evaluation_plan": evaluation_plan,
                },
            )
            result = execute_evaluation_plan(evaluation_plan)
            evaluation_identity = {
                **training["identity"],
                "source": evaluation_plan["evaluation_source"],
                "environment": result["evaluation_environment"],
            }
            atomic_json(
                canonical,
                {
                    "identity": {
                        "training": selected.get("training_identity", training["identity"]),
                        "evaluation": evaluation_identity,
                        "checkpoint": selected,
                        "role": role,
                    },
                    "result": result["result"],
                    "independent_evaluation_receipt": str(args.output_root / "COMPLETE.json"),
                    "independent_evaluation_sha256": sha256_file(
                        args.output_root / "COMPLETE.json"
                    ),
                    "worker_runtime_measurements": result["worker_runtime_measurements"],
                },
            )
        complete = finalize_deferred_run(plan, training)
        atomic_json(
            root / "POSTFIT_STATE.json",
            {
                "stage": "complete",
                "wall_seconds": time.perf_counter() - started,
                "test_roles": training["test_roles"],
                "scientific_complete": True,
            },
        )
        return complete
    except BaseException as error:
        atomic_json(
            root / "EVALUATION_FAILURE.json",
            {
                "stage": "evaluation_failed",
                "training_complete": True,
                "scientific_complete": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "training_complete_sha256": sha256_file(root / "TRAIN_COMPLETE.json"),
            },
        )
        raise
