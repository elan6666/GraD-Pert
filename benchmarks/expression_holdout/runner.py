"""Server-only official fit and public evaluator, with independent run identities."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np

from benchmarks.common.official_checkout import official_module_session
from gradpert.data._io import atomic_json
from gradpert.evaluation.data import CanonicalEvaluationData
from gradpert.evaluation.state import load_evaluation_state
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file

from .config import (
    load_config,
    require_data_identity,
    require_evaluation_identity,
    validate_preflight,
)
from .data import prepare_data
from .evaluation import evaluate
from .provenance import persist_inference_receipt, verify_environment, verify_evaluation_files

ROOT = Path(__file__).resolve().parents[2]


def restore_last(fitted, model_id: str, path: Path) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if model_id == "scouter":
        fitted.model.network.load_state_dict(payload, strict=True)
    elif model_id == "gears":
        fitted.model.model.load_state_dict(payload, strict=True)
    else:
        # Official scDFM save_checkpoint uses model_state_dict.
        state = payload["model_state_dict"]
        fitted.model.vf.load_state_dict(state, strict=True)


def run(
    config_path: Path,
    runtime_path: Path,
    output: Path,
    *,
    mode: str,
    preflight_path: Path | None = None,
) -> dict:
    config = load_config(config_path)
    runtime = json.loads(runtime_path.read_text())
    for value in [output, Path(runtime["data_root"]), Path(runtime["official_checkout"])]:
        if not value.resolve().is_relative_to("/data/yilangliu"):
            raise ValueError("official runs and scientific artifacts are server-only")
    if output.exists():
        raise FileExistsError("new official behavior requires a fresh run ID")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator contract missing")
    source = inspect_source_identity(
        ROOT,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=Path(runtime["publication_receipt"]),
        expected_publication_receipt_sha256=runtime["publication_sha256"],
    )
    if runtime["source_commit"] != source.commit:
        raise ValueError("runtime/source version differs")
    environment_identity = verify_environment(runtime)
    evaluation_identity = verify_evaluation_files(config, Path(runtime["data_root"]))
    identity = {
        **environment_identity,
        **evaluation_identity,
        "source_commit": source.commit,
        "source_clean": not source.dirty,
        "official_commit": config["source_code"]["commit"],
        "config_sha256": sha256_file(config_path),
        "runtime_sha256": sha256_file(runtime_path),
        "run_id": output.name,
        "checkpoint_role": "last",
    }
    if mode == "full":
        if preflight_path is None:
            raise ValueError("full fit requires an accepted actual official preflight")
        validate_preflight(
            json.loads(preflight_path.read_text()),
            config=config_path,
            source_commit=source.commit,
            official_commit=identity["official_commit"],
            runtime_sha256=identity["runtime_sha256"],
        )
    import torch

    if runtime["device"].startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(float(runtime["cuda_memory_fraction"]), 0)
    output.mkdir(parents=True)
    atomic_json(output / "RUN.json", identity | {"mode": mode, "scientific_complete": False})
    started = time.monotonic()
    try:
        prepared = prepare_data(config, Path(runtime["data_root"]))
        require_data_identity(prepared, config)
        atomic_json(output / "data-receipt.json", prepared.receipt)
        adapter = importlib.import_module(f"benchmarks.expression_holdout.{config['model_id']}")
        names = {"scouter": ("scouter",), "gears": ("gears", "gears.gears")}
        import contextlib

        session = (
            official_module_session(
                checkout_root=runtime["official_checkout"],
                expected_commit=identity["official_commit"],
                module_names=names[config["model_id"]],
            )
            if config["model_id"] in names
            else contextlib.nullcontext()
        )
        with session:
            fitted = adapter.fit(
                prepared,
                config,
                runtime,
                output / "fit",
                max_updates=10 if mode == "preflight" else None,
            )
            fitted.receipt.update(identity)
            atomic_json(output / "fit-receipt.json", fitted.receipt)
            checkpoint = output / "fit/last.pt"
            identity["checkpoint_sha256"] = sha256_file(checkpoint)
            restore_last(fitted, config["model_id"], checkpoint)
            ev = config["evaluation"]
            with CanonicalEvaluationData(
                dataset_id=config["data"]["dataset_id"],
                protocol_id=config["data"]["protocol_id"],
                split_name="test",
                data_root=runtime["data_root"],
            ) as data:
                reference = load_evaluation_state(
                    dataset_id=config["data"]["dataset_id"],
                    protocol_id=config["data"]["protocol_id"],
                    data_root=runtime["data_root"],
                    evaluation_protocol="v2",
                )
                require_evaluation_identity(data, reference, config)

                def predictor(*args, **kwargs):
                    return adapter.predict(fitted, *args, **kwargs)

                if mode == "preflight":
                    draw = data.control_manifest.draws[0]
                    controls = data.load_control_rows(tuple(draw.ordered_row_ids))
                    targets = tuple(g for g in draw.condition_id.split("+") if g != "ctrl")
                    result = predictor(
                        controls.expression,
                        targets,
                        tuple(data.expression_gene_ids),
                        cell_batch=ev["cell_batch"],
                        query_count=ev["query_count"],
                        seed=config["seed"],
                    )
                    finite = result.shape == controls.expression.shape and np.isfinite(result).all()
                    if not finite:
                        raise ValueError("restored official checkpoint inference is invalid")
                    completed = fitted.receipt.get(
                        "completed_updates", fitted.receipt.get("diagnostic_updates_completed")
                    )
                    if completed != 10:
                        raise ValueError(
                            "official diagnostic must complete exactly10 optimizer updates"
                        )
                    receipt = identity | {
                        "schema": "official-expression-holdout-preflight-v1",
                        "status": "passed",
                        "optimizer_updates": completed,
                        "restored_checkpoint_predictions_finite": True,
                        "controls_per_condition": 300,
                        "zero_pkl": not list(output.rglob("*.pkl")),
                        "data_identity": prepared.receipt,
                        "control_manifest_sha256": data.control_manifest_file_sha256,
                        "evaluation_state_sha256": reference.manifest_file_sha256,
                        "scientific_complete": False,
                    }
                else:
                    seen = tuple(
                        fitted.receipt.get(
                            "observed_training_expression_gene_indices",
                            fitted.receipt.get("seen_expression_gene_indices", ()),
                        )
                    )

                    def progress(number, total, condition):
                        atomic_json(
                            output / "progress.json",
                            {
                                "phase": "test",
                                "condition": condition,
                                "completed": number,
                                "total": total,
                                "percent": 100 * number / total,
                            },
                        )

                    result = evaluate(
                        data,
                        reference,
                        predictor,
                        seen_indices=seen,
                        cell_batch=ev["cell_batch"],
                        query_count=ev["query_count"],
                        seed=config["seed"],
                        progress=progress,
                    )
                    result.update(identity)
                    atomic_json(output / "last-test.json", result)
                    receipt = identity | {
                        "schema": "official-expression-holdout-complete-v1",
                        "status": "complete",
                        "scientific_complete": True,
                        "last_test_sha256": sha256_file(output / "last-test.json"),
                        "data_identity": prepared.receipt,
                        "control_manifest_sha256": result["control_manifest_sha256"],
                        "evaluation_state_sha256": result["evaluation_state_sha256"],
                        "zero_pkl": not list(output.rglob("*.pkl")),
                    }
                receipt.update(persist_inference_receipt(output, fitted, identity))
                if not receipt["zero_pkl"]:
                    raise ValueError("zero-PKL postcondition violated")
                receipt["wall_seconds"] = time.monotonic() - started
                atomic_json(output / "receipt.json", receipt)
                if mode == "full":
                    atomic_json(output / "COMPLETE.json", receipt)
                return receipt
    except BaseException as error:
        atomic_json(
            output / "FAILURE.json",
            identity
            | {
                "error_type": type(error).__name__,
                "error": str(error),
                "scientific_complete": False,
            },
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "runtime", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--mode", choices=["preflight", "full"], required=True)
    p.add_argument("--preflight", type=Path)
    args = p.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.runtime,
                args.output,
                mode=args.mode,
                preflight_path=args.preflight,
            ),
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
