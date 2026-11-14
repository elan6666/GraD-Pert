"""Prepare, screen and fit the original small model on the frozen K562 fold."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE), str(SOURCE / "scripts/v2")]

import yaml  # noqa: E402

from gradpert.config import load_experiment_config  # noqa: E402
from gradpert.data._io import atomic_json  # noqa: E402
from gradpert.execution.identity import inspect_source_identity  # noqa: E402
from gradpert.execution.train_entry import resolve_plan  # noqa: E402
from gradpert.features.text_prior import verify_text_prior_npz  # noqa: E402
from gradpert.graphs.materialization import DatasetGraphLayout  # noqa: E402
from gradpert.hashing import sha256_file  # noqa: E402
from gradpert.training.v2.crosscell import DATASET, PROTOCOL, seal_inputs  # noqa: E402
from scripts.v2.run_dataset_transfer_300 import native_run  # noqa: E402
from scripts.v2.run_group import lock  # noqa: E402
from scripts.v2.run_small_softmax import admit_selected_idle  # noqa: E402
from scripts.v2.shared_gpu import snapshot  # noqa: E402

ROOT = Path("/data/yilangliu/GraD-Pert")
REPO = "https://github.com/elan6666/GraD-Pert.git"
FRACTION = 0.9


def candidate_config(root: Path, graph: Path, batch: int) -> Path:
    historical = (
        SOURCE
        / "configs/v2/small_softmax_mhc_original_full_dual_10ep"
        / "gradpert_v2/nadig_jurkat.yaml"
    )
    recipe = yaml.safe_load(historical.read_text())
    recipe.update(dataset_id=DATASET, experiment_id=f"gradpert_v2__{DATASET}")
    recipe["data"].update(
        dataset_id=DATASET,
        protocol_id=PROTOCOL,
        preprocessing_id=PROTOCOL,
        canonical_data_root=str(graph.parent.parent),
        split_policy=PROTOCOL,
    )
    recipe["model"]["exclude_test_target_expression"] = False
    parameters = recipe["model"]["parameters"]
    for name, value in {
        "graph_manifest_path": str(graph),
        "graph_manifest_sha256": sha256_file(graph),
        "world_size": 1,
        "microbatch": batch,
    }.items():
        parameters[name] = {
            "value": value,
            "source": "project_preregistered",
            "reference": "scripts/v2/run_crosscell_k562.py",
        }
    recipe["training"]["train_batch_size"] = parameters["microbatch"].copy()
    recipe["training"]["max_epochs"]["reference"] = "User request: cross-cell single-GPU 10 epochs"
    path = root / "configs/experiments/gradpert_v2" / f"{DATASET}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        yaml.safe_dump(recipe, stream, sort_keys=False)
    load_experiment_config(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", required=True, choices=("0", "1"))
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--publication", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(ROOT / "development"):
        raise ValueError("cross-cell run must stay in the server development root")
    for key, expected in {
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": args.gpu,
    }.items():
        if os.environ.get(key) != expected:
            raise ValueError(f"missing fixed execution environment {key}")
    publication_sha = sha256_file(args.publication)
    source = inspect_source_identity(
        SOURCE,
        formal=True,
        expected_repository=REPO,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=publication_sha,
    )
    root.mkdir(exist_ok=False)
    data = root / "data"
    graph = DatasetGraphLayout(data, DATASET, PROTOCOL).manifest
    stage = root / "STATE.json"
    state = {
        "source_commit": source.commit,
        "gpu": args.gpu,
        "gpu_uuid": args.gpu_uuid,
        "stage": "prepare",
        "capacity_trials": [],
    }

    def mark(phase: str, **extra) -> None:
        state.update(stage=phase, **extra)
        atomic_json(stage, state)

    mark("prepare")
    try:
        seal_inputs(
            source=ROOT / "crosscell-fixed-v1/source/K562_cross_cell_lines/de_adata_test.h5ad",
            folds=ROOT / "crosscell-fixed-v1/folds-baadb85/manifest.json",
            checkout=ROOT / "upstreams/txpert",
            registry_path=SOURCE / "registry/graphs/public_string_go.yaml",
            output=graph,
        )
        inputs = json.loads(graph.read_text())
        prior_path = Path(
            "/data/yilangliu/DinoGenePT/data/embeddings/seed-go-protein-pathway-master-aligned.npz"
        )
        prior = verify_text_prior_npz(
            prior_path,
            expected_sha256="34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318",
            expected_gene_ids=tuple(inputs["gene_ids"]),
            perturbation_target_gene_ids=tuple(inputs["gene_ids"]),
        )
        prior_receipt = root / "GENEPT.json"
        atomic_json(
            prior_receipt,
            {
                "path": str(prior_path),
                "sha256": sha256_file(prior_path),
                "gene_count": len(prior.gene_ids),
                "width": prior.embedding_width,
                "graph_manifest_sha256": sha256_file(graph),
            },
        )
        runtime = root / "RUNTIME.json"
        atomic_json(
            runtime,
            {
                "data_root": str(data),
                "runs_root": str(root / "runs"),
                "publication_receipt": str(args.publication.resolve()),
                "publication_sha256": publication_sha,
                "genept_receipt": str(prior_receipt),
                "genept_sha256": sha256_file(prior_receipt),
            },
        )

        def guard() -> None:
            observed = snapshot(os.getpid())
            admit_selected_idle(observed, int(args.gpu), args.gpu_uuid)
            atomic_json(root / "RESOURCE_CURRENT.json", observed)

        def screen(batch: int) -> bool:
            mark("screen", trial_batch=batch)
            attempt = root / "capacity" / f"b{batch}"
            config = candidate_config(attempt, graph, batch)
            output = attempt / "probe"
            command = [
                sys.executable,
                str(SOURCE / "scripts/v2/capacity_probe.py"),
                "--config",
                str(config),
                "--data-root",
                str(data),
                "--output",
                str(output),
                "--gpu",
                args.gpu,
                "--publication",
                str(args.publication),
                "--publication-sha256",
                publication_sha,
                "--benchmark-only",
                "--batch-capacity-only",
                "--steps",
                "3",
            ]
            try:
                guard()
                native_run(command, attempt / "probe.log", {}, admission_check=guard)
            except RuntimeError:
                if not (output / "receipt.json").is_file():
                    raise
                receipt = json.loads((output / "receipt.json").read_text())
                if (
                    receipt.get("batch_capacity_screen", {}).get("failure_class")
                    != "cuda_update_oom"
                ):
                    raise
                ok = False
            else:
                receipt = json.loads((output / "receipt.json").read_text())
                if receipt.get("status") != "passed" or receipt.get("steps_completed") != 3:
                    raise ValueError("incomplete capacity screen")
                ok = True
            state["capacity_trials"].append(
                {
                    "batch": batch,
                    "passed": ok,
                    "receipt": str(output / "receipt.json"),
                    "sha256": sha256_file(output / "receipt.json"),
                    "peak_reserved_bytes": receipt.get("peak_reserved_bytes"),
                    "cells_per_second": receipt.get(
                        "measured_cells_per_second_including_data_wait"
                    ),
                }
            )
            mark("screen")
            return ok

        with lock(ROOT / "runtime" / f"v2-gpu-{args.gpu}.lock"):
            guard()
            low, high = 1, len(inputs["gene_ids"]) + 1
            # First use the nearby Jurkat capacity as a starting point. An OOM
            # brackets only for identical hardware, model and update workload.
            if screen(128):
                low, trial = 128, 144
                while screen(trial):
                    low, trial = trial, trial + 16
                high = trial
            else:
                high = 128
                if screen(96):
                    low = 96
            while high - low > 1:
                trial = (low + high) // 2
                if screen(trial):
                    low = trial
                else:
                    high = trial
            if low < 2:
                raise ValueError("no valid full-batch capacity screen")
            mark("selected", maximum_screened_batch=low, adjacent_oom_batch=high)
        # The existing runner acquires the same lock for each separate stage.
        # Only a verified update OOM can justify lowering the screened batch.
        for batch in range(low, 1, -1):
            attempt = root / f"fit-b{batch}"
            attempt.mkdir()
            config_path = candidate_config(attempt, graph, batch)
            plan = resolve_plan(
                argparse.Namespace(
                    config=config_path,
                    runtime=runtime,
                    data_root=data,
                    gpu=args.gpu,
                    seed=1,
                    defer_test=True,
                )
            )
            plan["cuda_memory_fraction"] = FRACTION
            plan_file = attempt / "PLAN.json"
            atomic_json(plan_file, plan)
            base_command = [
                sys.executable,
                str(SOURCE / "scripts/v2/run_small_softmax.py"),
                "--plan",
                str(plan_file),
                "--plan-sha256",
                sha256_file(plan_file),
                "--gpu-uuid",
                args.gpu_uuid,
            ]
            attempt_identity = {
                "selected_batch": batch,
                "plan": str(plan_file),
                "plan_sha256": sha256_file(plan_file),
                "config_sha256": plan["config_sha256"],
                "run_id": plan["run_id"],
                "run_root": plan["run_root"],
            }
            for phase in ("preflight", "train"):
                mark(phase, **attempt_identity)
                result = subprocess.run([*base_command, "--stage", phase], check=False).returncode
                if result:
                    log = attempt / (phase + ".log")
                    if not log.is_file() or "torch.OutOfMemoryError" not in log.read_text():
                        raise RuntimeError(f"{phase} exited {result}; see {attempt}")
                    mark("retry_cuda_oom", failed_stage=phase, **attempt_identity)
                    break
            else:
                mark("training_complete", **attempt_identity)
                return
        raise RuntimeError("no batch completed the 10-epoch training")
    except BaseException as error:
        mark("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
