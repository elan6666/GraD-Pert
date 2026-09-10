"""Post-fit best/final test evaluation; never fits or selects from test scores."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from gradpert.config import ExperimentConfig, NativeArchitectureOptions, load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.evaluation import CanonicalEvaluationData
from gradpert.execution.artifact_run import seal_evaluation_outputs
from gradpert.execution.identity import inspect_environment, inspect_source_identity
from gradpert.graphs import GraphTopology, build_prediction_graph_view
from gradpert.hashing import sha256_file, sha256_json
from gradpert.modeling import GraDPertJointModel
from gradpert.pilots import load_vnext_graph_topology
from gradpert.training.data import CanonicalTrainingData
from gradpert.training.inference import predict_frozen_controls


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def load_frozen_config(small: Path) -> ExperimentConfig:
    """Use the original self-contained path; require its sealed copy to match."""
    meta = read_json(small / "run_meta.json")
    manifest = read_json(small / "run_manifest.json")
    original = Path(meta["config_path"]).resolve(strict=True)
    expected = manifest["config_sha256"]
    if sha256_file(original) != expected or sha256_file(small / "config.resolved.yaml") != expected:
        raise ValueError("original/resolved config hash differs from the sealed training config")
    return load_experiment_config(original)


def checkpoint_progress(
    checkpoint: dict[str, Any], manifest: dict[str, Any], meta: dict[str, Any]
) -> int:
    """Reject foreign, test-consumed or non-epoch checkpoint states."""
    identity = checkpoint["identity"]
    for key in (
        "source_commit",
        "config_sha256",
        "environment_sha256",
        "canonical_data_sha256",
        "split_content_sha256",
    ):
        if identity[key] != manifest[key]:
            raise ValueError(f"checkpoint identity differs: {key}")
    if identity["source_tree_sha256"] != meta["source"]["tree_sha256"]:
        raise ValueError("checkpoint training source tree differs")
    progress = checkpoint["progress"]
    epochs = int(progress["completed_epochs"])
    if (
        checkpoint["schema_version"] != "gradpert-training-checkpoint-v2"
        or progress["test_evaluations"] != 0
        or not 1 <= epochs <= meta["max_epochs"]
        or progress["global_step"] != epochs * meta["steps_per_epoch"]
    ):
        raise ValueError("invalid checkpoint epoch/test lifecycle")
    return epochs


def verify_existing(root: Path) -> dict[str, Any]:
    """A complete output is reusable only with all recorded small hashes intact."""
    receipt = read_json(root / "COMPLETE.json")
    for relative, digest in receipt["output_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError("post-fit output hash differs")
    if list(root.rglob("*.pkl")) or list(root.rglob(".result-work-*")):
        raise ValueError("post-fit outputs require zero persistent PKL/work")
    return receipt


def evaluate_best_last(
    *,
    training_root: Path,
    output_root: Path,
    data_root: Path,
    repository_root: Path,
    publication: Path,
    publication_sha256: str,
    device_name: str,
    archived_last: Path | None = None,
    memory_fraction: float | None = None,
) -> dict[str, Any]:
    """Evaluate frozen checkpoints on the canonical test split in separate roots.

    Training evidence is read-only. Missing historical last is recorded, not
    regenerated. If best is the final epoch, last aliases its identical weights.
    A started but incomplete evaluation is never automatically retried.
    """
    training_root = training_root.resolve(strict=True)
    output_root = output_root.resolve()
    small = training_root / "small_results"
    config_path = small / "config.resolved.yaml"
    config = load_frozen_config(small)
    manifest = read_json(small / "run_manifest.json")
    meta = read_json(small / "run_meta.json")
    selection = read_json(small / "selection_receipt.json")
    if (
        config.training.formal_run_policy != "r50_selection"
        or config.artifacts.result_mode != "metrics_only"
        or manifest["status"] != "trained"
        or manifest["test_evaluations"] != 0
        or not manifest["formal_eligible"]
        or manifest["source_dirty"]
        or manifest["config_sha256"] != sha256_file(config_path)
        or selection["status"] != "complete"
        or selection["epochs_completed"] != 50
        or meta["max_epochs"] != 50
        or selection["optimizer_steps"] != 50 * meta["steps_per_epoch"]
        or selection["test_evaluations"] != 0
        or selection["run_manifest_sha256"] != sha256_json(manifest)
    ):
        raise ValueError("post-fit requires a sealed full R50 selection run")
    training_manifest_sha = sha256_file(small / "run_manifest.json")
    if output_root.exists():
        receipt = verify_existing(output_root)
        if receipt["training_manifest_sha256"] != training_manifest_sha:
            raise ValueError("post-fit destination belongs to another training run")
        return receipt
    if output_root.is_relative_to(training_root) or training_root.is_relative_to(output_root):
        raise ValueError("post-fit outputs must be separate from immutable training roots")
    if (
        device_name.startswith("cuda")
        and os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
    ):
        raise RuntimeError("post-fit CUDA requires expandable_segments:True")
    source = inspect_source_identity(
        repository_root,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=publication,
        expected_publication_receipt_sha256=publication_sha256,
        remote_ref="refs/heads/main",
    )
    environment = inspect_environment(repository_root, device_name=device_name)
    if device_name.startswith("cuda"):
        from gradpert.execution.system_resources import shared_gpu_budget

        free, total = torch.cuda.mem_get_info(torch.device(device_name))
        reserve = 4096 * 1024**2
        budget = shared_gpu_budget(free, total, reserve, memory_fraction)
        torch.cuda.set_per_process_memory_fraction(budget / total)
    architecture = NativeArchitectureOptions.from_parameters(config.model.parameters)
    if architecture.graph_axis_policy != "recomputed_hvg_union_candidate_targets":
        raise ValueError("post-fit R50 currently requires the sealed vNext HVG graph")
    relative = Path(str(config.model.parameters["runtime_graph_root"].value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe graph root")
    graph_root = data_root / relative
    topology, graph_manifest = load_vnext_graph_topology(graph_root)
    topology = GraphTopology(
        gene_ids=topology.gene_ids,
        sources=topology.sources,
        active_sources=architecture.graph_sources,
    )
    if (
        sha256_json(list(topology.gene_ids)) != meta["runtime_graph_gene_order_sha256"]
        or graph_manifest.canonical_data_sha256 != manifest["canonical_data_sha256"]
        or graph_manifest.split_content_sha256 != manifest["split_content_sha256"]
    ):
        raise ValueError("post-fit graph/data identity differs")
    best = training_root / "checkpoints/best.pt"
    if sha256_file(best) != manifest["best_checkpoint_sha256"]:
        raise ValueError("best checkpoint hash differs")
    paths = {"best": best}
    final = archived_last or training_root / "checkpoints/last.pt"
    if final.is_file():
        paths["last"] = final
    output_root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output_root / "STARTED.json",
        {
            "training_manifest_sha256": training_manifest_sha,
            "evaluation_source": source.payload(),
            "policy": "best_and_last_test_once",
        },
    )
    roles: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    best_epoch = 0
    with CanonicalTrainingData(
        dataset_id=config.dataset_id,
        protocol_id=config.data.protocol_id,
        data_root=data_root,
        run_seed=manifest["run_seed"],
        graph_gene_ids_override=topology.gene_ids,
        graph_manifest_path_override=graph_root / "manifest.json",
    ) as train:
        if (
            train.manifest.canonical_adata_sha256 != manifest["canonical_data_sha256"]
            or train.split.split_content_sha256 != manifest["split_content_sha256"]
        ):
            raise ValueError("live canonical identity differs from training")
        for role, path in paths.items():
            digest = sha256_file(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            epoch = checkpoint_progress(payload, manifest, meta)
            if role == "last" and epoch != 50:
                raise ValueError(
                    "archived last is not the final epoch; never label partial as last"
                )
            if role == "best":
                best_epoch = epoch
            if digest in hashes:
                roles[role] = {
                    "status": "alias",
                    "alias_of": hashes[digest],
                    "epoch": epoch,
                    "checkpoint_sha256": digest,
                }
                continue
            role_root = output_root / role
            role_root.mkdir()
            atomic_json(
                role_root / "TEST_STARTED.json", {"checkpoint_sha256": digest, "epoch": epoch}
            )
            # Construction values are overwritten by strict state restoration;
            # no reinitialization of learned E3 weights is used for inference.
            prior = None
            if architecture.gene_feature_mode != "learned_id":
                prior = torch.ones(
                    len(topology.gene_ids), meta["genept_feature"]["embedding_width"]
                )
            model = GraDPertJointModel(
                graph_gene_count=len(topology.gene_ids),
                expression_gene_count=train.manifest.n_expression_genes,
                prototype_count=int(config.model.parameters["prototype_count"].value),
                architecture=architecture,
                genept_matrix=prior,
            )
            model.load_state_dict(payload["model"], strict=True)
            del payload
            model.to(torch.device(device_name))
            with CanonicalEvaluationData(
                dataset_id=config.dataset_id,
                protocol_id=config.data.protocol_id,
                split_name="test",
                data_root=data_root,
            ) as test:
                predictions = predict_frozen_controls(
                    model=model,
                    prediction_view=build_prediction_graph_view(topology),
                    control_manifest=test.control_manifest,
                    anchors_by_condition={
                        k: train.anchors_by_condition[k] for k in train.split.test_conditions
                    },
                    load_control_rows=test.load_control_rows,
                    device=torch.device(device_name),
                    decode_batch_size=int(config.training.eval_batch_size.value),
                )
                seal_evaluation_outputs(
                    destination=role_root,
                    config=config,
                    config_sha256=manifest["config_sha256"],
                    run_id=manifest["run_id"] + "/postfit/" + role,
                    run_seed=manifest["run_seed"],
                    source=source,
                    environment=environment,
                    training_data=train,
                    test_data=test,
                    predictions=predictions,
                    checkpoint_sha256=digest,
                )
                roles[role] = {
                    "status": "evaluated",
                    "epoch": epoch,
                    "checkpoint_sha256": digest,
                    "test_evaluations": 1,
                    "control_manifest_sha256": test.control_manifest_file_sha256,
                    "metrics": read_json(role_root / "small_results/metrics_summary.json"),
                }
            del predictions, model
            if device_name.startswith("cuda"):
                torch.cuda.empty_cache()
            if sha256_file(path) != digest:
                raise ValueError("checkpoint changed during evaluation")
            hashes[digest] = role
    if "last" not in roles:
        roles["last"] = (
            {
                "status": "alias",
                "alias_of": "best",
                "epoch": 50,
                "checkpoint_sha256": roles["best"]["checkpoint_sha256"],
            }
            if best_epoch == 50
            else {"status": "unavailable", "reason": "historical_final_checkpoint_not_retained"}
        )
    if list(output_root.rglob("*.pkl")) or list(output_root.rglob(".result-work-*")):
        raise RuntimeError("post-fit successful root must contain zero PKL/work")
    receipt = {
        "schema_version": "r50-postfit-test-v1",
        "status": "complete",
        "training_manifest_sha256": training_manifest_sha,
        "training_source_commit": manifest["source_commit"],
        "evaluation_source": source.payload(),
        "test_used_for_selection": False,
        "roles": roles,
        "output_sha256": {
            str(p.relative_to(output_root)): sha256_file(p) for p in output_root.rglob("*.json")
        },
    }
    atomic_json(output_root / "COMPLETE.json", receipt)
    return verify_existing(output_root)
