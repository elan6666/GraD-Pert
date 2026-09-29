"""Condition-sharded v2 evaluation must recover the single-worker macro result."""

import copy
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.execution import v2_checkpoint_eval
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.v2.evaluation import merge_evaluation_shards

METRICS = (
    "txpert_macro_pearson_delta_all",
    "txpert_macro_pearson_delta_deg",
    "trishift_pearson_delta_all",
    "trishift_pearson_delta_deg",
    "systema_pearson_all",
    "systema_pearson_deg",
)


def shard(condition, value):
    metrics = {
        "condition_id": condition,
        "results": [
            {"metric_id": name, "value": value, "reason": None, "gene_count": 2} for name in METRICS
        ],
    }
    return {
        "split": "test",
        "prediction_loss": value,
        "selection_gene_ids": None,
        "control_manifest_sha256": "controls",
        "reference_sha256": "reference",
        "query_recipe": {"name": "ordered_disjoint_blocks", "query_count": 1000},
        "metric_gene_groups": {
            "seen_expression": {
                "gene_count": 2,
                "gene_ids_sha256": "seen",
                "metrics": [],
                "unavailable_reason": None,
            },
            "unseen_expression": {
                "gene_count": 0,
                "gene_ids_sha256": "unseen",
                "metrics": [],
                "unavailable_reason": "empty_expression_group",
            },
        },
        "metrics": [],
        "conditions": [
            {
                "condition_id": condition,
                "loss": value,
                "metrics": metrics,
                "metric_gene_groups": {"seen_expression": metrics, "unseen_expression": None},
            }
        ],
    }


def test_two_shards_match_one_worker_and_preserve_manifest_order():
    a, b = shard("A", 0.2), shard("B", 0.4)
    single = copy.deepcopy(a)
    single["conditions"].extend(b["conditions"])
    one = merge_evaluation_shards([single], ordered_conditions=("A", "B"))
    two = merge_evaluation_shards([b, a], ordered_conditions=("A", "B"))
    assert one == two
    assert two["prediction_loss"] == pytest.approx(0.3)
    assert two["metrics"][0]["macro_mean"] == pytest.approx(0.3)
    assert two["metric_gene_groups"]["unseen_expression"]["metrics"] == []


def test_shards_reject_overlap_or_missing_conditions():
    a, b = shard("A", 0.2), shard("B", 0.4)
    with pytest.raises(ValueError, match="duplicated"):
        merge_evaluation_shards([a, a], ordered_conditions=("A", "B"))
    with pytest.raises(ValueError, match="do not cover"):
        merge_evaluation_shards([a], ordered_conditions=("A", "B"))
    b["reference_sha256"] = "other"
    with pytest.raises(ValueError, match="reference_sha256"):
        merge_evaluation_shards([a, b], ordered_conditions=("A", "B"))


def test_child_parent_best_evaluation_uses_checkpoint_training_identity(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    parent_config = (
        repository
        / "configs/v2/no_mhc_joint_eval_jurkat/three_epoch_m74_a2/gradpert_v2/nadig_jurkat.yaml"
    )
    child_config = (
        repository
        / "configs/v2/no_mhc_joint_eval_jurkat/continue_full_m74_a2/gradpert_v2/nadig_jurkat.yaml"
    )
    run = tmp_path / "child"
    (run / "fit").mkdir(parents=True)
    checkpoint = run / "fit/parent-epoch-0002.pt"
    checkpoint.write_bytes(b"sealed-parent-checkpoint")
    digest = sha256_file(checkpoint)
    parent_identity = {
        "resolved_config_sha256": sha256_json(
            load_experiment_config(parent_config).model_dump(mode="json")
        )
    }
    child_identity = {"resolved_config_sha256": "child-config-hash"}
    atomic_json(run / "run_manifest.json", child_identity)
    atomic_json(
        run / "fit/epoch_state.json",
        {
            "identity": child_identity,
            "best": {
                "file": checkpoint.name,
                "sha256": digest,
                "epoch": 2,
                "training_identity": parent_identity,
            },
            "last": None,
        },
    )
    publication = tmp_path / "publication.json"
    publication.write_text("{}")
    data = tmp_path / "data"
    data.mkdir()
    runtime = tmp_path / "runtime.json"
    atomic_json(
        runtime,
        {
            "publication_receipt": str(publication),
            "publication_sha256": sha256_file(publication),
            "data_root": str(data),
        },
    )
    monkeypatch.setattr(v2_checkpoint_eval, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr("gradpert.execution.train_entry.repository_root", lambda: repository)
    monkeypatch.setattr(
        v2_checkpoint_eval,
        "inspect_source_identity",
        lambda *args, **kwargs: SimpleNamespace(
            payload=lambda: {"repository_root": str(repository)}
        ),
    )
    args = Namespace(
        config=parent_config,
        training_run_root=run,
        checkpoint=checkpoint,
        output_root=tmp_path / "new-evaluation",
        checkpoint_sha256=digest,
        runtime=runtime,
        gpu="0",
        split="test",
    )
    plan = v2_checkpoint_eval.resolve_evaluation_plan(args)
    assert plan["training_identity"] == parent_identity
    with pytest.raises(ValueError, match="configuration differs"):
        v2_checkpoint_eval.resolve_evaluation_plan(
            Namespace(**{**vars(args), "config": child_config})
        )
