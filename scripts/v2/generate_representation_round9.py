"""Seal one no-MSE SSL epoch and two continuing R6 joint epochs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_representation_round6 as previous
import generate_representation_round7 as provenance
import representation_family as sealed

SCHEMA = "representation-round9-staged-joint-medium-1"
FAMILY = "representation_round9_20261024"
REFERENCE = ".byte-os/plans/representation-round9-staged-joint-20261024.md"
ARMS = ("R9-SSL1J2",)
BASELINE_SOURCE = "19b266ce43c63868478894753232c403814f2172"
CHANGES = {**previous.CHANGES, "training_mode": "ssl_then_joint", "ssl_pretrain_epochs": 1}
ENDPOINTS = {
    "pretrain": {"role": "pretrain", "epoch": 1, "training_stage": "ssl_pretrain"},
    "last": {"role": "last", "epoch": 3, "training_stage": "joint"},
}
PARENT_CONFIG = provenance.PARENT_CONFIG
PARENT_CONFIG_SHA256 = provenance.PARENT_CONFIG_SHA256
PARENT_TRAINING_SOURCE = provenance.PARENT_TRAINING_SOURCE
PARENT_RUN = provenance.PARENT_RUN
PARENT_CHECKPOINT_SHA256 = provenance.PARENT_CHECKPOINT_SHA256


def configs(source: Path) -> dict:
    value = previous.configs(source)["R6-EMA99"]
    value["training"]["formal_run_policy"] = "v2_fixed_3"
    value["training"]["max_epochs"] = dict(value=3, source="user_locked", reference=REFERENCE)
    for key, setting in (("training_mode", "ssl_then_joint"), ("ssl_pretrain_epochs", 1)):
        value["model"]["parameters"][key] = dict(
            value=setting, source="project_preregistered", reference=REFERENCE
        )
    return {ARMS[0]: value}


def metadata() -> dict:
    value = previous.metadata()
    value.update(
        schema=SCHEMA,
        family=FAMILY,
        priority=list(ARMS),
        preflight_priority=list(ARMS),
        epochs=3,
        diagnostic_endpoints=ENDPOINTS,
        reused_controls=[*value["reused_controls"], "R6-EMA99", "R7-S1D05", "R8-CR0"],
        scientific_complete=False,
    )
    value["schedule_semantics"].update(
        lr="independent endpoint cosine prefix150 and joint300; joint300 equals original R6",
        comparison="450 total updates (+50%); 300 prediction-supervised updates as original R6",
        sampler_epochs=[0, 0, 1],
        view_rng="continuous_checkpointed_not_baseline_replay",
        teacher_centers_optimizer="continuous; no reset at boundary",
        auxiliary_warmup="global completed-update clock; once at start",
    )
    value["research_iteration"].update(
        round=9,
        stage_review_after_round=9,
        hypothesis=(
            "an SSL prefix may mature Teacher geometry before prediction; utility remains required"
        ),
    )
    value["coefficient_decision"] = dict(
        weights=CHANGES,
        parent_arm="R6-EMA99",
        unchanged_lambda1=1.0,
        unchanged_lambda2=dict(value=1.0, source="user_locked"),
        evidence=dict(
            parent_training_commit=PARENT_TRAINING_SOURCE,
            parent_config=PARENT_CONFIG,
            parent_config_sha256=PARENT_CONFIG_SHA256,
            parent_run_id=PARENT_RUN,
            parent_checkpoint_role="last",
            parent_checkpoint_epoch=2,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        ),
    )
    value["staged_joint"] = dict(
        mode="ssl_then_joint",
        pretrain_epochs=1,
        joint_epochs=2,
        pretrain_steps=150,
        joint_steps=300,
        total_steps=450,
        prefix_prediction="disabled; untrained readout is diagnostic only",
        prefix_regularizers="R6 SSL1+SSL2+A=.05/U=0/CR=.00004; explicit design interpretation",
        final_validation_conditions=445,
        final_validation_metrics=18,
        dual_endpoint_medium="one physical manifest",
        dual_endpoint_panel="one fixed32x4 panel",
        total_budget_matched=False,
        baseline_initialization=False,
    )
    value["teacher_tracking"].update(
        parent={"teacher_start": 0.99, "teacher_end": 0.99},
        native_source_baseline_commit=BASELINE_SOURCE,
        training_views="native R6 views; continuous checkpointed RNG",
        center_temperature_prototypes="unchanged R6, continuously updated across450 steps",
    )
    value["teacher_tracking"].pop("native_source_git_tree", None)
    value["teacher_tracking"]["native_source_git_tree_path"] = "src"
    value["teacher_tracking"]["limits"] = [
        "450 total updates versus R6 300; not budget matched",
        "one seed; prefix rank or lower CE alone does not establish scientific success",
        "global auxiliary warmup is continuous; joint start dose differs from R6",
    ]
    return value


def verify_manifest(source: Path, path: Path) -> dict:
    return sealed.verify(source, path, metadata(), configs(source), CHANGES)


def generate(source: Path, output: Path) -> dict:
    return sealed.generate(source, output, metadata(), configs(source), CHANGES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/v2") / FAMILY)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(root, root / args.output), indent=2))
