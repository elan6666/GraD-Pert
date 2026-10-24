"""Seal the R6-parent CR necessity ablation; reuse all completed controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_representation_round6 as previous
import generate_representation_round7 as dose
import representation_family as sealed

SCHEMA = "representation-round8-cr0-medium-1"
FAMILY = "representation_round8_20261024"
REFERENCE = ".byte-os/plans/representation-round8-cr0-20261024.md"
ARMS = ("R8-CR0",)
CHANGES = {**previous.CHANGES, "cr_weight": 0.0}
BASELINE_SOURCE = "3d61cb0b50f66535023bf276a7cd04f69adafc3b"
PARENT_TRAINING_SOURCE = dose.PARENT_TRAINING_SOURCE
PARENT_CONFIG = dose.PARENT_CONFIG
PARENT_CONFIG_SHA256 = dose.PARENT_CONFIG_SHA256
PARENT_RUN = dose.PARENT_RUN
PARENT_CHECKPOINT_SHA256 = dose.PARENT_CHECKPOINT_SHA256
PANEL_SHA256 = previous.PANEL_SHA256


def configs(source: Path) -> dict:
    value = previous.configs(source)["R6-EMA99"]
    value["model"]["parameters"]["cr_weight"] = {
        "value": 0.0,
        "source": "project_preregistered",
        "reference": REFERENCE,
    }
    return {"R8-CR0": value}


def metadata() -> dict:
    value = previous.metadata()
    value.update(
        schema=SCHEMA,
        family=FAMILY,
        priority=list(ARMS),
        preflight_priority=list(ARMS),
        reused_controls=[*value["reused_controls"], "R6-EMA99", "R7-S1D05"],
    )
    value["schedule_semantics"]["comparison"] = (
        "original R6 parent; only cr_weight changes from 0.00004 to 0; not a R7-parent ablation"
    )
    value["research_iteration"].update(
        round=8,
        stage_review_after_round=8,
        hypothesis=(
            "removing global CR expansion may improve R6 prediction while retaining Teacher "
            "condition signal and view stability; local gradient evidence does not ensure utility"
        ),
    )
    value["coefficient_decision"] = {
        "weights": CHANGES,
        "parent_arm": "R6-EMA99",
        "intervention": {"cr_weight": {"previous": 0.00004, "value": 0.0}},
        "unchanged_lambda1": 1.0,
        "unchanged_lambda2": {"value": 1.0, "source": "user_locked"},
        "evidence": {
            "parent_training_commit": PARENT_TRAINING_SOURCE,
            "parent_config": PARENT_CONFIG,
            "parent_config_sha256": PARENT_CONFIG_SHA256,
            "parent_run_id": PARENT_RUN,
            "parent_checkpoint_role": "last",
            "parent_checkpoint_epoch": 2,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "panel_sha256": PANEL_SHA256,
            "panel_rows": 128,
            "medium_population_sha256": (
                "afa9c95fa26dc19167e9919bc09c86a8d99f9446466de95a1efd613b1f93bd86"
            ),
            "validation_summary_sha256": (
                "9c29ffc5eb7556fc9e758ba8b1107a647ef732aea67f7f55755649a032dac320"
            ),
            "research_evidence_sha256": (
                "1f26707932b2bd65d52d2933255d584c1e529a2fbd04689eb171827ce285bcc6"
            ),
            "r7_dose_counterexample": {
                "training_commit": BASELINE_SOURCE,
                "validation_mse": 0.006258086035934465,
                "pearson_below_r6": 14,
                "pearson_count": 18,
                "interpretation": (
                    "lower SSL1 dose did not restore prediction; R7 is not the parent"
                ),
            },
            "full_population_observation": {
                "analysis_commit": PARENT_TRAINING_SOURCE,
                "batch_sha256": (
                    "705c0f5137e34e58f932124cb2cb7ba7ecf5d7be056aa33f6c015806a547cb55"
                ),
                "observed_populations_per_endpoint": 1,
                "original_planned_populations": 8,
                "population_size": 320,
                "updates": 0,
                "replay_tolerance": 2e-6,
                "scope": "single-GPU frozen population; not DDP update equivalence",
                "r6_graph_cr_mse_cosine": -0.3724035124119608,
                "r5_cr0_total_mse_cosine_counterexample": [0.9653325039478091, 0.5703735062490288],
                "limits": "R6 response direction also did not improve; no universal CR harm claim",
            },
        },
    }
    value["teacher_tracking"].update(
        parent={"teacher_start": 0.99, "teacher_end": 0.99},
        candidate={"teacher_start": 0.99, "teacher_end": 0.99},
        center_temperature_prototypes="unchanged R6 rules",
        training_views="native R6 views",
        native_source_baseline_commit=BASELINE_SOURCE,
        native_source_git_tree_path="src",
    )
    value["coding_rate"].update(
        enabled=False,
        training_weight=0.0,
        partial_diagnostics="disabled CR has no history observation; missing measurements stay NA",
        runtime_protocol="coding_rate absent; alignment_uniformity retained unchanged",
    )
    value["loss_reporting"] = (
        "retain original SSL losses and actual weighted contributions; disabled CR history "
        "must not be fabricated as zero-valued observations; "
        "joint across CR weights is not comparable; "
        "global negative CR is not whole-versus-within-class MCR2"
    )
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
