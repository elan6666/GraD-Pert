"""Seal one SSL1-dose follow-up to R6 without changing native model code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_representation_round6 as previous
import representation_family as sealed

SCHEMA = "representation-round7-ssl1-dose-medium-1"
FAMILY = "representation_round7_20261024"
REFERENCE = ".byte-os/plans/representation-round7-ssl1-dose-20261024.md"
ARMS = ("R7-S1D05",)
CHANGES = {**previous.CHANGES, "lambda1": 0.05}
BASELINE_SOURCE = "8a78218424802f1fab01be2b8b134d120c91dd71"
PARENT_TRAINING_SOURCE = BASELINE_SOURCE
PARENT_CONFIG = "configs/v2/representation_round6_20261024/R6-EMA99/gradpert_v2/nadig_jurkat.yaml"
PARENT_CONFIG_SHA256 = "ffda3bd855a6eaa2d9417dfb90ca3878a9947c2d4944900a78dd1bac9450bf8a"
PARENT_RUN = "nadig_jurkat-seed1-20261023T210003Z-5435b634f57449a384fe0b2af01ef692"
PARENT_CHECKPOINT_SHA256 = "610eb0c6058870e13eb226cfe037d1899af5841ec452fc2fbbf813eb6ae0ceab"
PANEL_SHA256 = previous.PANEL_SHA256


def configs(source: Path) -> dict:
    value = previous.configs(source)["R6-EMA99"]
    value["model"]["parameters"]["lambda1"] = {
        "value": 0.05,
        "source": "project_preregistered",
        "reference": REFERENCE,
    }
    return {"R7-S1D05": value}


def metadata() -> dict:
    value = previous.metadata()
    value.update(
        schema=SCHEMA,
        family=FAMILY,
        priority=list(ARMS),
        preflight_priority=list(ARMS),
        reused_controls=[*value["reused_controls"], "R6-EMA99"],
    )
    value["schedule_semantics"]["comparison"] = (
        "R6 two-epoch protocol; only lambda1 changes from 1.0 to 0.05"
    )
    value["research_iteration"].update(
        round=7,
        stage_review_after_round=7,
        hypothesis=(
            "reducing amplified shared-graph SSL1 pressure may preserve R6 Teacher signal "
            "while improving prediction utility; gradient scale alone is not causal evidence"
        ),
    )
    value["coefficient_decision"] = {
        "weights": CHANGES,
        "parent_arm": "R6-EMA99",
        "intervention": {"lambda1": {"previous": 1.0, "value": 0.05}},
        "unchanged_lambda2": {"value": 1.0, "source": "user_locked"},
        "rationale": (
            "matched frozen graph probes amplify SSL1 mean gradient norm about 21.93-fold; "
            "0.0120669/0.264641 is about 0.0456, preregistered as one fixed 0.05 dose; "
            "mixed gradient directions do not establish universal conflict"
        ),
        "evidence": {
            "parent_training_commit": PARENT_TRAINING_SOURCE,
            "parent_run_id": PARENT_RUN,
            "parent_config": PARENT_CONFIG,
            "parent_config_sha256": PARENT_CONFIG_SHA256,
            "parent_checkpoint_role": "last",
            "parent_checkpoint_epoch": 2,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "panel_sha256": PANEL_SHA256,
            "panel_rows": 128,
            "medium_population_sha256": (
                "afa9c95fa26dc19167e9919bc09c86a8d99f9446466de95a1efd613b1f93bd86"
            ),
            "parent_medium_summary_sha256": (
                "97210d874344687e89ce448054269b3cdbd7513343691afb40ec0df918d9a0d8"
            ),
            "parent_gradient_probe_sha256": (
                "0d818ff7fa1d5612e881f8584e08742706fbd96c71d214dc8a694a375ba17869"
            ),
            "parent_report_manifest_sha256": (
                "305f60eb171444df92180afa2db305598331f8f4a7cbce1d630851ee4b941f92"
            ),
            "parent_report_sha256": (
                "12323ea9163c91e3b85ad2480fb301c4ebb46a69e3623af2d6503e7a24d84172"
            ),
            "scope": "matched train-only frozen probes; not full-population update gradients",
        },
    }
    value["teacher_tracking"].update(
        parent={"teacher_start": 0.99, "teacher_end": 0.99},
        candidate={"teacher_start": 0.99, "teacher_end": 0.99},
        center_temperature_prototypes="unchanged R6 rules",
        training_views="native R6 views; no clean-Global training intervention",
        native_source_baseline_commit=BASELINE_SOURCE,
        native_source_git_tree_path="src",
        limits=[
            "one fixed coefficient is a falsifiable dose experiment, not GradNorm or PCGrad",
            "gradient scaling at a frozen state does not imply scaled optimizer displacement",
            "one seed and two epochs do not establish general resolution",
            "entropy, CE or rank alone cannot satisfy scientific acceptance",
            "new source/config/checkpoint identities are required; never overwrite R6",
        ],
    )
    value["zero_update_launch_gate"] = {
        "status": "required_before_launch; receipt supplied independently",
        "source_commit": BASELINE_SOURCE,
        "profile": "mechanism-v1",
        "matched_endpoints": ["R5-PROJ:last", "R6-EMA99:last"],
        "complete_populations_per_endpoint": 1,
        "population_size": 320,
        "updates": 0,
        "requires_state_and_rng_unchanged": True,
        "replay_max_abs_error_lte": 2e-6,
        "criterion": "R6/R5 full-population weighted SSL1 graph gradient norm > 2",
        "scope": "bounded one-population observation; not eight-population completion",
        "failure_action": "do not launch or retune; retain evidence and reconsider mechanism",
    }
    value["loss_reporting"] = (
        "report original SSL1/SSL2 losses and weighted contributions separately; "
        "joint values across different lambda1 are not directly comparable; "
        "partial medium A/CR and unavailable metrics remain NA"
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
