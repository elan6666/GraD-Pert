"""Seal a single constant-EMA follow-up to R5; reuse completed controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_representation_round5 as previous
import representation_family as sealed

SCHEMA = "representation-round6-ema99-medium-1"
FAMILY = "representation_round6_20261024"
REFERENCE = ".byte-os/plans/representation-round6-ema99-20261024.md"
ARMS = ("R6-EMA99",)
CHANGES = {**previous.CHANGES, "teacher_end": 0.99}
BASELINE_SOURCE = "493e295dbd8654876ac4da36710cdea3c400fc57"
PARENT_TRAINING_SOURCE = "0a8631f64844614ab72a4e050f70e3acd0a2150a"
PARENT_CONFIG = "configs/v2/representation_round5_20261023/R5-PROJ/gradpert_v2/nadig_jurkat.yaml"
PARENT_CONFIG_SHA256 = "ea5aa47982b1c1cd80b4f1725f234215c3db46915a913e14f327f0dcf3053411"
PARENT_RUN = "nadig_jurkat-seed1-20261023T154921Z-2a5c68d79b164f3aa99335c737e0a632"
PARENT_CHECKPOINT_SHA256 = "19252c4c1464284be13c88c743305cbc31de903ce5d72fb96211e85d149ed901"
PANEL_SHA256 = "b0b4d6e09ca40ab40cb51d881cfaaeb00d23d71d832bf45b0d1f82de45a1913b"


def configs(source: Path) -> dict:
    value = previous.configs(source)["R5-PROJ"]
    value["model"]["parameters"]["teacher_end"] = {
        "value": 0.99,
        "source": "project_preregistered",
        "reference": REFERENCE,
    }
    return {"R6-EMA99": value}


def metadata() -> dict:
    value = previous.metadata()
    value.update(
        schema=SCHEMA,
        family=FAMILY,
        priority=list(ARMS),
        preflight_priority=list(ARMS),
        reused_controls=["R3-RI", "R3-AUB", "R3-AUS", "R4-CR", "R5-PROJ"],
    )
    value["schedule_semantics"].update(
        ema=(
            "constant 0.99 at every successful optimizer update; unchanged native cosine "
            "formula with equal teacher_start and teacher_end"
        ),
        comparison="R5 two-epoch protocol; only teacher_end changes from 1.0 to 0.99",
    )
    value["research_iteration"].update(
        round=6,
        authorized_rounds=None,
        continuation_authority=(
            "user-authorized ongoing representation-resolution loop, reaffirmed 2026-10-24; "
            "continue from evidence until resolved or explicitly paused, without expanding scope"
        ),
        stage_review_after_round=6,
        hypothesis=(
            "retaining Teacher tracking through the short protocol may improve "
            "input-specific targets while preserving view consistency and prediction utility"
        ),
    )
    value["coefficient_decision"] = {
        "weights": CHANGES,
        "parent_arm": "R5-PROJ",
        "intervention": {"teacher_end": {"previous": 1.0, "value": 0.99}},
        "unchanged_teacher_start": 0.99,
        "rationale": (
            "clean-Global zero-update probes retain weak Teacher-encoder-path signal; "
            "constant EMA is one falsifiable tracking intervention, not a causal diagnosis"
        ),
        "updates": 0,
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
            "native_signal": {
                "analysis_commit": "256ebb45c3bd4892d0b2199c04e83bfc9b692195",
                "complete_sha256": (
                    "cc6c99daa1892140ef9f7149ea5e738c8d0be386adc52ae6cadb568e6c66fe55"
                ),
                "input_mode": "native_student_masked_teacher_clean",
            },
            "clean_global_signal": {
                "analysis_commit": BASELINE_SOURCE,
                "complete_sha256": (
                    "4ce41195f37fdf303c473eadbdd9c5cc5a017f78969b443b3b8899aa35508383"
                ),
                "teacher_signal_sha256": (
                    "cf6fa4b6a6b53fc1dbda39acf13e01e24d7b63d51b44d9513f4d68df33a82974"
                ),
                "input_mode": "clean_globals",
                "state_unchanged": True,
                "updates": 0,
            },
            "parameter_lag_audit_sha256": (
                "dbfe4cc2e53729988cefe867a124219fb9c368aa91f9423c1f6d3307feb45a81"
            ),
            "schedule_arithmetic_sha256": (
                "c0af057af9172a05aa51c13c3c64b059ec8cefe3e4d5dbf9549b6c62c8b02d15"
            ),
            "scope": "train-only frozen diagnostics and arithmetic; no new training result",
        },
    }
    value["teacher_tracking"] = {
        "parent": {"teacher_start": 0.99, "teacher_end": 1.0},
        "candidate": {"teacher_start": 0.99, "teacher_end": 0.99},
        "implementation": "existing lifecycle cosine and successful-step commit_statistics",
        "center_temperature_prototypes": "unchanged R5 rules",
        "training_views": "native R5 views; clean-Global intervention is diagnostic-only",
        "native_source_baseline_commit": BASELINE_SOURCE,
        "native_source_git_tree": "baa481e0ae0d99a0d5830bbdc7a9f8409741f4fe",
        "limits": [
            "frozen encoder/head swaps do not uniquely identify EMA as the cause",
            "EMA recurrence coefficients are not fractions of biological information",
            "one seed and two epochs do not establish general resolution",
            "lower entropy, CE or higher rank alone cannot satisfy scientific acceptance",
            "new source/config/checkpoint identities are required; never overwrite R5",
        ],
    }
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
