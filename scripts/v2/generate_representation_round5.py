"""Move the existing A/CR auxiliary losses to the existing SSL2 bottleneck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import generate_representation_round4 as previous
import representation_family as sealed

SCHEMA = "representation-round5-projected-cr-medium-1"
FAMILY = "representation_round5_20261023"
REFERENCE = ".byte-os/plans/representation-round5-projector-20261023.md"
ARMS = ("R5-PROJ",)
CHANGES = {**previous.CHANGES, "population_aux_space": "ssl2_cls_bottleneck"}


def configs(source: Path) -> dict:
    value = previous.configs(source)["R4-CR"]
    value["model"]["parameters"]["population_aux_space"] = {
        "value": "ssl2_cls_bottleneck",
        "source": "project_preregistered",
        "reference": REFERENCE,
    }
    return {"R5-PROJ": value}


def metadata() -> dict:
    value = previous.metadata()
    value.update(
        schema=SCHEMA,
        family=FAMILY,
        priority=list(ARMS),
        preflight_priority=list(ARMS),
        reused_controls=["R3-RI", "R3-AUB", "R3-AUS", "R4-CR"],
    )
    value["research_iteration"].update(
        round=5,
        stage_review_after_round=5,
        hypothesis=(
            "existing projection bottleneck may absorb auxiliary geometry "
            "while retaining prediction utility"
        ),
    )
    value["coefficient_decision"] = {
        "weights": CHANGES,
        "rationale": (
            "retain R4 numeric coefficients; a space intervention, not matched gradient dose"
        ),
        "frozen_path_audit_source": "ae2ce0098416c9f6767d696183b6368cdf85e475",
        "frozen_path_audit_run": "representation-response-path-ae2ce00-20261023-a1",
        "updates": 0,
        "panel_sha256": "b0b4d6e09ca40ab40cb51d881cfaaeb00d23d71d832bf45b0d1f82de45a1913b",
    }
    value["coding_rate"]["representation"] = (
        "existing Student ssl2_cls.features 256-dimensional bottleneck before L2 and prototypes"
    )
    value["coding_rate"]["limitations"] = [
        "ordinary projector retains backbone gradients; no gradient isolation",
        "same coefficients do not imply matched gradient dose across spaces",
        "projector-only high rank is not task-relevant representation success",
        "full-population CR can reward context differences; not official SimDINO replication",
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
