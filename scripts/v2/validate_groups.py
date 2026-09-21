"""Read-only check of v2 group definitions and capacity-dependent batch levels."""

import argparse
import json
from pathlib import Path


def validate(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data["schema_version"] != "gradpert-v2-groups-1":
        raise ValueError("unknown group schema")
    if set(data["order"]) != set(data["groups"]) or len(data["order"]) != len(set(data["order"])):
        raise ValueError("group order must enumerate every group exactly once")
    if data["groups"]["H3"]["levels"] is not None:
        raise ValueError("design matrix cannot invent measured batch levels")
    if data["groups"]["H3"]["level_source"] != "verified_capacity_receipt":
        raise ValueError("batch levels must come from capacity evidence")
    if data["groups"]["H1"]["levels"] != [0.001, 0.0001]:
        raise ValueError("LR levels differ from user decision")
    if data["fixed"]["heads"] != 4 or data["fixed"]["dropout"] != 0.1:
        raise ValueError("fixed architecture differs from user decision")
    return {
        "status": "design_valid",
        "groups": len(data["groups"]),
        "runnable": False,
        "reason": "Requires capacity receipts and self-contained run configurations",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=Path("experiments/v2/groups.json"))
    args = parser.parse_args()
    print(json.dumps(validate(args.matrix), indent=2))
