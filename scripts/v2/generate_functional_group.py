"""Generate only the five authorized fresh-six cap40 B0/K/A configurations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

PARENT = (
    "configs/v2/cap40_combined_jurkat/"
    "E23_prototypes16384_unit_distillation/gradpert_v2/nadig_jurkat.yaml"
)
PARENT_SHA256 = "7214e3e382232ae8856998544514c54aecd048aeec2ab3cb8f329c7157c08df8"
ARMS = {
    "B0": {},
    "K1": {"relay_passes": 2},
    "K2": {"self_readout": "position"},
    "A1": {"attention_replacement": "softmax"},
    "A2": {"attention_replacement": "retention"},
}
REFERENCE = "docs/experiments/GRADPERT_V2_FUNCTIONAL_B0_K_A_20261006.md"


def generate(source: Path, output: Path, microbatch: int = 68) -> dict:
    parent = source / PARENT
    if sha256_file(parent) != PARENT_SHA256 or microbatch < 1:
        raise ValueError("frozen parent checksum or common microbatch is invalid")
    base = yaml.safe_load(parent.read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, changes in ARMS.items():
        value = copy.deepcopy(base)
        parameters = value["model"]["parameters"]
        for key, item in {
            "validation_mode": "disabled",
            "microbatch": microbatch,
            **changes,
        }.items():
            parameters[key] = {
                "value": item,
                "source": "user_locked" if key != "microbatch" else "project_preregistered",
                "reference": REFERENCE,
            }
        training = value["training"]
        training["formal_run_policy"] = "v2_fixed_6"
        training["max_epochs"] = {"value": 6, "source": "user_locked", "reference": REFERENCE}
        training["monitor"], training["monitor_mode"] = "none", "none"
        training["train_batch_size"] = {
            "value": microbatch * 4,
            "source": "project_preregistered",
            "reference": REFERENCE,
        }
        path = output / name / "gradpert_v2/nadig_jurkat.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        load_experiment_config(path)
        rows.append(
            {
                "name": name,
                "config": str(path.relative_to(source)),
                "sha256": sha256_file(path),
                "changes": changes,
            }
        )
    manifest = {
        "schema": "gradpert-v2-functional-b0-ka-1",
        "parent": PARENT,
        "parent_sha256": PARENT_SHA256,
        "rows": rows,
        "epochs": 6,
        "validation": "disabled",
        "test_roles": ["last"],
        "common_microbatch": microbatch,
        "global_batch": microbatch * 4,
        "seed": 1,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=68)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(source, args.output.resolve(), args.microbatch), indent=2))


if __name__ == "__main__":
    main()
