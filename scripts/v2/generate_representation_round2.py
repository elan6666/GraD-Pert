"""Second adaptive round: distinguish CLS-head statistics from backbone dispersion."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml
from generate_representation_round1 import configs as first_configs
from generate_representation_round1 import metadata as first_metadata

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

SCHEMA = "representation-round2-medium-1"
FAMILY = "representation_round2_20261022"
REFERENCE = "scripts/v2/generate_representation_round2.py#head-versus-backbone"
ARMS = {
    "R2-RI": {},
    "R2-HN": {"ssl2_cls_input_layernorm": True},
    "R2-KL": {"ssl2_koleo": 0.1},
}


def configs(source: Path) -> dict:
    anchor = first_configs(source)["RS-RI"]
    values = {}
    for name, changes in ARMS.items():
        value = copy.deepcopy(anchor)
        for key, setting in changes.items():
            value["model"]["parameters"][key] = {
                "value": setting,
                "source": "project_preregistered",
                "reference": REFERENCE,
            }
        values[name] = value
    return values


def metadata() -> dict:
    value = first_metadata()
    value.update(
        schema=SCHEMA,
        family=FAMILY,
        priority=list(ARMS),
        preflight_priority=["R2-KL", "R2-RI", "R2-HN"],
    )
    value["test_timing"] = "after_research_design_lock"
    value["research_iteration"].update(
        round=2,
        authorized_rounds=None,
        stage_review_after_round=4,
        continuation="until task-relevant representation problem is resolved",
        future_rounds=(
            "adaptive after deep analysis; four rounds are a review, not automatic success"
        ),
        original_anchor="representation_round1_20261021/RS-B0",
        repeated_candidate="R2-RI repeats RS-RI with the same two-epoch schedule",
        hypothesis=(
            "separate CLS projector entry conditioning from a missing backbone dispersion "
            "pressure; neither sharp probabilities nor higher rank alone is success"
        ),
    )
    value["koleo"] = {
        "weight": 0.1,
        "source": "project_preregistered",
        "global_views_reduction": "mean of two Global losses; not official sum scaling",
        "population": "complete differentiably gathered two-GPU effective cell population",
        "exclude_same_condition": True,
        "context_restriction": False,
        "risk": "can amplify context or condition identity without improving biological response",
    }
    value["head_norm"] = {
        "scope": "Student and EMA Teacher SSL2 CLS projector input only",
        "type": "FP32 LayerNorm, epsilon1e-6, trainable scale/bias",
        "other_heads": "unchanged; iBOT has no added input norm",
    }
    return value


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest must remain source contained")
    value = json.loads(path.read_text())
    if {k: v for k, v in value.items() if k != "rows"} != metadata():
        raise ValueError("round2 registered research protocol changed")
    if [r["name"] for r in value["rows"]] != list(ARMS):
        raise ValueError("round2 requires RI, head norm and separate KoLeo arms")
    expected = configs(source)
    for row in value["rows"]:
        target = path.parent / row["name"] / "gradpert_v2/nadig_jurkat.yaml"
        if not target.resolve().is_relative_to(source):
            raise ValueError("configuration must remain source contained")
        if (
            row
            != {
                "name": row["name"],
                "config": str(target.relative_to(source)),
                "sha256": sha256_file(target),
                "changes": ARMS[row["name"]],
            }
            or yaml.safe_load(target.read_text()) != expected[row["name"]]
        ):
            raise ValueError("round2 must preserve RI and change only its registered axis")
        load_experiment_config(target)
    return value


def generate(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("fresh source-contained family required")
    rows = []
    for name, value in configs(source).items():
        target = output / name / "gradpert_v2/nadig_jurkat.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        rows.append(
            {
                "name": name,
                "config": str(target.relative_to(source)),
                "sha256": sha256_file(target),
                "changes": ARMS[name],
            }
        )
    path = output / "manifest.json"
    path.write_text(json.dumps({**metadata(), "rows": rows}, indent=2) + "\n")
    return verify_manifest(source, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("configs/v2/representation_round2_20261022")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(root, root / args.output), indent=2))
