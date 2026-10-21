"""Seal the first two-epoch representation study; later rounds remain undesigned."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

SCHEMA = "representation-round1-medium-1"
FAMILY = "representation_round1_20261021"
REFERENCE = "scripts/v2/generate_representation_round1.py#research-protocol"
BASELINE = "configs/v2/fgpo_node0_20261016_b320/B0/gradpert_v2/nadig_jurkat.yaml"
BASELINE_SHA256 = "80f5ee1d6c4d826d9d2c023f304c08f1734b5a05acd28d565f7053e367117830"
ARMS = {"RS-B0": {}, "RS-RI": {"response_injection_residual": True}}


def configs(source: Path) -> dict:
    """Copy the sealed B0 file, without a runtime defaults/inheritance chain."""
    path = source / BASELINE
    if sha256_file(path) != BASELINE_SHA256:
        raise ValueError("frozen B0 anchor changed")
    anchor = yaml.safe_load(path.read_text())
    values = {}
    for name, changes in ARMS.items():
        value = copy.deepcopy(anchor)
        value["training"]["formal_run_policy"] = "v2_fixed_2"
        value["training"]["max_epochs"] = {
            "value": 2,
            "source": "user_locked",
            "reference": REFERENCE,
        }
        for key, setting in changes.items():
            value["model"]["parameters"][key] = {
                "value": setting,
                "source": "project_preregistered",
                "reference": REFERENCE,
            }
        values[name] = value
    return values


def metadata() -> dict:
    return {
        "schema": SCHEMA,
        "family": FAMILY,
        "priority": list(ARMS),
        "preflight_priority": list(ARMS),
        "common_microbatch": 80,
        "world_size": 2,
        "accumulation": 2,
        "global_batch": 320,
        "epochs": 2,
        "seed": 1,
        "eval_batch": 128,
        "validation": "disabled",
        "postfit_validation": True,
        "test_timing": "after_four_round_design_lock",
        "selection": "final_epoch",
        "test_roles": ["last"],
        "preflight_updates": 10,
        "eval_capacity_reference_batch": 128,
        "initialization": "from_scratch; common B0 parameter initialization and seed",
        "analysis": "same sealed medium population plus a separate fixed-panel training probe",
        "baseline": {"config": BASELINE, "sha256": BASELINE_SHA256},
        "schedule_semantics": {
            "lr": "unchanged endpoint warmup cosine, warmup_fraction=0.16, over two epochs",
            "ema": "unchanged cosine 0.99 to 1.0 over two epochs",
            "comparison": (
                "both arms share the shorter relative schedule; historical epoch2 "
                "is not this anchor"
            ),
        },
        "fixed_panel_diagnostics": {
            "split": "train",
            "conditions": 32,
            "controls_per_condition": 4,
            "query_genes": 1000,
            "deep_gene_sample": 128,
            "panel": (
                "one fixed allowed training-expression panel shared by all arms and conditions"
            ),
            "pairing": ["same_condition_different_control", "different_condition_same_control"],
            "stratification": ["cell_line", "cell_type", "context_id"],
            "identity": (
                "reuse sealed medium condition/row identities where eligible; "
                "seal the extra probe separately"
            ),
            "independence": (
                "deduplicate controls; report actual valid counts; do not replace "
                "missing rows silently"
            ),
            "stages": (
                "Response injection input, branch and actual state; self/cross/FFN; final norm"
            ),
            "state_updates": False,
            "persist_full_activations": False,
        },
        "research_iteration": {
            "round": 1,
            "authorized_rounds": 4,
            "future_rounds": "design only after deep analysis of the preceding round",
            "design_evidence": ["train", "validation"],
            "test_results_for_selection": False,
            "success": (
                "joint representation separation and response utility; not CE or DINO rank alone"
            ),
        },
        "rerun_old_BR_or_officials": False,
    }


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest must remain source contained")
    value = json.loads(path.read_text())
    if {k: v for k, v in value.items() if k != "rows"} != metadata():
        raise ValueError("round1 two-epoch research protocol changed")
    if [r["name"] for r in value["rows"]] != list(ARMS):
        raise ValueError("only RS-B0 then RS-RI are defined for round1")
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
            raise ValueError("round1 must preserve B0 and change only the RI residual switch")
        load_experiment_config(target)
    return value


def generate(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("fresh source-contained family required")
    rows = []
    for name, value in configs(source).items():
        path = output / name / "gradpert_v2/nadig_jurkat.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        rows.append(
            {
                "name": name,
                "config": str(path.relative_to(source)),
                "sha256": sha256_file(path),
                "changes": ARMS[name],
            }
        )
    path = output / "manifest.json"
    path.write_text(json.dumps({**metadata(), "rows": rows}, indent=2) + "\n")
    return verify_manifest(source, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("configs/v2/representation_round1_20261021")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(root, root / args.output), indent=2))
