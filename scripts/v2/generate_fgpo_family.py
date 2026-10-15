"""Seal the new FG+PO/node0 anchor and its nine authorized ablations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml
from generate_ffn_five import configurations

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

SCHEMA = "fgpo-node0-medium-1"
FAMILY = "fgpo_node0_20261015"
REFERENCE = ".byte-os/plans/MEDIUM_PERFORMANCE_AND_NEW_B0_20261015.plan.md"
ARMS = {
    "B0": {},
    "VH": {
        "global_min_ratio": 0.8,
        "global_max_ratio": 1.0,
        "local_min_ratio": 0.4,
        "local_max_ratio": 0.65,
    },
    "S12-L4": {
        "ssl1_local_views": 4,
        "ssl2_local_views": 4,
        "ssl1_local_layout": "go_go_string_string",
    },
    "CG1": {"control_conditioned_graph": True},
    "U24": {"gene_conditioned_readout": True},
    "MR1": {"masked_response_ratio": 0.25, "lambda_masked_response": 1.0},
    "P1": {"population_response": True, "max_conditions": 1},
    "C1": {"perturbation_injection": "entry"},
    "O1": {"random_gene_order": False, "fixed_gene_order": True},
    "K2-fixed-prefix": {
        "random_gene_order": False,
        "fixed_gene_order": True,
        "all_kda_prefix": True,
    },
}


def parameter(value):
    return {"value": value, "source": "user_locked", "reference": REFERENCE}


def configs(source: Path, microbatch: int = 100) -> dict:
    if type(microbatch) is not int or microbatch < 2:
        raise ValueError("common microbatch must be an integer >=2")
    base = configurations(source, eval_batch=128)["FG"]
    fixed = {
        "prediction_parameterization": "softplus",
        "ssl1_node": 0.0,
        "microbatch": microbatch,
        "ssl1_local_views": 2,
        "ssl2_local_views": 2,
        "ssl1_local_layout": "go_string",
    }
    for key, value in fixed.items():
        base["model"]["parameters"][key] = parameter(value)
    base["training"]["train_batch_size"] = parameter(4 * microbatch)
    result = {}
    for name, changes in ARMS.items():
        value = copy.deepcopy(base)
        for key, setting in changes.items():
            value["model"]["parameters"][key] = parameter(setting)
        result[name] = value
    return result


def metadata(microbatch: int) -> dict:
    return {
        "schema": SCHEMA,
        "family": FAMILY,
        "priority": list(ARMS),
        "preflight_priority": [
            "CG1",
            "S12-L4",
            "VH",
            "K2-fixed-prefix",
            "B0",
            "U24",
            "MR1",
            "P1",
            "C1",
            "O1",
        ],
        "common_microbatch": microbatch,
        "world_size": 2,
        "accumulation": 2,
        "global_batch": 4 * microbatch,
        "epochs": 4,
        "seed": 1,
        "eval_batch": 128,
        "validation": "disabled",
        "selection": "final_epoch",
        "test_roles": ["last"],
        "preflight_updates": 10,
        "eval_capacity_reference_batch": 128,
        "analysis": "same medium basic/deep/gradient lists; declared view recipes resealed",
        "rerun_old_BR_or_officials": False,
    }


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest must remain source contained")
    value = json.loads(path.read_text())
    microbatch = value["common_microbatch"]
    expected = configs(source, microbatch)
    if {k: v for k, v in value.items() if k != "rows"} != metadata(microbatch):
        raise ValueError("new B0 family protocol changed")
    if [r["name"] for r in value["rows"]] != list(ARMS):
        raise ValueError("authorized arm set/order changed")
    for row in value["rows"]:
        target = path.parent / row["name"] / "gradpert_v2/nadig_jurkat.yaml"
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
            raise ValueError("resolved arm config or identity differs")
        load_experiment_config(target)
    return value


def generate(source: Path, output: Path, microbatch: int = 100) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("fresh source-contained family required")
    rows = []
    for name, value in configs(source, microbatch).items():
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
    path.write_text(json.dumps({**metadata(microbatch), "rows": rows}, indent=2) + "\n")
    return verify_manifest(source, path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("configs/v2") / FAMILY)
    p.add_argument("--microbatch", type=int, default=100)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(root, root / args.output, args.microbatch), indent=2))
