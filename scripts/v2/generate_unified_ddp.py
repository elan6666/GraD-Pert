"""Seal the unified MLP arms under the restored two-rank accumulated protocol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml
from generate_unified_group import ARMS, baseline

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

FAMILY = "unified_ddp_20261010"
REFERENCE = ".byte-os/plans/GRADPERT_V2_UNIFIED_DDP_20261010.plan.md"
PRIORITY = ("N0", "CG1", "U24", "MR1", "P1")


def configs(source: Path, microbatch: int) -> dict:
    if type(microbatch) is not int or microbatch < 2:
        raise ValueError("common per-rank microbatch must be at least two")
    base = baseline(source, 32)
    for key, value in (("world_size", 2), ("accumulation", 2), ("microbatch", microbatch)):
        base["model"]["parameters"][key] = {
            "value": value,
            "source": "user_locked",
            "reference": REFERENCE,
        }
    base["training"]["train_batch_size"] = {
        "value": 4 * microbatch,
        "source": "user_locked",
        "reference": REFERENCE,
    }
    result = {}
    for name, changes in ARMS.items():
        value = copy.deepcopy(base)
        for key, setting in changes.items():
            value["model"]["parameters"][key] = {
                "value": setting,
                "source": "user_locked",
                "reference": REFERENCE,
            }
        result[name] = value
    return result


def metadata(microbatch: int) -> dict:
    return {
        "schema": "unified-ddp-1",
        "family": FAMILY,
        "world_size": 2,
        "accumulation": 2,
        "common_microbatch": microbatch,
        "global_batch": 4 * microbatch,
        "eval_batch": 32,
        "epochs": 6,
        "validation": "disabled",
        "test_roles": ["last"],
        "seed": 1,
        "preflight_updates": 10,
        "priority": list(PRIORITY),
        "parent_family": "unified_first_20261010",
        "population_batch": "actual same-condition rows; do not duplicate truths to fill budget",
    }


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest escaped source")
    manifest = json.loads(path.read_text())
    microbatch = manifest["common_microbatch"]
    expected = configs(source, microbatch)
    if {k: v for k, v in manifest.items() if k != "rows"} != metadata(microbatch):
        raise ValueError("DDP protocol or priority changed")
    if [row["name"] for row in manifest["rows"]] != list(ARMS):
        raise ValueError("DDP arm set/order changed")
    for row in manifest["rows"]:
        name = row["name"]
        target = path.parent / name / "gradpert_v2/nadig_jurkat.yaml"
        if (
            row
            != {
                "name": name,
                "config": str(target.relative_to(source)),
                "sha256": sha256_file(target),
                "changes": ARMS[name],
            }
            or yaml.safe_load(target.read_text()) != expected[name]
        ):
            raise ValueError("config, checksum, arm or an unrelated setting changed")
        load_experiment_config(target)
    return manifest


def generate(source: Path, output: Path, microbatch: int = 68) -> dict:
    source, output = source.resolve(), output.resolve()
    if not output.is_relative_to(source) or output.exists():
        raise ValueError("new family must be a fresh directory within source")
    rows = []
    for name, value in configs(source, microbatch).items():
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
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({**metadata(microbatch), "rows": rows}, indent=2) + "\n")
    return verify_manifest(source, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch", type=int, default=68)
    parser.add_argument("--output", type=Path, default=Path(f"configs/v2/{FAMILY}"))
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(source, source / args.output, args.microbatch), indent=2))


if __name__ == "__main__":
    main()
