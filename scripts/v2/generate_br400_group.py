"""Four fresh global400 arms, validation-joint selection and both checkpoint roles."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

SCHEMA = "br400-best-last-1"
PRIORITY = ("BR", "F1", "N1", "GM1")
PREFLIGHT_PRIORITY = ("GM1", "F1", "BR", "N1")
CHECKPOINT_PREFLIGHT_PRIORITY = ("F1", "GM1", "BR", "N1")
REFERENCE = ".byte-os/plans/GRADPERT_V2_BR400_BEST_LAST_20261012.plan.md"
PARENTS = {
    "BR": (
        "configs/v2/ffn_four_20261011/BR/gradpert_v2/nadig_jurkat.yaml",
        "075331f51d2ff5699ce0f22386d756ed5b4ee80cce29ee10727cf676c47e62eb",
    ),
    "F1": (
        "configs/v2/ffn_four_20261011/F1/gradpert_v2/nadig_jurkat.yaml",
        "837c280829398902da3140c59f3a21b106d87bdf7bc72501ee92a6a219b2dc0b",
    ),
    "N1": (
        "configs/v2/ffn_four_20261011/N1/gradpert_v2/nadig_jurkat.yaml",
        "40dbd308706ef86b92e0ea779a6514a8d0469cb3da3fc421eb0e9dec8730885e",
    ),
    "GM1": (
        "configs/v2/graph_mhc_four_20261011/GM1/gradpert_v2/nadig_jurkat.yaml",
        "7b03765950313dba5dacd8067725fa62b1583d03e5f5ffdf96759346a1380ba3",
    ),
}


def configurations(source: Path, eval_batch: int = 300, checkpoint_genept: bool = False) -> dict:
    if type(checkpoint_genept) is not bool:
        raise ValueError("checkpoint_genept must be an explicit bool")
    if type(eval_batch) is not int or not 2 <= eval_batch <= 300:
        raise ValueError("eval cell batch must be an explicit common 2..300")
    result = {}
    for name, (relative, digest) in PARENTS.items():
        path = source / relative
        if sha256_file(path) != digest:
            raise ValueError("frozen scientific parent changed")
        value = copy.deepcopy(yaml.safe_load(path.read_text()))
        params, train = value["model"]["parameters"], value["training"]
        for key, setting in (
            ("microbatch", 100),
            ("world_size", 2),
            ("accumulation", 2),
            ("validation_mode", "joint_only"),
        ):
            params[key] = {"value": setting, "source": "user_locked", "reference": REFERENCE}
        if checkpoint_genept:
            params["checkpoint_genept_mapping"] = {
                "value": True,
                "source": "project_preregistered",
                "reference": REFERENCE,
            }
        for key, setting in (
            ("train_batch_size", 400),
            ("eval_batch_size", eval_batch),
            ("max_epochs", 4),
        ):
            train[key] = {"value": setting, "source": "user_locked", "reference": REFERENCE}
        train.update(formal_run_policy="v2_fixed_4", monitor="val/joint_loss", monitor_mode="min")
        result[name] = value
    return result


def metadata(eval_batch: int = 300, checkpoint_genept: bool = False) -> dict:
    if type(checkpoint_genept) is not bool:
        raise ValueError("checkpoint_genept must be an explicit bool")
    result = {
        "schema": SCHEMA,
        "parents": {
            name: {"config": relative, "sha256": digest}
            for name, (relative, digest) in PARENTS.items()
        },
        "priority": list(PRIORITY),
        "preflight_priority": list(
            CHECKPOINT_PREFLIGHT_PRIORITY if checkpoint_genept else PREFLIGHT_PRIORITY
        ),
        "common_microbatch": 100,
        "world_size": 2,
        "accumulation": 2,
        "global_batch": 400,
        "epochs": 4,
        "seed": 1,
        "eval_batch": eval_batch,
        "eval_capacity_reference_batch": 128,
        "validation": "joint_only",
        "selection": "minimum_validation_joint_loss",
        "test_roles": ["best", "last"],
        "preflight_updates": 10,
        "execution": "sequential_dual_fit_then_dual_best_last",
    }
    if checkpoint_genept:
        result["checkpoint_genept"] = True
    return result


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest must be source-contained")
    manifest = json.loads(path.read_text())
    checkpoint_genept = manifest.get("checkpoint_genept", False)
    expected = configurations(source, manifest["eval_batch"], checkpoint_genept)
    if {k: v for k, v in manifest.items() if k != "rows"} != metadata(
        manifest["eval_batch"], checkpoint_genept
    ):
        raise ValueError("BR400 common protocol changed")
    if [row["name"] for row in manifest["rows"]] != list(PRIORITY):
        raise ValueError("BR400 arm scope/order changed")
    for row in manifest["rows"]:
        config = path.parent / row["name"] / "gradpert_v2/nadig_jurkat.yaml"
        if (
            row
            != {
                "name": row["name"],
                "config": str(config.relative_to(source)),
                "sha256": sha256_file(config),
            }
            or yaml.safe_load(config.read_text()) != expected[row["name"]]
        ):
            raise ValueError("BR400 config/hash or scientific setting changed")
        load_experiment_config(config)
    return manifest


def generate(
    source: Path, output: Path, eval_batch: int = 300, checkpoint_genept: bool = False
) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(source):
        raise ValueError("fresh source-contained configuration directory required")
    rows = []
    for name, config in configurations(source, eval_batch, checkpoint_genept).items():
        path = output / name / "gradpert_v2/nadig_jurkat.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        rows.append(
            {"name": name, "config": str(path.relative_to(source)), "sha256": sha256_file(path)}
        )
    path = output / "manifest.json"
    path.write_text(
        json.dumps({**metadata(eval_batch, checkpoint_genept), "rows": rows}, indent=2) + "\n"
    )
    return verify_manifest(source, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-batch", type=int, default=300)
    parser.add_argument("--checkpoint-genept", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                Path(__file__).resolve().parents[2],
                args.output,
                args.eval_batch,
                args.checkpoint_genept,
            )
        )
    )
