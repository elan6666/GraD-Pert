"""Seal exactly BR/F1/N1 with restored functional B0 and four-epoch last tests."""

import argparse
import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

PARENT = "configs/v2/cap40_functional_b0_ka/B0/gradpert_v2/nadig_jurkat.yaml"
PARENT_SHA256 = "8961c93fd42bf0141722a42e23a3577c9209e5cdfb9aea1e8bba3c5de92467a5"
REFERENCE = "docs/experiments/GRADPERT_V2_GLOBAL_FFN_DESIGN_20261011.md"
PRIORITY = ("BR", "F1", "N1")
ARMS = {"BR": {}, "F1": {"ffn_variant": "swiglu"}, "N1": {"ffn_input_norm": "rmsnorm"}}


def configs(source: Path, microbatch: int) -> dict:
    if (
        type(microbatch) is not int
        or microbatch < 2
        or sha256_file(source / PARENT) != PARENT_SHA256
    ):
        raise ValueError("frozen restored B0 parent or common microbatch changed")
    base = yaml.safe_load((source / PARENT).read_text())
    params = base["model"]["parameters"]
    for key, value in {
        "mlp_profile": "restored",
        "ffn_clip": False,
        "ffn_variant": "mixed",
        "ffn_input_norm": "historical",
        "genept_hidden_width": 4096,
        "genept_middle_width": 1024,
        "learned_genept_projection": True,
        "genept_projection_activation": "gelu",
        "microbatch": microbatch,
        "world_size": 2,
        "accumulation": 2,
    }.items():
        params[key] = {"value": value, "source": "user_locked", "reference": REFERENCE}
    train = base["training"]
    train["formal_run_policy"] = "v2_fixed_4"
    train["max_epochs"] = {"value": 4, "source": "user_locked", "reference": REFERENCE}
    for key, value in (("train_batch_size", 4 * microbatch), ("eval_batch_size", 128)):
        train[key] = {"value": value, "source": "project_preregistered", "reference": REFERENCE}
    result = {}
    for name, changes in ARMS.items():
        result[name] = copy.deepcopy(base)
        for key, value in changes.items():
            result[name]["model"]["parameters"][key] = {
                "value": value,
                "source": "user_locked",
                "reference": REFERENCE,
            }
    return result


def metadata(microbatch):
    return {
        "schema": "ffn-four-1",
        "parent": PARENT,
        "parent_sha256": PARENT_SHA256,
        "world_size": 2,
        "accumulation": 2,
        "common_microbatch": microbatch,
        "global_batch": 4 * microbatch,
        "eval_batch": 128,
        "epochs": 4,
        "validation": "disabled",
        "test_roles": ["last"],
        "seed": 1,
        "preflight_updates": 10,
        "priority": list(PRIORITY),
    }


def verify_manifest(source: Path, path: Path) -> dict:
    source, path = source.resolve(), path.resolve()
    if not path.is_relative_to(source):
        raise ValueError("manifest escaped source")
    manifest = json.loads(path.read_text())
    expected = configs(source, manifest["common_microbatch"])
    if {k: v for k, v in manifest.items() if k != "rows"} != metadata(
        manifest["common_microbatch"]
    ):
        raise ValueError("FFN protocol changed")
    if [r["name"] for r in manifest["rows"]] != list(PRIORITY):
        raise ValueError("FFN arm order changed")
    for row in manifest["rows"]:
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
            raise ValueError("FFN config/hash or unrelated setting changed")
        load_experiment_config(target)
    return manifest


def generate(source: Path, output: Path, microbatch=48):
    source, output = source.resolve(), output.resolve()
    if not output.is_relative_to(source) or output.exists():
        raise ValueError("fresh config directory required")
    output.mkdir(parents=True)
    rows = []
    for name, value in configs(source, microbatch).items():
        p = output / name / "gradpert_v2/nadig_jurkat.yaml"
        p.parent.mkdir(parents=True)
        p.write_text(yaml.safe_dump(value, sort_keys=False))
        load_experiment_config(p)
        rows.append(
            {
                "name": name,
                "config": str(p.relative_to(source)),
                "sha256": sha256_file(p),
                "changes": ARMS[name],
            }
        )
    manifest = {**metadata(microbatch), "rows": rows}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return verify_manifest(source, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=48)
    args = parser.parse_args()
    print(json.dumps(generate(Path(__file__).resolve().parents[2], args.output, args.microbatch)))
