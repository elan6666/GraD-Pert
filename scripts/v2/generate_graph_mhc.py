"""One graph-mHC arm against the immutable restored BR four-epoch protocol."""

import argparse
import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

PARENT = "configs/v2/ffn_four_20261011/BR/gradpert_v2/nadig_jurkat.yaml"
PARENT_SHA256 = "075331f51d2ff5699ce0f22386d756ed5b4ee80cce29ee10727cf676c47e62eb"
REFERENCE = "docs/experiments/GRADPERT_V2_GRAPH_MHC_20261011.md"
PRIORITY = ("GM1",)
CHANGES = {"graph_mhc": True}


def configuration(source: Path) -> dict:
    if sha256_file(source / PARENT) != PARENT_SHA256:
        raise ValueError("frozen BR parent changed")
    value = copy.deepcopy(yaml.safe_load((source / PARENT).read_text()))
    value["model"]["parameters"]["graph_mhc"] = {
        "value": True,
        "source": "user_locked",
        "reference": REFERENCE,
    }
    return value


def metadata() -> dict:
    return {
        "schema": "graph-mhc-four-1",
        "parent": PARENT,
        "parent_sha256": PARENT_SHA256,
        "parent_training_commit": "b37b171fd68cb29e5c8142c66778aac6da9a6f06",
        "world_size": 2,
        "accumulation": 2,
        "common_microbatch": 48,
        "global_batch": 192,
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
    if {k: v for k, v in manifest.items() if k != "rows"} != metadata():
        raise ValueError("GM1 protocol changed")
    target = path.parent / "GM1/gradpert_v2/nadig_jurkat.yaml"
    if manifest["rows"] != [
        {
            "name": "GM1",
            "config": str(target.relative_to(source)),
            "sha256": sha256_file(target),
            "changes": CHANGES,
        }
    ] or yaml.safe_load(target.read_text()) != configuration(source):
        raise ValueError("GM1 must differ from sealed BR only by graph_mhc")
    load_experiment_config(target)
    return manifest


def generate(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if not output.is_relative_to(source) or output.exists():
        raise ValueError("fresh source-contained config directory required")
    value = configuration(source)
    path = output / "GM1/gradpert_v2/nadig_jurkat.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    load_experiment_config(path)
    manifest = {
        **metadata(),
        "rows": [
            {
                "name": "GM1",
                "config": str(path.relative_to(source)),
                "sha256": sha256_file(path),
                "changes": CHANGES,
            }
        ],
    }
    target = output / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return verify_manifest(source, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(Path(__file__).resolve().parents[2], args.output)))
