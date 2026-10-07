"""Generate the six matched, fresh-six Jurkat prediction/reconstruction arms."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

PARENT = "configs/v2/cap40_functional_b0_ka/B0/gradpert_v2/nadig_jurkat.yaml"
PARENT_SHA256 = "8961c93fd42bf0141722a42e23a3577c9209e5cdfb9aea1e8bba3c5de92467a5"
REFERENCE = "docs/experiments/GRADPERT_V2_LOSS_ABLATIONS_20261007.md"
FIELDS = (
    "prediction_error_power",
    "prediction_reduction_override",
    "auxiliary_mask_ratio",
    "lambda_gene_mask",
    "lambda_cls_mask",
)
ARMS = {
    "L0": (2, "row_mean", 0.0, 0.0, 0.0),
    "L1": (4, "row_mean", 0.0, 0.0, 0.0),
    "L2": (2, "condition_mean", 0.0, 0.0, 0.0),
    "M2": (2, "row_mean", 0.2, 1.0, 1.0),
    "L3": (4, "condition_mean", 0.0, 0.0, 0.0),
    "M1": (2, "row_mean", 0.2, 1.0, 0.0),
}


def generate(source: Path, output: Path) -> dict:
    if sha256_file(source / PARENT) != PARENT_SHA256:
        raise ValueError("loss baseline checksum changed")
    base = yaml.safe_load((source / PARENT).read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, settings in ARMS.items():
        value = copy.deepcopy(base)
        changes = dict(zip(FIELDS, settings, strict=True))
        value["model"]["parameters"].update(
            {
                key: {"value": setting, "source": "project_preregistered", "reference": REFERENCE}
                for key, setting in changes.items()
            }
        )
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
        "schema": "gradpert-v2-loss-six-1",
        "parent": PARENT,
        "parent_sha256": PARENT_SHA256,
        "parent_source_commit": "3d3f5ad2d6b831baed1eead45d61862fdea2db33",
        "rows": rows,
        "epochs": 6,
        "validation": "disabled",
        "test_roles": ["last"],
        "common_microbatch": 68,
        "global_batch": 272,
        "seed": 1,
        "primary_metric": "txpert_macro_pearson_delta_deg",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(source, args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
