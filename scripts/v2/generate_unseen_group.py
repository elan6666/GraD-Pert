"""Three independent unseen-expression mechanisms on the frozen L0 protocol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

PARENT = "configs/v2/cap40_loss_ablations_20261007/L0/gradpert_v2/nadig_jurkat.yaml"
PARENT_SHA256 = "705d962aa1182a6ccedd99096d2af2c3b0531a61dc62bb36d1fea974c4a9322c"
REFERENCE = "docs/experiments/GRADPERT_V2_UNSEEN_GENE_ABLATIONS_20261008.md"
FIELDS = ("prior_shared_adapter", "gene_conditioned_readout", "direct_target_flag")
ARMS = {"U1": (True, False, False), "U2": (False, True, False), "U3": (False, False, True)}
RAW_PRIOR = "/data/yilangliu/DinoGenePT/data/embeddings/seed-go-protein-pathway-master-aligned.npz"
RAW_SHA = "34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318"
FIELDS_FOUR = (*FIELDS, "learned_genept_projection")
ARMS_FOUR = {
    "U1": (True, False, False, False),
    "U2": (False, True, False, False),
    "U4": (False, False, False, True),
    "U3": (False, False, True, False),
}


def generate(source: Path, output: Path, *, with_u4: bool = False) -> dict:
    if sha256_file(source / PARENT) != PARENT_SHA256:
        raise ValueError("frozen L0 baseline checksum changed")
    base = yaml.safe_load((source / PARENT).read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, settings in (ARMS_FOUR if with_u4 else ARMS).items():
        changes = dict(zip(FIELDS_FOUR if with_u4 else FIELDS, settings, strict=True))
        value = copy.deepcopy(base)
        value["model"]["parameters"].update(
            {
                key: {"value": setting, "source": "project_preregistered", "reference": REFERENCE}
                for key, setting in changes.items()
            }
        )
        if name == "U4":
            for key, setting in (("genept_artifact_path", RAW_PRIOR), ("genept_sha256", RAW_SHA)):
                value["model"]["parameters"][key] = {
                    "value": setting,
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
        "schema": "gradpert-v2-unseen-four-1" if with_u4 else "gradpert-v2-unseen-three-1",
        "parent": PARENT,
        "parent_sha256": sha256_file(source / PARENT),
        "reference_training_commit": "ab022caa57a3b45dc5a14c6ae38bc280702112e4",
        "reference_comparison": "cross_version_disabled_path_parity_verified",
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
    parser.add_argument("--with-u4", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    print(json.dumps(generate(source, args.output.resolve(), with_u4=args.with_u4), indent=2))


if __name__ == "__main__":
    main()
