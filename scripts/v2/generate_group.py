"""Materialize independent v2 group configs from one explicitly frozen parent."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file

GROUPS = ("B0", "H1", "H2", "H3", "P1", "L0", "L1", "L2", "A1", "A2", "A3", "A4", "S1", "S2", "G1")


def levels(
    group: str,
    batch_levels: list[int] | None = None,
    world_size: int = 1,
    holdout: dict | None = None,
) -> list[tuple[str, dict]]:
    if group == "G1":
        if not holdout or set(holdout) != {"path", "sha256"}:
            raise ValueError("G1 requires a sealed expression holdout")
        path = Path(holdout["path"])
        if sha256_file(path) != holdout["sha256"]:
            raise ValueError("expression holdout checksum mismatch")
        partition = json.loads(path.read_text())
        training, hidden = partition["training_gene_ids"], partition["heldout_gene_ids"]
        if (
            partition["schema_version"]
            not in {"gradpert-v2-expression-holdout-1", "gradpert-v2-expression-holdout-2"}
            or partition["selection"] != "preregistered_uniform_without_expression_values"
            or not training
            or not hidden
            or len(set(training + hidden)) != len(training) + len(hidden)
        ):
            raise ValueError("invalid expression holdout partition")
        return [
            (
                "expression_holdout",
                {
                    "expression_holdout_path": str(path.resolve()),
                    "expression_holdout_sha256": holdout["sha256"],
                },
            )
        ]
    if group == "H3":
        if (
            batch_levels is None
            or len(batch_levels) not in (2, 3)
            or any(type(n) is not int or n < 1 for n in batch_levels)
            or batch_levels != sorted(set(batch_levels))
            or world_size not in (1, 2)
            or any(n % world_size for n in batch_levels)
        ):
            raise ValueError("H3 requires two or three distinct measured batch levels")
        return [
            (f"batch_{n}", {"microbatch": n // world_size, "train_batch_size": n})
            for n in batch_levels
        ]
    if group == "B0":
        return [
            ("prediction_only", {"lambda1": 0, "lambda2": 0}),
            ("joint_ssl", {"lambda1": 1, "lambda2": 0.1}),
        ]
    if group == "H1":
        return [("lr_1e3", {"learning_rate": 0.001}), ("lr_1e4", {"learning_rate": 0.0001})]
    if group == "H2":
        return [("wd_0", {"weight_decay": 0.0}), ("wd_1e5", {"weight_decay": 0.00001})]
    if group == "P1":
        return [(f"width_{width}", {"width": width}) for width in (256, 128)]
    if group == "L0":
        return [
            (f"ssl_{a}{b}", {"lambda1": a, "lambda2": b * 0.1})
            for a, b in itertools.product((0, 1), repeat=2)
        ]
    if group in ("L1", "L2"):
        stage = int(group[-1])
        names = ("condition", "node", "spread") if stage == 1 else ("dino", "ibot", "koleo")
        rows = []
        for bits in itertools.product((0, 1), repeat=3):
            values = {
                "lambda1": 1 if stage == 1 and any(bits) else 0,
                "lambda2": 0.1 if stage == 2 and any(bits) else 0,
            }
            values.update(
                {
                    f"ssl{stage}_{name}": bit * weight
                    for name, bit, weight in zip(names, bits, (0.8, 0.4, 0.1), strict=True)
                }
            )
            rows.append(("components_" + "".join(map(str, bits)), values))
        return rows
    if group == "S2":
        return [
            ("mixed_koleo_off", {"ssl2_koleo": 0}),
            ("single_koleo_off", {"max_conditions": 1, "ssl2_koleo": 0}),
            ("mixed_koleo_on", {"ssl2_koleo": 0.1}),
        ]
    key, values = {
        "A1": ("attention", ("hybrid", "per_gene")),
        "A2": ("attention", ("hybrid", "full_latent", "delta_full", "full")),
        "A3": ("streams", (4, 1)),
        "A4": ("gene_initialization", ("genept", "random")),
        "S1": ("ssl1_reduction", ("condition_mean", "row_mean")),
    }[group]
    return [(str(value), {key: value}) for value in values]


def row_configuration(raw: dict, overrides: dict) -> dict:
    config = copy.deepcopy(raw)
    for key, value in overrides.items():
        owner = (
            config["training"]
            if key in ("learning_rate", "weight_decay", "train_batch_size")
            else config["model"]["parameters"]
        )
        owner[key] = {
            "value": value,
            "source": "project_preregistered",
            "reference": "docs/design/GRADPERT_V2.md",
        }
    if "learning_rate" in overrides:
        schedule = config["training"]["scheduler"]["value"]
        schedule["max_lr"] = overrides["learning_rate"]
        schedule["min_lr"] = 0.2 * overrides["learning_rate"]
    return config


def measured_batches(parent: Path, probes: list[dict]) -> list[int]:
    from capacity_report import collect

    raw = yaml.safe_load(parent.read_text())
    batches = []
    for probe in probes:
        receipt, config = Path(probe["receipt"]), Path(probe["config"])
        if sha256_file(receipt) != probe["receipt_sha256"]:
            raise ValueError("capacity receipt changed")
        observed = collect(receipt, config)
        candidate = yaml.safe_load(config.read_text())
        if (
            load_experiment_config(config).model.excludes_test_target_expression
            != load_experiment_config(parent).model.excludes_test_target_expression
        ):
            raise ValueError("capacity expression visibility differs from parent")
        if (
            observed["world_size"] != raw["model"]["parameters"]["world_size"]["value"]
            or observed["accumulation"] != 1
        ):
            raise ValueError("H3 requires matching-world unaccumulated probes")
        if candidate["dataset_id"] != raw["dataset_id"] or candidate["data"] != raw["data"]:
            raise ValueError("capacity probe dataset differs from parent")
        for key, value in raw["model"]["parameters"].items():
            if key != "microbatch" and candidate["model"]["parameters"].get(key) != value:
                raise ValueError("capacity model or loss profile differs from parent")
        batches.append(observed["effective_batch"])
    levels("H3", batches, raw["model"]["parameters"]["world_size"]["value"])
    return batches


def generate(
    parent: Path,
    group: str,
    output: Path,
    *,
    batch_probes: list[dict] | None = None,
    holdout: dict | None = None,
) -> dict:
    load_experiment_config(parent)
    raw = yaml.safe_load(parent.read_text())
    if raw["model_id"] != "gradpert_v2" or group not in GROUPS:
        raise ValueError("requires a v2 parent and a supported group")
    if group == "S2" and raw["model"]["parameters"]["max_conditions"]["value"] <= 1:
        raise ValueError("S2 needs a mixed-condition parent")
    if output.exists():
        raise FileExistsError("group output must be new; existing experiments are immutable")
    rows = []
    prepared = []
    batch_levels = measured_batches(parent, batch_probes or []) if group == "H3" else None
    if group != "H3" and batch_probes:
        raise ValueError("capacity batch probes apply only to H3")
    if group != "G1" and holdout:
        raise ValueError("expression holdout applies only to G1")
    if group == "G1" and raw["model"]["parameters"].get("expression_holdout_path", {}).get("value"):
        raise ValueError("G1 requires an unrestricted selected parent")
    for name, overrides in levels(
        group, batch_levels, raw["model"]["parameters"]["world_size"]["value"], holdout
    ):
        config = row_configuration(raw, overrides)
        prepared.append((name, config, overrides))
    output.mkdir(parents=True)
    for name, config, overrides in prepared:
        path = output / name / "gradpert_v2" / f"{raw['dataset_id']}.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        load_experiment_config(path)
        rows.append(
            {
                "name": name,
                "config": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "overrides": overrides,
            }
        )
    manifest = {
        "schema_version": "gradpert-v2-group-configs-1",
        "group": group,
        "parent_config": str(parent.resolve()),
        "parent_sha256": sha256_file(parent),
        "generator_sha256": sha256_file(Path(__file__)),
        "status": "configs_valid_pending_capacity_and_launch_preflight",
        "selection_rule": "validation prediction loss only; no within-group rolling parent",
        "rows": rows,
    }
    if group == "G1":
        manifest["expression_holdout"] = holdout
    if group == "H3":
        manifest["batch_probes"] = batch_probes
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_group(manifest_path: Path, parent: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("superseded_reason"):
        raise ValueError("superseded group cannot be launched: " + manifest["superseded_reason"])
    if manifest.get("schema_version") != "gradpert-v2-group-configs-1":
        raise ValueError("unknown group manifest schema")
    if sha256_file(parent) != manifest["parent_sha256"]:
        raise ValueError("frozen group parent checksum mismatch")
    load_experiment_config(parent)
    raw = yaml.safe_load(parent.read_text())
    batch_levels = (
        measured_batches(parent, manifest.get("batch_probes", []))
        if manifest["group"] == "H3"
        else None
    )
    expected = dict(
        levels(
            manifest["group"],
            batch_levels,
            raw["model"]["parameters"]["world_size"]["value"],
            manifest.get("expression_holdout"),
        )
    )
    rows = manifest["rows"]
    if len(rows) != len(expected) or {r["name"] for r in rows} != set(expected):
        raise ValueError("group rows differ from the registered factor levels")
    root = manifest_path.parent.resolve()
    verified = []
    for row in rows:
        path = (root / row["config"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("row config escapes its group directory")
        if row["overrides"] != expected[row["name"]]:
            raise ValueError("row overrides differ from registered factor levels")
        if sha256_file(path) != row["sha256"]:
            raise ValueError("row config changed after group generation")
        load_experiment_config(path)
        if yaml.safe_load(path.read_text()) != row_configuration(raw, row["overrides"]):
            raise ValueError("row changes fields outside its declared factor")
        verified.append({"name": row["name"], "config": str(path), "sha256": row["sha256"]})
    return {
        "status": "group_config_verified_not_launch_preflight",
        "manifest_sha256": sha256_file(manifest_path),
        "group": manifest["group"],
        "parent_sha256": manifest["parent_sha256"],
        "rows": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--batch-probes", type=Path)
    parser.add_argument("--holdout", type=Path, help="JSON with partition path and SHA256")
    args = parser.parse_args()
    if args.verify_manifest:
        if args.group or args.output:
            parser.error("verification does not accept generation arguments")
        print(json.dumps(verify_group(args.verify_manifest, args.parent), indent=2))
        return
    if not args.group or not args.output:
        parser.error("generation requires --group and --output")
    result = generate(
        args.parent,
        args.group,
        args.output,
        batch_probes=(json.loads(args.batch_probes.read_text()) if args.batch_probes else None),
        holdout=json.loads(args.holdout.read_text()) if args.holdout else None,
    )
    print(
        json.dumps({"group": args.group, "rows": len(result["rows"]), "status": result["status"]})
    )


if __name__ == "__main__":
    main()
