"""Materialize the full preregistered matrix without inventing validation winners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_group import GROUPS, generate, levels, measured_batches, verify_group
from prepare_dataset import DATASETS, prepare_initial
from prepare_followup import validate_dependency

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file


def prepare(
    dataset: str,
    config: Path,
    receipt: Path,
    receipt_sha256: str,
    output: Path,
    *,
    batch_probes: list[dict],
    holdout: dict,
) -> dict:
    # Check capacity and partition inputs before creating a partial matrix.
    batches = measured_batches(config, batch_probes)
    levels("G1", holdout=holdout)
    initial = prepare_initial(dataset, config, receipt, receipt_sha256, output)
    parent = output / "parent/gradpert_v2" / f"{dataset}.yaml"
    records = {}
    for group in GROUPS:
        manifest_path = output / group / "manifest.json"
        if group not in ("B0", "H1"):
            manifest = generate(
                parent,
                group,
                output / group,
                batch_probes=batch_probes if group == "H3" else None,
                holdout=holdout if group == "G1" else None,
            )
            manifest["status"] = "prospective_requires_validation_selected_parent"
            manifest["activation"] = "Regenerate in a new directory with prepare_followup.py"
            atomic_json(manifest_path, manifest)
            # The actual queue must reject every prospective later group.
            try:
                validate_dependency(manifest_path, parent)
            except ValueError as error:
                if "validation-selected parent" not in str(error):
                    raise
            else:
                raise AssertionError("prospective group unexpectedly eligible for launch")
        else:
            validate_dependency(manifest_path, parent)
        verified = verify_group(manifest_path, parent)
        records[group] = {
            "manifest": str(manifest_path.relative_to(output)),
            "manifest_sha256": sha256_file(manifest_path),
            "rows": len(verified["rows"]),
            "stage": "initial_preflight_pending" if group in ("B0", "H1") else "prospective",
            "required_selection": (
                None
                if group in ("B0", "H1")
                else "H1"
                if group == "H2"
                else "H2"
                if group == "H3"
                else "H3"
            ),
        }
    result = {
        "schema_version": "gradpert-v2-preregistered-matrix-1",
        "status": "initial_preflight_pending_later_groups_prospective",
        "dataset": dataset,
        "reference_capacity": initial["capacity"],
        "parent": str(parent.relative_to(output)),
        "parent_sha256": sha256_file(parent),
        "measured_global_batches": batches,
        "groups": records,
        "total_training_rows": sum(row["rows"] for row in records.values()),
        "diagnostics": {
            "G1_context": {
                "entry": "scripts/v2/evaluate_context.py",
                "requires": "sealed fixed-axis/context protocol and frozen checkpoint",
            },
            "D1": {
                "entry": "scripts/v2/evaluate_diagnostics.py",
                "levels": ["change_control", "change_perturbation", "block_response_cls_to_gene"],
                "requires": "sealed condition/query protocol and frozen checkpoint",
            },
        },
        "selection_rule": "H1 -> H2 -> H3 uses validation prediction loss only, never test scores",
        "activation_rule": (
            "Initial rows still require exact source/config/seed CUDA preflight. "
            "Later YAML files are complete nominal configurations, not selected winners. "
            "Regenerate them from actual selection receipts; do not edit a prospective manifest "
            "to bypass the launch dependency gate."
        ),
    }
    atomic_json(output / "matrix.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="nadig_jurkat")
    for name in ("capacity-config", "receipt", "batch-probes", "holdout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    result = prepare(
        args.dataset,
        args.capacity_config,
        args.receipt,
        args.receipt_sha256,
        args.output,
        batch_probes=json.loads(args.batch_probes.read_text()),
        holdout=json.loads(args.holdout.read_text()),
    )
    print(json.dumps({"dataset": result["dataset"], "rows": result["total_training_rows"]}))


if __name__ == "__main__":
    main()
