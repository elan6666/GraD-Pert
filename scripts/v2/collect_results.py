"""Collect v2 best/last receipts with explicit missing and invalid evidence states."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from gradpert.hashing import sha256_file


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def collect_run(root: Path) -> list[dict]:
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        return [
            {"run_root": str(root), "role": role, "status": "missing_manifest"}
            for role in ("best", "last")
        ]
    manifest = read(manifest_path)
    source = manifest["source"]
    if source["dirty"] or source["commit"] != source["published_commit"]:
        raise ValueError("training source is not clean and published")
    resolved = read(root / "resolved_config.json")
    policy = resolved["training"]["formal_run_policy"]
    expected_epochs = {"v2_fixed_5": 5, "v2_fixed_50": 50}.get(policy)
    if expected_epochs is None or resolved["training"]["max_epochs"]["value"] != expected_epochs:
        raise ValueError("run has no supported fixed-epoch v2 contract")
    journal_path = root / "fit/epoch_state.json"
    journal = read(journal_path) if journal_path.exists() else None
    if journal and journal["identity"] != manifest:
        raise ValueError("epoch journal training identity differs from manifest")
    complete_path = root / "COMPLETE.json"
    complete = read(complete_path) if complete_path.exists() else None
    if complete:
        if complete["identity"] != manifest or complete["epoch"] != expected_epochs:
            raise ValueError("completion identity or epoch differs from fixed-epoch contract")
        if (
            not journal
            or journal["epoch"] != expected_epochs
            or journal["budget"][0] != expected_epochs
        ):
            raise ValueError("completion has no complete fixed-epoch journal")
        history = read(root / "fit/history.json")
        if [h["epoch"] for h in history] != list(range(1, expected_epochs + 1)):
            raise ValueError("completion lacks all committed epochs")
        if any(
            h["validation"]["split"] != "val"
            or not math.isfinite(h["validation"]["prediction_loss"])
            for h in history
        ):
            raise ValueError("selection history must contain finite validation losses")
        best = min(history, key=lambda h: h["validation"]["prediction_loss"])
        for role, row in (("best", best), ("last", history[-1])):
            if (
                journal[role]["epoch"] != row["epoch"]
                or journal[role]["prediction_loss"] != row["validation"]["prediction_loss"]
            ):
                raise ValueError("checkpoint selection differs from validation history")
        if set(complete["test_roles"]) != {"best", "last"} or not complete["zero_pkl"]:
            raise ValueError("completion lacks both test roles or zero-PKL evidence")
        if any(root.rglob("*.pkl")):
            raise ValueError("completed run contains PKL files")
    rows = []
    for role in ("best", "last"):
        row = {
            "run_root": str(root.resolve()),
            "run_id": manifest["run_id"],
            "role": role,
            "training_sha": source["commit"],
            "config_sha256": manifest["config_sha256"],
            "resolved_config_sha256": manifest["resolved_config_sha256"],
            "run_manifest_sha256": sha256_file(manifest_path),
            "status": "missing_checkpoint",
        }
        selected = journal.get(role) if journal else None
        if not selected:
            rows.append(row)
            continue
        if complete and complete[role] != selected:
            raise ValueError("completion selected checkpoint differs from epoch journal")
        checkpoint = (root / "fit" / selected["file"]).resolve()
        if not checkpoint.is_relative_to((root / "fit").resolve()):
            raise ValueError("checkpoint escapes owned run directory")
        if not checkpoint.exists() or sha256_file(checkpoint) != selected["sha256"]:
            raise ValueError("selected checkpoint missing or hash mismatch")
        row.update(
            checkpoint_sha256=selected["sha256"],
            checkpoint_epoch=selected["epoch"],
            validation_prediction_loss=selected["prediction_loss"],
            status="missing_test",
        )
        path = root / "fit" / f"{role}-test.json"
        if not path.exists():
            if complete:
                raise ValueError("completion claims a missing test receipt")
            rows.append(row)
            continue
        test = read(path)
        identity, result = test["identity"], test["result"]
        if (
            identity["training"] != manifest
            or identity["checkpoint"] != selected
            or identity["role"] != role
            or result["split"] != "test"
        ):
            raise ValueError("test receipt checkpoint, role, split or training identity mismatch")
        evaluation = identity["evaluation"]
        if evaluation["source"]["dirty"]:
            raise ValueError("test evaluation source is dirty")
        row.update(
            evaluation_sha=evaluation["source"]["commit"],
            test_receipt_sha256=sha256_file(path),
            test_prediction_loss=result["prediction_loss"],
            status="complete" if complete else "tested_completion_missing",
        )
        for metric in result["metrics"]:
            row[metric["metric_id"]] = metric["macro_mean"]
            row[metric["metric_id"] + "_finite_conditions"] = metric["finite_condition_count"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(p.resolve() for p in args.run)) != len(args.run):
        parser.error("duplicate run roots")
    rows = []
    for root in args.run:
        try:
            rows.extend(collect_run(root))
        except (ValueError, KeyError, OSError) as error:
            rows.extend(
                {
                    "run_root": str(root),
                    "role": role,
                    "status": "invalid_evidence",
                    "error": str(error),
                }
                for role in ("best", "last")
            )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {"rows": len(rows), "complete_roles": sum(r["status"] == "complete" for r in rows)}
        )
    )


if __name__ == "__main__":
    main()
