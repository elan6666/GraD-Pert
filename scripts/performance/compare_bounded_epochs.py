"""Strict read-only comparison of uninterrupted and resumed epoch diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gradpert.hashing import sha256_file


def require_equal(left: Any, right: Any, label: str) -> None:
    if left != right:
        raise ValueError(f"exact comparison failed: {label}")


def validate_segment(receipt: dict[str, Any], start: int, stop: int) -> None:
    require_equal(receipt["status"], "complete", "status")
    require_equal(receipt["error"], None, "error")
    require_equal(receipt["scientific_completion"], False, "scientific completion")
    require_equal(receipt["source"]["dirty"], False, "source cleanliness")
    require_equal(receipt["start_epoch"], start, "start epoch")
    require_equal(receipt["stop_before_epoch"], stop, "stop epoch")
    require_equal(receipt["schedule_epochs"], 50, "schedule horizon")
    require_equal(receipt["persistent_pkl_count"], 0, "PKL count")
    require_equal(
        receipt["guard"],
        {"validation_constructor_count": 1, "test_access_attempts": 0},
        "evaluation guard",
    )
    predicates = receipt["preflight_predicates"]
    if not predicates or not all(v is True for v in predicates.values()):
        raise ValueError("preflight predicates failed or absent")
    if not receipt["terminal_predicates"] or not all(
        v is True for v in receipt["terminal_predicates"].values()
    ):
        raise ValueError("terminal predicates failed or absent")
    require_equal(
        [r["completed_epochs"] for r in receipt["boundaries"]],
        list(range(start, stop + 1)),
        "boundaries",
    )
    for boundary in receipt["boundaries"]:
        require_equal(
            set(boundary["state"]),
            {"model", "teacher", "gradients", "optimizer", "centers", "rng"},
            "state coverage",
        )


def compare_segments(reference, candidate, parent, resumed) -> None:
    for receipt, start, stop, impl in (
        (reference, 0, 3, "cpu_vectorized"),
        (candidate, 0, 3, "cpu_array"),
        (parent, 0, 1, "cpu_array"),
        (resumed, 1, 3, "cpu_array"),
    ):
        validate_segment(receipt, start, stop)
        require_equal(receipt["implementation"], impl, "implementation")
        for key in ("config_sha256", "frozen_identity"):
            require_equal(receipt[key], reference[key], key)
        for key in ("commit", "tree_sha256"):
            require_equal(receipt["source"][key], reference["source"][key], key)
    trajectories = [candidate["boundaries"], parent["boundaries"] + resumed["boundaries"][1:]]
    for trajectory in trajectories:
        for old, new in zip(reference["boundaries"], trajectory, strict=True):
            for key in ("completed_epochs", "global_step", "state"):
                require_equal(old[key], new[key], key)
            if old["completed_epochs"]:
                for key in ("checkpoint_identity", "validation_rows"):
                    require_equal(old[key], new[key], key)
                for role in ("best", "last"):
                    for key in ("content_sha256", "progress"):
                        require_equal(
                            old["checkpoints"][role][key], new["checkpoints"][role][key], role + key
                        )
    # Gradients are intentionally absent from serialized checkpoints. Compare
    # persistent state/RNG on restart; all six surfaces again after training.
    for key in ("model", "teacher", "optimizer", "centers", "rng"):
        require_equal(
            parent["boundaries"][-1]["state"][key],
            resumed["boundaries"][0]["state"][key],
            "resume " + key,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "resumed", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("comparison evidence already exists")
    files = [
        args.reference / "bounded_evidence/segment-0-to-3.json",
        args.candidate / "bounded_evidence/segment-0-to-3.json",
        args.resumed / "bounded_evidence/segment-0-to-1.json",
        args.resumed / "bounded_evidence/segment-1-to-3.json",
    ]
    receipts = [json.loads(p.read_text()) for p in files]
    compare_segments(*receipts)
    curves = []
    for root in (args.reference, args.candidate, args.resumed):
        if list(root.rglob("*.pkl")):
            raise ValueError("live PKL postcondition failed")
        with (root / "small_results/train_steps.csv").open() as stream:
            curves.append(
                [
                    {k: v for k, v in r.items() if not k.endswith("_ms")}
                    for r in csv.DictReader(stream)
                ]
            )
        final = (
            json.loads((root / "bounded_evidence/segment-0-to-3.json").read_text())
            if root != args.resumed
            else receipts[3]
        )
        for relative, key in (
            ("train_steps.csv", "steps_sha256"),
            ("validation.csv", "validation_sha256"),
        ):
            require_equal(
                sha256_file(root / "small_results" / relative),
                final["boundaries"][-1][key],
                "live " + relative,
            )
        for role in ("best", "last"):
            require_equal(
                sha256_file(root / f"checkpoints/{role}.pt"),
                final["boundaries"][-1]["checkpoints"][role]["sha256"],
                "live checkpoint SHA",
            )
    require_equal(curves[0], curves[1], "complete candidate curve")
    require_equal(curves[0], curves[2], "complete resumed curve")
    with args.output.open("x") as stream:
        json.dump(
            {
                "exact": True,
                "scientific_completion": False,
                "receipt_sha256": {str(p): sha256_file(p) for p in files},
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
