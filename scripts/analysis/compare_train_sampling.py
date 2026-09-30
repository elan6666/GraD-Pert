"""Compare terminal, identity-compatible Jurkat sampling analysis receipts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file


def collect(roots: list[Path]) -> tuple[dict, list[dict]]:
    sources = {}
    base = None
    combined = []
    for root in roots:
        summary = json.loads((root / "summary.json").read_text())
        receipt = json.loads((root / "receipt.json").read_text())
        complete = json.loads((root / "COMPLETE.json").read_text())
        records = json.loads((root / "conditions.json").read_text())
        method = summary.get("method", "cap40")
        if method in sources:
            raise ValueError("duplicate method")
        if receipt["status"] != "complete" or complete["status"] != "complete":
            raise ValueError("nonterminal source")
        if (
            receipt["source_commit"] != summary["source_commit"]
            or complete["source_commit"] != summary["source_commit"]
        ):
            raise ValueError("source commit mismatch")
        if (
            sha256_file(root / "summary.json") != receipt["summary_sha256"]
            or receipt["summary_sha256"] != complete["summary_sha256"]
        ):
            raise ValueError("summary hash mismatch")
        if sha256_file(root / "conditions.json") != receipt["outputs"]["conditions.json"]["sha256"]:
            raise ValueError("condition hash mismatch")
        if not summary["source_clean"] or not summary["original_data_unchanged"]:
            raise ValueError("unverified input/source")
        if base is None:
            base = (summary, records)
            combined = [
                {
                    "condition": r["condition"],
                    "original_cells": r["original_cells"],
                    "original_delta": r["original_delta"],
                    "original_expression": r["original_expression"],
                }
                for r in records
            ]
        reference, original = base
        for key in (
            "parent_h5ad_sha256",
            "parent_split_sha256",
            "parent_manifest_sha256",
            "control_reference_content_sha256",
            "seed",
            "repeats",
            "expression_genes",
            "original_train_cells",
            "conditions",
        ):
            if summary[key] != reference[key]:
                raise ValueError(f"incompatible reference: {key}")
        if [r["condition"] for r in records] != [r["condition"] for r in original]:
            raise ValueError("condition order mismatch")
        for current, old, target in zip(records, original, combined, strict=True):
            for key in (
                "original_cells",
                "original_batches",
                "original_half_assignment_sha256",
                "original_valid_repeats",
            ):
                if current[key] != old[key]:
                    raise ValueError(f"reference mismatch: {key}")
            for key in ("original_delta", "original_expression"):
                if current[key] != old[key] and not (
                    current[key] is not None
                    and old[key] is not None
                    and np.isclose(current[key], old[key], rtol=0, atol=1e-13)
                ):
                    raise ValueError(f"reference score mismatch: {key}")
            target[method] = {
                k: current[k]
                for k in (
                    "cap_cells",
                    "cap_batches",
                    "cap_delta",
                    "cap_expression",
                    "affected",
                    "sample_vs_full_delta",
                    "retained_vs_removed_delta",
                )
            }
        sources[method] = {
            "root": str(root),
            "source_commit": summary["source_commit"],
            "summary_sha256": receipt["summary_sha256"],
            "conditions_sha256": receipt["outputs"]["conditions.json"]["sha256"],
            "summary": summary,
        }
    if set(sources) != {"cap40", "proportion50", "proportion25"}:
        raise ValueError("expected all three sampling methods")
    return sources, combined


def render(sources: dict, records: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "original": "#235f9a",
        "proportion50": "#278965",
        "cap40": "#d5762c",
        "proportion25": "#9467bd",
    }
    labels = {
        "original": "Original",
        "proportion50": "50%",
        "cap40": "Cap40",
        "proportion25": "25%",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method in colors:
        values = [
            r["original_delta"] if method == "original" else r[method]["cap_delta"] for r in records
        ]
        axes[0, 0].hist(
            [v for v in values if v is not None],
            bins=np.linspace(-1, 1, 51),
            histtype="step",
            linewidth=1.8,
            color=colors[method],
            label=labels[method],
        )
        cells = (
            sources["cap40"]["summary"]["original_train_cells"]
            if method == "original"
            else sources[method]["summary"]["sampled_train_cells"]
        )
        mean = (
            sources["cap40"]["summary"]["original"]["mean"]
            if method == "original"
            else sources[method]["summary"][method]["mean"]
        )
        axes[0, 1].scatter(cells, mean, color=colors[method], s=65)
        axes[0, 1].annotate(
            labels[method], (cells, mean), xytext=(6, 5), textcoords="offset points"
        )
    axes[0, 0].legend()
    axes[0, 0].set(xlabel="Mean split-half Pearson delta (100 splits)", ylabel="Conditions")
    axes[0, 1].set(xlabel="Retained training cells", ylabel="Condition-equal Pearson delta")
    pairs = [
        r
        for r in records
        if r["cap40"]["cap_delta"] is not None and r["proportion50"]["cap_delta"] is not None
    ]
    axes[1, 0].scatter(
        [r["cap40"]["cap_delta"] for r in pairs],
        [r["proportion50"]["cap_delta"] for r in pairs],
        s=8,
        alpha=0.4,
        color=colors["proportion50"],
    )
    axes[1, 0].plot([-1, 1], [-1, 1], "--", color="gray")
    axes[1, 0].set(xlabel="Cap40 Pearson delta", ylabel="50% Pearson delta")
    strata = ["1-40", "41-80", "81-160", ">160"]
    x = np.arange(4)
    for index, method in enumerate(colors):
        summary = sources["cap40" if method == "original" else method]["summary"]
        key = "original" if method == "original" else "cap"
        values = [summary["cell_count_strata"][s][key]["mean"] for s in strata]
        axes[1, 1].bar(
            x + (index - 1.5) * 0.2, values, width=0.2, label=labels[method], color=colors[method]
        )
    axes[1, 1].set(
        xticks=x,
        xticklabels=strata,
        xlabel="Original cells per condition",
        ylabel="Condition-equal Pearson delta",
    )
    axes[1, 1].legend()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"comparison.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to("/data/yilangliu"):
        parser.error("comparison requires a new server output directory")
    repo = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        parser.error("clean source required")
    sources, records = collect(args.inputs)
    args.output.mkdir(parents=True)
    atomic_json(args.output / "sources.json", sources)
    atomic_json(args.output / "conditions.json", records)
    render(sources, records, args.output)
    atomic_json(
        args.output / "COMPLETE.json",
        {
            "source_commit": source,
            "source_clean": True,
            "status": "complete",
            "reference_parity": "identities and half assignments exact; scores atol1e-13 rtol0",
            "outputs": {
                p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in args.output.iterdir()
                if p.is_file()
            },
        },
    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
