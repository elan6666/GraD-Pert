"""Server rendering and scalar-table assembly for split-half sensitivity."""

# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from gradpert.hashing import sha256_file
from scripts.analysis.render_dataset_uncertainty import COLORS, NAMES, panel, save, setup_font


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/data/yilangliu")):
        raise ValueError("render on server")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("clean source required")
    receipt = json.loads((args.results / "manifest.json").read_text())
    if receipt["status"] != "complete":
        raise ValueError("terminal results required")
    args.output.mkdir(parents=True, exist_ok=False)
    setup_font()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    results = json.loads((args.results / "summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), constrained_layout=True)
    a, b, c, d = axes.flat
    publication = {}
    for i, (name, result) in enumerate(results.items()):
        sens = result["sensitivity"]
        for offset, key, marker, alpha in [
            (-0.23, "frozen_300", "x", 0.5),
            (0, "shared_symmetric_300", "o", 0.8),
            (0.23, "disjoint_300", "s", 1),
        ]:
            values = np.array(sens[key]["per_split_median"])
            a.scatter(
                i + offset + np.linspace(-0.04, 0.04, 20),
                values,
                s=12,
                marker=marker,
                color=COLORS[i],
                alpha=alpha,
            )
            a.errorbar(
                i + offset,
                values.mean(),
                yerr=values.std(ddof=1),
                color="#333333",
                capsize=3,
                marker="_",
                ms=12,
                lw=1,
            )
        table = pd.read_csv(args.results / f"{name}-control-sensitivity-long.csv.gz")
        complete = table.dropna(subset=["shared_symmetric_300", "disjoint_300"]).copy()
        complete["shared_minus_disjoint"] = (
            complete["shared_symmetric_300"] - complete["disjoint_300"]
        )
        medians = complete.groupby("condition")[
            ["frozen_300", "shared_symmetric_300", "disjoint_300", "shared_minus_disjoint"]
        ].median()
        b.scatter(
            medians.shared_symmetric_300,
            medians.disjoint_300,
            s=10,
            color=COLORS[i],
            alpha=0.4,
            label=NAMES[i],
        )
        publication[name] = {
            "conditions": len(medians),
            "paired_change_median": float(medians.shared_minus_disjoint.median()),
            "disjoint_condition_median": float(medians.disjoint_300.median()),
            "shared_condition_median": float(medians.shared_symmetric_300.median()),
        }
        summaries = table.groupby("condition").agg(truth_cells=("truth_cells", "first"))
        wide = summaries.copy()
        for metric in [
            "frozen_300",
            "shared_symmetric_300",
            "disjoint_300",
            "disjoint_ab_300",
            "disjoint_ba_300",
        ]:
            pivot = table.pivot(index="condition", columns="repeat", values=metric)
            pivot.columns = [f"{metric}_split_{j + 1:02d}" for j in pivot.columns]
            wide = wide.join(pivot)
        wide.join(medians, rsuffix="_median").to_csv(args.output / f"{name}-control-wide.csv")
        blocktable = pd.read_csv(args.results / f"{name}-batch-matched-long.csv.gz")
        blockwide = blocktable.pivot(
            index=["cells_per_condition_batch", "batch", "condition"],
            columns="repeat",
            values="same_condition_r",
        )
        blockwide.columns = [f"split_{j + 1:02d}" for j in blockwide.columns]
        metadata = blocktable.groupby(["cells_per_condition_batch", "batch", "condition"]).agg(
            available_condition_cells=("available_condition_cells", "first"),
            left_controls=("left_controls", "first"),
            right_controls=("right_controls", "first"),
            block_conditions=("block_conditions", "first"),
        )
        metadata.join(blockwide).to_csv(args.output / f"{name}-batch-matched-wide.csv")
        for ax, test in [(c, result["tests"][0]), (d, result["tests"][1])]:
            if test["status"] != "complete":
                continue
            full = receipt["results"][name]["tests"][test["cells_per_condition_batch"] == 10]
            null = np.asarray(full["null"])
            parts = ax.boxplot(
                null,
                positions=[i + 0.12],
                widths=0.18,
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
            )
            parts["boxes"][0].set(facecolor="#D9D9D9", edgecolor="#888888")
            parts["medians"][0].set(color="#555555")
            values = test["per_split"]
            ax.scatter(
                i - 0.12 + np.linspace(-0.04, 0.04, 20), values, s=12, color=COLORS[i], alpha=0.6
            )
            ax.scatter(
                i - 0.12,
                test["observed"],
                marker="D",
                s=42,
                color=COLORS[i],
                edgecolor="white",
                zorder=4,
            )
            ax.text(
                i,
                0.97,
                f"{test['covered_conditions']}/{test['total_test_conditions']}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
            )
            ax.text(
                i,
                0.035,
                f"校正 p={test['p_holm']:.4f}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    a.set(xticks=range(5), xticklabels=NAMES, ylabel="每次条件中位 Pearson", ylim=(-0.05, 1.02))
    handles = [
        Line2D([], [], color="#555555", marker=marker, linestyle="", label=label)
        for marker, label in [("x", "原固定对照"), ("o", "重新抽样共享对照"), ("s", "互斥对照")]
    ]
    a.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    panel(a, "a", "300 次对照抽样：20 次划分的全部总体值")
    b.plot([-1, 1], [-1, 1], ls="--", color="#999999", lw=1)
    b.set(
        xlabel="共享对照：条件中位 Pearson",
        ylabel="互斥对照：条件中位 Pearson",
        xlim=(-0.35, 1),
        ylim=(-0.35, 1),
    )
    b.legend(frameon=False, fontsize=8, loc="upper left")
    panel(b, "b", "同条件配对：对照误差敏感性")
    for ax, letter, title in [
        (c, "c", "同批次等细胞数：每半 2 个扰动细胞"),
        (d, "d", "同批次等细胞数：每半 5 个扰动细胞"),
    ]:
        panel(ax, letter, title)
        ax.set(
            xticks=range(5), xticklabels=NAMES, ylabel="条件等权平均 Pearson", ylim=(-0.22, 1.02)
        )
        ax.axhline(0, color="#BBBBBB", lw=0.6)
    legend = [
        Line2D([], [], color="#555555", marker="D", linestyle="", label="同扰动；点为20次值"),
        Line2D([], [], color="#AAAAAA", marker="s", linestyle="", label="9999次身份重排参照"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    save(fig, args.output, "fig5-split-significance")
    (args.output / "publication_summary.json").write_text(json.dumps(publication, indent=2))
    outputs = {
        p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in args.output.iterdir()
        if p.is_file()
    }
    render_receipt = {
        "status": "complete",
        "source_clean": True,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "analysis_commit": receipt["source_commit"],
        "input_manifest_sha256": sha256_file(args.results / "manifest.json"),
        "outputs": outputs,
    }
    (args.output / "render_receipt.json").write_text(json.dumps(render_receipt, indent=2))


if __name__ == "__main__":
    main()
