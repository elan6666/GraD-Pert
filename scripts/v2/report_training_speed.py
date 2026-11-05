"""Render only already-approved small benchmark aggregates; no scientific data."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(folder: Path, output: Path, title: str):
    result = json.loads((folder / "TIMING_COMPARISON.json").read_text())
    plan = json.loads((folder / "TIMING_PLAN.json").read_text())
    order = plan["order"]
    colors = ["#7e91ab" if name.startswith("A") else "#d69b45" for name in order]
    rates = [
        result["arms"][name]["measured_cells_per_second_including_data_wait"] for name in order
    ]
    plt.rcParams.update({"font.family": "Arial", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    bars = axes[0].bar(order, rates, color=colors, width=0.62)
    for bar, value in zip(bars, rates, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.4f}", ha="center")
    axes[0].set_ylim(0, max(rates) * 1.18)
    axes[0].set_ylabel("Cells / second, including data wait")
    axes[0].set_title("Chronological benchmark arms")
    for name, color in zip(order, colors, strict=True):
        values = result["arms"][name]["measured_step_seconds_including_data_wait"]
        indices = range(plan["warmup_per_arm"] + 1, plan["warmup_per_arm"] + 1 + len(values))
        axes[1].plot(
            indices,
            values,
            marker="o",
            label=name,
            color=color,
            linestyle="-" if name.endswith("1") else "--",
        )
    axes[1].set_xticks(list(indices))
    axes[1].set_ylabel("Total seconds per update")
    axes[1].set_xlabel("Matched update index")
    axes[1].set_title("All measured samples retained")
    axes[1].legend(ncol=2, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    pair = result["pair_throughput_ratios"]
    verdict = "criterion passed" if result["speed_criterion_passed"] else "criterion failed"
    fig.suptitle(f"{title} | pair ratios {pair[0]:.5f}, {pair[1]:.5f} | {verdict}", fontsize=12)
    fig.supxlabel(
        "Source " + result["source"] + " | physical GPU1 | global320 / micro80 / accumulation4",
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    render(args.folder, args.output, args.title)
