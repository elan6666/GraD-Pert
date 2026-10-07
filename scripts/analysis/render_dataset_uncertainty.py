"""Render Chinese observational figures from sealed scalar/coordinate results.

Run on the server: sampled coordinates remain there. Only figures, scalar
summary JSON, and condition-level tables are transferable manuscript assets.
"""

# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

from gradpert.hashing import sha256_file
from scripts.analysis.five_dataset_diagnostics import DATASETS

NAMES = ["K562", "RPE1", "Jurkat", "HepG2", "Norman"]
COLORS = ["#538F89", "#DBAD58", "#857AA8", "#6B9DC5", "#C67A79"]


def stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
    }


def setup_font():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font.is_file():
        raise RuntimeError("Chinese font unavailable")
    font_manager.fontManager.addfont(str(font))
    name = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update(
        {
            "font.family": name,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=10)
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=11, weight="bold")
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, directory, name):
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(directory / f"{name}.{ext}", dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def violin(ax, values, colors, *, positions=None, width=0.7):
    positions = np.arange(len(values)) if positions is None else positions
    for x, value, color in zip(positions, values, colors, strict=True):
        v = np.asarray(value)
        v = v[np.isfinite(v)]
        if len(v) > 1 and np.ptp(v) > 1e-9:
            artist = ax.violinplot(v, positions=[x], widths=width, showextrema=False)
            artist["bodies"][0].set(facecolor=color, edgecolor=color, alpha=0.4)
        ax.boxplot(
            [v],
            positions=[x],
            widths=width * 0.25,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": color},
            medianprops={"color": "#222222"},
            whiskerprops={"color": color},
            capprops={"color": color},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--legacy-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to("/data/yilangliu") or args.output.exists():
        parser.error("new server-only output required")
    if json.loads((args.input / "manifest.json").read_text())["status"] != "complete":
        parser.error("terminal completed analysis required")
    args.output.mkdir(parents=True)
    setup_font()
    splits = json.loads((args.input / "split_summary.json").read_text())
    details = {
        d: json.loads((args.input / f"{d}-landscape-details.json").read_text())
        for d in [*DATASETS, "crosscell"]
    }
    legacy = json.loads(args.legacy_receipt.read_text())
    groups = []
    for dataset, name, color in zip(DATASETS, NAMES, COLORS, strict=True):
        line = next(k for k in details[dataset] if k != "embedding")
        groups.append((name, details[dataset][line], color))
    for line, color in zip(["K562", "RPE1", "jurkat", "hepg2"], COLORS[:4], strict=True):
        groups.append(
            (
                "X-"
                + line.title()
                .replace("K562", "K562")
                .replace("Rpe1", "RPE1")
                .replace("Hepg2", "HepG2"),
                details["crosscell"][line],
                color,
            )
        )
    names = [g[0] for g in groups]
    x = np.arange(len(groups))
    summary = {
        "split": {},
        "landscape": {},
        "interpretation": (
            "SD of random partitions and condition heterogeneity; no biological-replicate CI"
        ),
    }
    for dataset in DATASETS:
        item = splits[dataset]
        summary["split"][dataset] = {
            "n_conditions": len(item["conditions"]),
            "median_across_splits": stats([r["median_delta"] for r in item["per_split"]]),
            "expression_across_splits": stats([r["median_expression"] for r in item["per_split"]]),
            "condition_medians": stats([r.get("median", np.nan) for r in item["conditions"]]),
            "condition_sd": stats([r.get("sd", np.nan) for r in item["conditions"]]),
        }
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.1), layout="constrained")
    ax = axes[0, 0]
    for i, (_name, item, color) in enumerate(groups):
        b = item["batch_information"]
        if b is None:
            ax.text(i, 0, "不适用", ha="center", rotation=90, fontsize=7)
            continue
        null = np.asarray(b["null"])
        zeroed = null - null.mean()
        ax.boxplot(
            [zeroed],
            positions=[i],
            widths=0.4,
            showfliers=False,
            boxprops={"color": "#999999"},
            medianprops={"color": "#999999"},
        )
        ax.scatter(i, b["excess"], color=color, s=28, marker="D", zorder=3)
    panel(ax, "a", "条件—批次关联及置换参照")
    ax.set_ylabel("超额互信息比例")
    ax.axhline(0, color="#777777", lw=0.6)
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="D", color="none", markerfacecolor="#538F89", label="观测超额关联"
            ),
            Line2D([], [], color="#999999", label="999 次置换零分布"),
        ],
        loc="upper left",
    )
    ax = axes[0, 1]
    for i, (_, item, color) in enumerate(groups):
        vals = [v["dissimilarity"] for v in item["within_control"]]
        between = [v["dissimilarity"] for v in item["across_control"]]
        violin(ax, [vals], [color], positions=[i - 0.18], width=0.3)
        if between:
            violin(ax, [between], ["#B5B5B5"], positions=[i + 0.18], width=0.3)
        else:
            ax.text(i + 0.18, 0.3, "不适用", rotation=90, fontsize=6, ha="center")
    panel(ax, "b", "对照表达：同批与跨批分布")
    ax.set_ylabel("不相似度 1000(1−Pearson)")
    ax.legend(
        handles=[
            Line2D([], [], color="#538F89", lw=5, label="同批拆半"),
            Line2D([], [], color="#B5B5B5", lw=5, label="跨批均值"),
        ]
    )
    ax = axes[1, 0]
    rng = np.random.default_rng(42)
    for i, (_, item, color) in enumerate(groups):
        v = np.asarray([r["top1"] for r in item["retrieval"]])
        ax.bar(i, v.mean(), color=color, alpha=0.3, width=0.65)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color="#222222", capsize=3, lw=1)
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, color=color, s=9, zorder=3)
        ax.scatter(
            i,
            item["retrieval_randomization"]["random_expectation"],
            color="#222222",
            marker="_",
            s=45,
        )
    panel(ax, "c", "固定候选集：20 次拆半检索")
    ax.set_ylabel("Top-1 检索率")
    ax.set_ylim(0, 1)
    ax.text(
        0.03, 0.96, "点：每次划分；柱与线：均值 ± SD", transform=ax.transAxes, va="top", fontsize=7
    )
    ax = axes[1, 1]
    values = [[r["pearson_shared"] for r in g[1]["common_response"]] for g in groups]
    violin(ax, values, [g[2] for g in groups])
    panel(ax, "d", "共同响应：逐条件相关的分布")
    ax.set_ylabel("与共同均值效应的 Pearson")
    ax.set_ylim(-1, 1)
    for ax in axes.flat:
        ax.set_xticks(x, names, rotation=48, ha="right")
        ax.set_xlim(-0.7, len(names) - 0.3)
    save(fig, args.output, "fig1-landscape")
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.8), layout="constrained")
    ax = axes[0, 0]
    for i, (dataset, color) in enumerate(zip(DATASETS, COLORS, strict=True)):
        for offset, key, alpha in [(-0.17, "median_delta", 0.5), (0.17, "median_expression", 0.18)]:
            v = np.asarray([r[key] for r in splits[dataset]["per_split"]])
            ax.bar(i + offset, v.mean(), width=0.28, color=color, alpha=alpha)
            ax.errorbar(
                i + offset, v.mean(), yerr=v.std(ddof=1), color="#333333", capsize=2, lw=0.8
            )
            ax.scatter(
                i + offset + rng.uniform(-0.08, 0.08, len(v)),
                v,
                s=7,
                color=color,
                marker="o" if key == "median_delta" else "x",
                linewidths=0.5,
            )
    panel(ax, "a", "总体水平与拆分波动")
    ax.set_ylabel("每次划分的条件中位相关")
    ax.set_ylim(0, 1.04)
    ax.legend(
        handles=[
            Line2D([], [], color="#777777", marker="o", ls="", label="效应"),
            Line2D([], [], color="#777777", marker="x", ls="", label="完整表达"),
        ],
        ncol=2,
    )
    ax = axes[0, 1]
    violin(
        ax,
        [[r["median"] for r in splits[d]["conditions"] if "median" in r] for d in DATASETS],
        COLORS,
    )
    panel(ax, "b", "条件之间的重复性差异")
    ax.set_ylabel("每条件 20 次的中位 Pearson Δ")
    ax.set_ylim(-1, 1.05)
    ax = axes[1, 0]
    violin(
        ax,
        [[r["sd"] for r in splits[d]["conditions"] if r.get("sd") is not None] for d in DATASETS],
        COLORS,
    )
    panel(ax, "c", "同一条件对随机拆分的敏感性")
    ax.set_ylabel("每条件 20 次 Pearson Δ 的 SD")
    ax = axes[1, 1]
    for d, name, color in zip(DATASETS, NAMES, COLORS, strict=True):
        rows = [r for r in splits[d]["conditions"] if "median" in r]
        ax.scatter(
            [r["truth_cells"] for r in rows],
            [r["median"] for r in rows],
            s=8,
            alpha=0.45,
            color=color,
            label=name,
            rasterized=True,
        )
    panel(ax, "d", "样本量与条件重复性")
    ax.set(xscale="log", xlabel="每条件实际细胞数", ylabel="条件中位 Pearson Δ", ylim=(-1, 1.05))
    ax.legend(ncol=2, fontsize=6.5)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xticks(range(5), NAMES, rotation=25)
    save(fig, args.output, "fig2-split-variation")
    norman = sorted(splits["norman"]["conditions"], key=lambda r: r.get("median", -2))
    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, 3.4), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1]}
    )
    ax = axes[0]
    image = ax.imshow(
        np.asarray([r["values"] for r in norman], dtype=float),
        aspect="auto",
        cmap="viridis",
        vmin=-1,
        vmax=1,
    )
    ax.set_yticks(
        range(len(norman)), [r["condition"].replace("+ctrl", "") for r in norman], fontsize=7
    )
    ax.set_xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
    ax.set_xlabel("随机划分编号")
    panel(ax, "a", "Norman：全部测试组合 × 20 次划分")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="Pearson Δ")
    ax = axes[1]
    for i, r in enumerate(norman):
        ax.plot([r["median"], r["expression_median"]], [i, i], color="#C7C7C7", lw=1)
        ax.scatter(r["median"], i, color=COLORS[4], s=20)
        ax.scatter(r["expression_median"], i, color="#555555", marker="x", s=20)
    ax.set_yticks(range(len(norman)), [str(i + 1) for i in range(len(norman))])
    ax.invert_yaxis()
    ax.set_xlabel("每条件 20 次的中位相关")
    ax.set_xlim(0, 1.04)
    panel(ax, "b", "相同行顺序：效应与完整表达")
    ax.legend(
        handles=[
            Line2D([], [], color=COLORS[4], marker="o", ls="", label="效应"),
            Line2D([], [], color="#555555", marker="x", ls="", label="完整表达"),
        ],
        fontsize=6,
    )
    save(fig, args.output, "fig3-norman-all-splits")
    for dataset, name in zip(DATASETS, NAMES, strict=True):
        points = details[dataset]["embedding"]["points"]
        kind = np.asarray([p["kind"] for p in points])
        part = np.asarray([p["split"] for p in points])
        batch = np.asarray([p["batch"] for p in points])
        top = {b for b, _ in Counter(batch).most_common(8)}
        batch = np.asarray([b if b in top else "其他批次" for b in batch])
        fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4), layout="constrained")
        kinds = {0: ("对照", "#666666"), 1: ("单基因扰动", "#6B9DC5"), 2: ("双基因扰动", "#C67A79")}
        parts = {
            "control": ("对照", "#666666"),
            "train": ("训练", "#538F89"),
            "val": ("验证", "#DBAD58"),
            "test": ("测试", "#C67A79"),
            "excluded": ("排除条件", "#B5B5B5"),
        }
        for row, prefix in enumerate(["pca", "umap"]):
            coords = np.asarray([[p[prefix + "1"], p[prefix + "2"]] for p in points])
            for col, (label, values, mapping) in enumerate(
                [
                    ("扰动类型", kind, kinds),
                    ("冻结划分", part, parts),
                    ("批次（主要 8 档）", batch, None),
                ]
            ):
                ax = axes[row, col]
                for j, value in enumerate(sorted(set(values))):
                    mask = values == value
                    text, color = (
                        mapping[value] if mapping else (str(value), plt.get_cmap("tab10")(j % 10))
                    )
                    ax.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        s=2,
                        alpha=0.5,
                        color=color,
                        label=text,
                        rasterized=True,
                    )
                ax.set_title(label)
                ax.set(xlabel=prefix.upper() + " 1", ylabel=prefix.upper() + " 2")
                ax.legend(
                    frameon=False,
                    fontsize=5.5,
                    ncol=2,
                    markerscale=3,
                    handletextpad=0.2,
                    columnspacing=0.5,
                )
        save(fig, args.output, f"embedding-{dataset}")
        rows = sorted(splits[dataset]["conditions"], key=lambda r: r.get("median", -2))
        fig, ax = plt.subplots(figsize=(7.6, 3.6), layout="constrained")
        image = ax.imshow(
            np.asarray([r["values"] for r in rows], dtype=float),
            aspect="auto",
            cmap="viridis",
            vmin=-1,
            vmax=1,
        )
        ax.set(
            xlabel="随机划分编号",
            ylabel="按中位相关排序的条件（完整身份见数值表）",
            title=name + "：全部条件 × 20 次拆半",
        )
        ax.set_xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
        fig.colorbar(image, ax=ax, label="Pearson Δ")
        save(fig, args.output, f"split-heatmap-{dataset}")
    transfer = legacy["datasets"]["crosscell"]["crosscell_transfer"]
    lines = ["K562", "RPE1", "jurkat", "hepg2"]
    display = ["K562", "RPE1", "Jurkat", "HepG2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7), layout="constrained")
    matrix = np.full((4, 4), np.nan)
    for i, target in enumerate(lines):
        for j, source_line in enumerate(lines):
            if i != j:
                matrix[i, j] = transfer[target]["pairwise"][source_line]["median_pearson_delta"]
    image = axes[0].imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    for i in range(4):
        for j in range(4):
            if i != j:
                axes[0].text(
                    j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9
                )
    axes[0].set_xticks(range(4), display)
    axes[0].set_yticks(range(4), display)
    axes[0].set(xlabel="来源细胞系", ylabel="目标细胞系")
    panel(axes[0], "a", "共享扰动的效应方向（历史口径）")
    fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.02, label="中位 Pearson Δ")
    axes[1].bar(
        range(4),
        [transfer[line]["source_average_pearson_delta_median"] for line in lines],
        color=COLORS[:4],
        width=0.6,
    )
    axes[1].set_xticks(range(4), display)
    axes[1].set(ylim=(0, 1), ylabel="与其他系观测均值的中位相关")
    panel(axes[1], "b", "来源均值参照；未估计重复性区间")
    save(fig, args.output, "fig4-crosscell")
    for name, item, _color in groups:
        within = [r["dissimilarity"] for r in item["within_control"]]
        across = [r["dissimilarity"] for r in item["across_control"]]
        summary["landscape"][name] = {
            "common": stats([r["pearson_shared"] for r in item["common_response"]]),
            "common_leave_one_out": stats(
                [r["pearson_leave_one_out"] for r in item["common_response"]]
            ),
            "within": stats(within),
            "across": stats(across),
            "retrieval": stats([r["top1"] for r in item["retrieval"]]),
            "candidates": len(item["candidate_conditions"]),
            "batch_test": {
                k: v for k, v in (item["batch_information"] or {}).items() if k != "null"
            },
            "retrieval_test": {
                k: v for k, v in item["retrieval_randomization"].items() if k != "null"
            },
        }
    (args.output / "paper_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    source = Path(__file__).resolve().parents[2]
    receipt = {
        "status": "complete",
        "rendering_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": sha256_file(Path(__file__)),
        "analysis_manifest_sha256": sha256_file(args.input / "manifest.json"),
        "files": {p.name: sha256_file(p) for p in args.output.iterdir() if p.is_file()},
    }
    (args.output / "render_receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
