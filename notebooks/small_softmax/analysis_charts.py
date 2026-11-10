# ruff: noqa: RUF001
"""Plot helpers for the three GraD-Pert post-training notebooks."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, Markdown, display

SOURCE = Path(os.environ.get("GRADPERT_ANALYSIS_INPUT", ".")).resolve()
OUTPUT = Path(os.environ.get("GRADPERT_ANALYSIS_OUTPUT", "./figures")).resolve()
METRICS = {
    "txpert_macro_pearson_delta_all": "TxPert",
    "trishift_pearson_delta_all": "TriShift",
    "systema_pearson_all": "Systema",
}
MODELS = {"model": "Model", "control_copy": "Control copy", "train_mean": "Train mean"}


def load(relative):
    return json.loads((SOURCE / relative).read_text())


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / (name + ".png")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    display(Image(filename=str(path)))


def note(message):
    display(Markdown(message))


def grid(count, *, width=4.3, height=2.7, columns=3):
    cols = min(columns, count)
    fig, axes = plt.subplots(
        (count + cols - 1) // cols,
        cols,
        squeeze=False,
        layout="constrained",
        figsize=(width * cols, height * ((count + cols - 1) // cols)),
    )
    flat = list(axes.flat)
    for ax in flat[count:]:
        ax.set_visible(False)
    return fig, flat[:count]


def flatten(tree, path=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from flatten(value, f"{path}/{key}" if path else key)
    elif isinstance(tree, list):
        for idx, value in enumerate(tree):
            label = (
                next(
                    (
                        value[k]
                        for k in ("name", "component", "objective", "group")
                        if isinstance(value.get(k), str)
                    ),
                    str(idx),
                )
                if isinstance(value, dict)
                else str(idx)
            )
            yield from flatten(value, f"{path}/{label}")
    elif type(tree) in (int, float) and np.isfinite(tree):
        yield path, tree


def series(name, groups, xlabel="Epoch"):
    if not groups:
        note("此运行未采集该组数值。")
        return
    fig, axes = grid(len(groups))
    for ax, (label, data) in zip(axes, groups.items(), strict=True):
        for key, xy in data.items():
            x, y = zip(*xy, strict=True)
            ax.plot(x, y, marker="o", ms=4, label=key)
        if len(data) > 1:
            ax.legend(fontsize=7)
        ax.set_title(label.replace("_", " "), fontsize=9)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.23)
    save(fig, name)


def histories(data):
    track = {}
    training = {}
    for row in data["history"]:
        for key, val in flatten(row["training"]):
            training.setdefault(key, {}).setdefault("train", []).append((row["epoch"], val))
        for key, val in flatten({k: v for k, v in row.items() if k not in ("epoch", "training")}):
            track.setdefault(key, {}).setdefault("train", []).append((row["epoch"], val))
    series("01-optimization", training)
    series("02-time-resources", track)


def snapshot_lines(data, sections, name):
    groups = {}
    for row in data["snapshots"]:
        for key, val in flatten({key: row[key] for key in sections if key in row}):
            panel, _, label = key.rpartition("/")
            groups.setdefault(f"{panel}/{label}", {}).setdefault("fixed probe", []).append(
                (row["epoch"], val)
            )
    # Limit each page, so a configuration with more layers/losses stays readable.
    keys = list(groups)
    for start in range(0, len(keys), 18):
        series(f"{name}-{start // 18 + 1}", {k: groups[k] for k in keys[start : start + 18]})
    if not keys:
        note("此运行未采集这组诊断；无蒸馏配置不生成虚构的原型或教师数值。")


def feature_heatmaps(data):
    epochs = [row["epoch"] for row in data["snapshots"]]
    roles = sorted({name for row in data["snapshots"] for name in row["features"]})
    stats = sorted(
        {
            k
            for row in data["snapshots"]
            for feature in row["features"].values()
            for k, _ in flatten(feature)
        }
    )
    for page, start in enumerate(range(0, len(roles), 24), 1):
        subset = roles[start : start + 24]
        if not stats:
            break
        fig, axes = grid(len(stats), height=max(3, 0.27 * len(subset) + 1))
        for ax, key in zip(axes, stats, strict=True):
            array = np.array(
                [
                    [row["features"].get(role, {}).get(key, np.nan) for row in data["snapshots"]]
                    for role in subset
                ],
                dtype=float,
            )
            img = ax.imshow(array, aspect="auto", cmap="viridis")
            ax.set(xticks=range(len(epochs)), xticklabels=epochs, yticks=range(len(subset)))
            ax.set_yticklabels([s.removeprefix("student.") for s in subset], fontsize=6)
            ax.set_title(key.replace("_", " "), fontsize=9)
            ax.set_xlabel("Epoch")
            fig.colorbar(img, ax=ax, shrink=0.75)
        save(fig, f"04-representation-{page}")
    if not stats:
        note("此运行没有可绘制的表征探针。")


def pearson_charts(rows, acceptance):
    fig, axes = grid(9, height=2.45)
    counts, labels = [], []
    for i, (scope, label) in enumerate((("all", "All"), ("seen", "Seen"), ("unseen", "Unseen"))):
        for j, (metric, title) in enumerate(METRICS.items()):
            ax = axes[i * 3 + j]
            values = [r["pearson"][scope].get(metric, {}).get("value") for r in rows]
            valid = [v for v in values if v is not None]
            receipt = next(
                x["candidate"]
                for x in acceptance["comparison"]
                if x["metric"] == metric and x["scope"] == scope + "_expression"
            )
            if len(valid) != receipt["valid"]:
                raise ValueError("per-condition Pearson validity differs from the accepted macro")
            if valid and not np.isclose(np.mean(valid), receipt["macro_mean"], atol=1e-12, rtol=0):
                raise ValueError("per-condition Pearson macro differs from the accepted receipt")
            counts.append(len(valid))
            labels.append(f"{label} · {title}")
            ax.hist(valid, bins=np.linspace(-1, 1, 31), color="#386a9e")
            if valid:
                ax.axvline(
                    receipt["macro_mean"],
                    color="#c54d34",
                    label=f"Mean {receipt['macro_mean']:.4f}",
                )
                ax.axvline(0, color="#555", linestyle=":", lw=0.75)
                ax.legend(fontsize=7)
            else:
                ax.text(0, 0.5, "No finite values", ha="center", transform=ax.get_xaxis_transform())
            ax.set(title=labels[-1], xlabel="Condition Pearson", ylabel="Conditions")
    save(fig, "01-pearson-distributions")
    fig, axes = grid(2, columns=2)
    axes[0].barh(labels, counts, label="Finite")
    axes[0].barh(labels, [len(rows) - n for n in counts], left=counts, label="Unavailable")
    axes[0].set_xlabel("Conditions")
    axes[0].legend(fontsize=7)
    reasons = Counter(
        r["pearson"][scope].get(metric, {}).get("reason") or "empty / no value"
        for r in rows
        for scope in ("all", "seen", "unseen")
        for metric in METRICS
        if r["pearson"][scope].get(metric, {}).get("value") is None
    )
    if reasons:
        axes[1].barh([k.replace("_", " ") for k in reasons], list(reasons.values()))
    else:
        axes[1].text(0.5, 0.5, "No missing Pearson entries", ha="center", va="center")
    axes[1].set(xlabel="Condition × scope × metric entries", title="Missingness reasons")
    save(fig, "02-pearson-coverage")


def expression_gap(rows):
    fig, axes = grid(3)
    for ax, (metric, title) in zip(axes, METRICS.items(), strict=True):
        seen = [r["pearson"]["seen"].get(metric, {}).get("value") for r in rows]
        unseen = [r["pearson"]["unseen"].get(metric, {}).get("value") for r in rows]
        scatter(ax, seen, unseen, "Seen expression Pearson", "Unseen expression Pearson", 0)
        finite = [y for y in unseen if y is not None]
        negative = sum(y < 0 for y in finite)
        ax.plot([-1, 1], [-1, 1], color="#777", linestyle=":", lw=1)
        ax.set(
            xlim=(-1, 1), ylim=(-1, 1), title=f"{title} · unseen negative {negative}/{len(finite)}"
        )
    save(fig, "03-expression-exposure-gap")


def scatter(ax, xs, ys, xname, yname, baseline=None):
    points = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
    if points:
        x, y = zip(*points, strict=True)
        ax.scatter(x, y, s=8, alpha=0.35, linewidths=0)
    if baseline is not None:
        ax.axhline(baseline, color="#c54d34", lw=1)
    ax.set(xlabel=xname, ylabel=yname)
    ax.grid(alpha=0.18)


def condition_errors(rows, acceptance):
    fig, axes = grid(3)
    x = [r["truth_count"] for r in rows]
    y = [r["mse"] for r in rows]
    axes[0].hist(y, bins=30, color="#386a9e")
    axes[0].axvline(
        acceptance["candidate_mse"],
        color="#c54d34",
        label=f"Macro {acceptance['candidate_mse']:.5f}",
    )
    axes[0].legend(fontsize=7)
    axes[0].set(xlabel="Condition MSE", ylabel="Conditions")
    scatter(axes[1], x, y, "Truth cells", "Condition MSE")
    axes[1].set_xscale("log")
    worst = sorted(rows, key=lambda r: r["mse"], reverse=True)[:10]
    axes[2].barh(
        [r["condition"].removesuffix("+ctrl") for r in worst][::-1],
        [r["mse"] for r in worst][::-1],
        color="#b04a4a",
    )
    axes[2].set(xlabel="Condition MSE", title="Largest errors · follow-up candidates")
    save(fig, "04-condition-errors")


def effect_diagnostics(rows):
    values = [row["effects"] for row in rows]
    x = [row["truth_effect_rms"] for row in values]
    fig, axes = grid(6)
    scatter(
        axes[0], x, [r["pred_effect_rms"] for r in values], "True effect RMS", "Model effect RMS"
    )
    bound = max(max(x), max(r["pred_effect_rms"] for r in values))
    axes[0].plot([0, bound], [0, bound], color="#c54d34", lw=1)
    for ax, key, label, base in zip(
        axes[1:],
        (
            "effect_rms_ratio",
            "effect_cosine",
            "mse_gain_over_copy",
            "mse_gain_over_train_mean",
            "target_direction_accuracy",
        ),
        (
            "Pred / truth RMS",
            "Effect cosine",
            "MSE gain vs copy",
            "MSE gain vs train mean",
            "Target sign correct",
        ),
        (1, 0, 0, 0, None),
        strict=True,
    ):
        scatter(ax, x, [r[key] for r in values], "True effect RMS", label, base)
    save(fig, "05-amplitude-direction-baselines")
    order = sorted(range(len(x)), key=lambda i: (x[i], rows[i]["condition"]))
    chunks = np.array_split(order, 4)
    fig, axes = grid(6)
    definitions = (
        ("Truth effect RMS", x),
        ("Effect cosine", [r["effect_cosine"] for r in values]),
        ("MSE gain vs copy", [r["mse_gain_over_copy"] for r in values]),
        *(
            (label + " Pearson", [r["pearson"]["all"].get(m, {}).get("value") for r in rows])
            for m, label in METRICS.items()
        ),
    )
    for ax, (label, col) in zip(axes, definitions, strict=True):
        groups = [[col[i] for i in chunk if col[i] is not None] for chunk in chunks]
        for j, group in enumerate(groups, 1):
            if group:
                ax.boxplot(group, positions=[j], widths=0.65, showfliers=False)
                ax.text(
                    j,
                    0.03,
                    f"n={len(group)}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    fontsize=7,
                )
        ax.set(ylabel=label, xticks=range(1, 5), xticklabels=["Weak", "Q2", "Q3", "Strong"])
    save(fig, "06-truth-effect-strata")


def downstream_top(rows):
    keys = ("precision", "recall", "direction_accuracy", "truth_top_count", "predicted_top_count")
    table = ["| K | Metric | Model | Mean | Valid |", "| --- | --- | --- | ---: | ---: |"]
    fig, axes = grid(len(keys) * 2, columns=2)
    for i, key in enumerate(keys):
        for j, k in enumerate(("20", "50")):
            ax = axes[i * 2 + j]
            for m, model in enumerate(MODELS, 1):
                col = [r["effects"]["top_effects"][k][model][key] for r in rows]
                finite = [v for v in col if v is not None]
                avg = f"{np.mean(finite):.4f}" if finite else "NA"
                table.append(f"| {k} | {key} | {MODELS[model]} | {avg} | {len(finite)} |")
                if finite:
                    ax.boxplot(finite, positions=[m], widths=0.65, showfliers=False, showmeans=True)
                ax.text(
                    m,
                    0.04,
                    f"n={len(finite)}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    fontsize=7,
                )
            ax.set(
                title=f"Top {k} · {key.replace('_', ' ')}",
                xticks=range(1, 4),
                xticklabels=list(MODELS.values()),
            )
    save(fig, "01-top-effect-recovery")
    note("\n".join(table))


def downstream_retrieval(rows, metadata):
    n = metadata["valid_candidate_count"]
    if not n:
        note("所有真值方向均不可用，无法做扰动身份检索。")
        return
    keys = ("top1", "top5", "reciprocal_rank", "rank")
    references = (1 / n, min(5, n) / n, np.mean(1 / np.arange(1, n + 1)), (n + 1) / 2)
    table = ["| Metric | Model | Mean | Uniform | Valid |", "| --- | --- | ---: | ---: | ---: |"]
    fig, axes = grid(4, columns=2)
    for ax, key, chance in zip(axes, keys, references, strict=True):
        for i, model in enumerate(MODELS):
            col = [r["effects"]["retrieval"][model][key] for r in rows]
            finite = [v for v in col if v is not None]
            avg = f"{np.mean(finite):.4f}" if finite else "NA"
            table.append(f"| {key} | {MODELS[model]} | {avg} | {chance:.4f} | {len(finite)} |")
            if finite:
                ax.bar(i, np.mean(finite))
            ax.text(
                i,
                0.04,
                f"n={len(finite)}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=7,
            )
        ax.axhline(chance, color="#c54d34", label="Uniform random")
        ax.legend(fontsize=7)
        ax.set(title=key.replace("_", " "), xticks=range(3), xticklabels=list(MODELS.values()))
    save(fig, "02-perturbation-retrieval")
    note("\n".join(table))
    fig, axes = grid(3)
    for ax, model in zip(axes, MODELS, strict=True):
        col = [r["effects"]["retrieval"][model]["rank"] for r in rows]
        finite = sorted(v for v in col if v is not None)
        ax.step(finite, np.arange(1, len(finite) + 1) / max(1, len(finite)), where="post")
        ax.set(
            xlim=(1, max(2, n)),
            ylim=(0, 1.02),
            xlabel="Average tie rank",
            ylabel="Cumulative condition fraction",
            title=MODELS[model],
        )
    save(fig, "03-retrieval-rank-distributions")


def comparison_errors(rows):
    fig, axes = grid(3)
    for ax, model in zip(axes, MODELS, strict=True):
        col = [row["effects"]["mse"][model] for row in rows]
        ax.hist(col, bins=30, color="#386a9e")
        ax.axvline(np.mean(col), color="#c54d34", label="Macro mean")
        ax.legend(fontsize=7)
        ax.set(title=MODELS[model], xlabel="Pseudobulk MSE", ylabel="Conditions")
    save(fig, "07-pseudobulk-baseline-errors")


def missing_diagnostics():
    fig, ax = plt.subplots(figsize=(7, 1.7))
    ax.axis("off")
    ax.text(
        0.5,
        0.57,
        "Response summaries were not collected for this checkpoint.",
        va="center",
        ha="center",
        fontsize=13,
    )
    ax.text(
        0.5,
        0.31,
        "Top-effect recovery and perturbation retrieval are unavailable.",
        va="center",
        ha="center",
        fontsize=10,
    )
    save(fig, "00-unavailable")
