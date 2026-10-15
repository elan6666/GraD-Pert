# ruff: noqa: RUF001, E501
"""Same-manifest five-arm comparison; small aggregate inputs, no model loading."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from cap40_posthoc import csv_write, read, sha, write
from cap40_posthoc_report import table
from cap40_posthoc_stats import condition_summary


def compare(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = read(args.queue)
    out = args.output
    if out.exists():
        raise ValueError("comparison requires fresh output")
    arms = [r["arm"] for r in q["rows"]]
    if arms != ["DP", "FN", "FG", "FB", "PO"]:
        raise ValueError("wrong comparison family")
    summaries = {}
    metrics = []
    heads = []
    features = []
    outputs = []
    rows = {}
    sources = []
    for row in q["rows"]:
        arm = row["arm"]
        report = Path(row["report"])
        rec = read(report / "receipt.json")
        if (
            rec["manifest_sha256"] != q["manifest_sha256"]
            or rec["checkpoint_sha256"] != row["checkpoint_sha256"]
        ):
            raise ValueError("comparison identity mismatch")
        for name, entry in rec["outputs"].items():
            if sha(report / name) != entry["sha256"]:
                raise ValueError("report artifact hash mismatch")
        s = read(report / "analysis_summary.json")
        summaries[arm] = s
        sources.append(
            {
                "arm": arm,
                "training_sha": s["method_sha"],
                "analysis_sha": s["analysis"]["commit"],
                "checkpoint_epoch": s["checkpoint_epoch"],
                "checkpoint_sha256": s["checkpoint_sha256"],
                "report_receipt_sha256": sha(report / "receipt.json"),
            }
        )
        with (report / "historical/metrics.csv").open() as f:
            metrics.extend({"arm": arm, **r} for r in csv.DictReader(f))
        for key, v in s["global_heads"].items():
            heads.append({"arm": arm, "head_role_view": key, **v})
        for key, v in s["global_features"].items():
            features.append({"arm": arm, "feature": key, **v})
        for key, v in s["metrics"].items():
            outputs.append({"arm": arm, "metric": key, **v})
        with (report / "outputs_full.csv").open() as f:
            rows[arm] = list(csv.DictReader(f))
    if (
        len({(s["scope"], s["rows"], s["conditions"], s["deep_rows"]) for s in summaries.values()})
        != 1
    ):
        raise ValueError("analysis scope or population differs across arms")
    reference = rows["DP"]

    def identity(r):
        return (r["condition_id"], r["truth_row_id"], r["control_row_id"])

    if any([identity(r) for r in rows[a]] != [identity(r) for r in reference] for a in arms):
        raise ValueError("row/control alignment differs across arms")
    paired = []
    for arm in arms:
        diff = [
            {
                "condition_id": r["condition_id"],
                "difference": float(r["prediction_mse"]) - float(b["prediction_mse"]),
            }
            for r, b in zip(rows[arm], reference, strict=True)
        ]
        paired.append(
            {
                "arm": arm,
                "reference": "DP",
                "metric": "prediction MSE difference",
                **condition_summary(diff, "difference"),
            }
        )
    out.mkdir(parents=True)
    for name, values in (
        ("sources", sources),
        ("test18", metrics),
        ("heads", heads),
        ("features", features),
        ("outputs", outputs),
        ("paired_condition_mse", paired),
    ):
        csv_write(out / (name + ".csv"), values)
    figures = []

    def save(name, fig, caption):
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        figures.append((name, caption))

    keys = sorted({(r["group"], r["metric_id"]) for r in metrics})
    matrix = np.array(
        [
            [
                float(
                    next(
                        r["macro_mean"]
                        for r in metrics
                        if r["arm"] == a and (r["group"], r["metric_id"]) == k
                    )
                )
                for a in arms
            ]
            for k in keys
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(matrix, aspect="auto", vmin=-0.2, vmax=0.8, cmap="coolwarm")
    ax.set_xticks(range(5), arms)
    ax.set_yticks(range(len(keys)), [" / ".join(k) for k in keys], fontsize=7)
    fig.colorbar(im, ax=ax)
    save(
        "C01_formal_test18.png",
        fig,
        "复用原有第4轮last正式测试，三Pearson的all/seen/unseen与全基因/DEG全部展示；有效条件数逐项见表。此图不是本次补测的新测试。",
    )
    hk = sorted({r["head_role_view"] for r in heads if r["head_role_view"].startswith("teacher")})
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    for ax, field in zip(axes, ("H_sample", "JS"), strict=True):
        for arm in arms:
            ax.plot(
                range(len(hk)),
                [
                    next(r[field] for r in heads if r["arm"] == arm and r["head_role_view"] == k)
                    for k in hk
                ],
                marker=".",
                label=arm,
            )
        ax.set_xticks(range(len(hk)), hk, rotation=60, fontsize=7)
        ax.set_ylabel(field + " (nats)")
        ax.legend()
    save(
        "C02_teacher_H_JS.png",
        fig,
        "Teacher各头/视图H与JS保持分开，未将不同view或头揉成一个总熵。低H低JS可能是共同集中，低H高JS可能是样本特异；均需结合CE/KL和表征判断。",
    )
    fk = sorted(
        {
            r["feature"]
            for r in features
            if ".main." in r["feature"]
            and ("cls" in r["feature"] or "response_tokens" in r["feature"])
        }
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    for arm in arms:
        ax.plot(
            range(len(fk)),
            [
                next(r["effective_rank"] for r in features if r["arm"] == arm and r["feature"] == k)
                for k in fk
            ],
            label=arm,
            marker=".",
        )
    ax.set_xticks(range(len(fk)), fk, rotation=90, fontsize=6)
    ax.set_ylabel("Global centered effective rank")
    ax.legend()
    save(
        "C03_main_representations.png",
        fig,
        "混合细胞/基因的主路径全局秩，不等于同细胞的基因秩。与每臂per_cell_summary、same_gene、CLS组内组间分解共同阅读；各实验共用实际采样ID。",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    means = [p["condition_mean"] for p in paired]
    lower = [m - p["ci95"][0] if p["ci95"] else 0 for m, p in zip(means, paired, strict=True)]
    upper = [p["ci95"][1] - m if p["ci95"] else 0 for m, p in zip(means, paired, strict=True)]
    ax.errorbar(arms, means, yerr=[lower, upper], fmt="o")
    ax.axhline(0, color="grey")
    ax.set_ylabel("Paired condition MSE difference vs DP")
    save(
        "C04_paired_output_mse.png",
        fig,
        "相同truth/control/query的训练输出MSE差值，先在每个条件内平均，再条件等权；95%区间以条件bootstrap。负值仅表示该训练输入诊断中误差低于DP，不作为测试选参证据。",
    )
    text = [
        "# DP/FN/FG/FB/PO：cap40 同清单分组补测总报告\n",
        "五组使用同一份实际row/control/query/view/mask清单，加载原last.pt，只读前向和autograd探针，无重训或状态更新。\n",
        "## 版本与覆盖\n" + table(sources),
        "共同清单SHA：`"
        + q["manifest_sha256"]
        + f"`。各臂同清单{summaries['DP']['rows']}行/{summaries['DP']['conditions']}条件；范围{summaries['DP']['scope']}，深度{summaries['DP']['deep_rows']}行，梯度条件数见各臂报告。中等规模不冒称全量总体；单Jurkat不能估计跨细胞系差异。\n",
        "## 正式测试全部18指标与有效条件数\n" + table(metrics),
        "## 当前清单输出与条件配对差值\n" + table(outputs) + table(paired),
        "## 四头原型：Student/Teacher/视图完整表\n" + table(heads),
        "## 模块表示完整汇总\n" + table(features),
        "## 读图与解释边界\nH、JS、KL与有效秩衡量不同对象，不能将低CE或单独低秩判成学习成功/坍塌。报告保留同细胞基因、同基因跨control、同条件同context、不同条件均值、跨view各自的表。冻结last补测不是过去epoch的训练激活轨迹；历史数据另标。\n",
        "## 新增横向图\n",
    ]
    for name, caption in figures:
        text.append(f"![{name}]({name})\n\n{caption}\n")
    text.append(
        "## 每臂全部图和完整分析数据\n以下45张图逐一保留；各臂报告正文展开全部历史数据与新增汇总，CSV记录全量明细、有效数量及hash。大明细只在服务器保留时须在交付清单注明，不冒称已下载。\n"
    )
    for row in q["rows"]:
        report = Path(row["report"])
        rel = os.path.relpath(report, out)
        text.append(f"### {row['arm']}\n\n[完整中文报告]({rel}/report.md)\n")
        for p in sorted(report.glob("*.png")):
            text.append(
                f"![{row['arm']} {p.name}]({rel}/{p.name})\n\n{row['arm']}：{p.name}；完整轴/分组/样本量说明见本臂报告，旧01–04与新增05–09区分。SHA256 `{sha(p)}`。\n"
            )
        rec = read(report / "receipt.json")
        text.append(
            table([{"file": str(report / name), **entry} for name, entry in rec["outputs"].items()])
        )
    (out / "report.md").write_text("\n\n".join(text))
    write(
        out / "receipt.json",
        {
            "status": "completed",
            "manifest_sha256": q["manifest_sha256"],
            "sources": sources,
            "outputs": {
                p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                for p in sorted(out.iterdir())
                if p.is_file()
            },
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    compare(p.parse_args())
