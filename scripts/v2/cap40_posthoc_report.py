# ruff: noqa: RUF001, E501
"""Render complete Chinese reports from sealed small aggregate artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cap40_posthoc import csv_write, read, sha, write
from cap40_posthoc_stats import condition_summary


def flatten(value, prefix=""):
    out = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            out.update(flatten(item, name))
        else:
            out[name] = json.dumps(item, ensure_ascii=False) if isinstance(item, list) else item
    return out


def table(rows):
    if not rows:
        return "未采集或不适用。\n"
    keys = list(dict.fromkeys(k for r in rows for k in r))

    def cell(x):
        if x is None:
            return "不可定义"
        if isinstance(x, float):
            return f"{x:.7g}"
        return str(x).replace("|", "\\|").replace("\n", " ")

    return (
        "\n".join(
            ["|" + "|".join(keys) + "|", "|" + "|".join(["---"] * len(keys)) + "|"]
            + ["|" + "|".join(cell(r.get(k)) for k in keys) + "|" for r in rows]
        )
        + "\n"
    )


def render(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.output
    if root.exists():
        raise ValueError("report requires fresh destination")
    completion = read(args.input / "COMPLETE.json")
    if completion["scope"] != "full_cap40_train" or not completion["state_unchanged"]:
        raise ValueError("report requires completed full-population immutable analysis")
    for name, entry in completion["outputs"].items():
        if sha(args.input / name) != entry["sha256"]:
            raise ValueError("analysis output checksum mismatch")
    summary, deep = read(args.input / "summary.json"), read(args.input / "deep.json")
    gradients = read(args.input / "gradients.json")
    old = read(args.original / "summary.json")
    if old["checkpoint_sha256"] != summary["checkpoint_sha256"]:
        raise ValueError("historical/expanded checkpoint identity mismatch")
    root.mkdir(parents=True)
    tables = {}
    for key, source in (
        ("heads", summary["global_heads"]),
        ("features", summary["global_features"]),
        ("outputs", summary["metrics"]),
        ("cls_decomposition", summary["cls_decomposition"]),
    ):
        tables[key] = [{"name": k, **flatten(v)} for k, v in source.items()]
    tables["centers"] = [{"head": k, **v} for k, v in summary.get("centers", {}).items()]
    for g in deep:
        p = g["condition_id"]
        for key in ("pairs", "same_gene", "cross_view", "per_cell"):
            tables.setdefault(key, []).extend({"condition_id": p, **r} for r in g[key])
        for key in ("heads", "features"):
            tables.setdefault("condition_" + key, []).extend(
                {"condition_id": p, "name": k, **v} for k, v in g[key].items()
            )
        tables.setdefault("frozen_losses", []).append(
            {
                "condition_id": p,
                "joint": g["joint"],
                "truth_rows": g["independent_truth_rows"],
                "controls": g["independent_controls"],
                **g["loss_terms"],
            }
        )
    tables["gradients"] = [{"condition_id": g["condition_id"], **flatten(g)} for g in gradients]
    pairs = {}
    for name in sorted({r["student"] for r in tables["pairs"]}):
        rows = [r for r in tables["pairs"] if r["student"] == name]
        for metric in ("CE", "H_pair", "KL"):
            pairs.setdefault(name, {})[metric] = condition_summary(rows, metric)
    tables["pair_summary"] = [{"name": k, **flatten(v)} for k, v in pairs.items()]
    per_cell = []
    for name in sorted({r["feature"] for r in tables["per_cell"]}):
        rows = [r for r in tables["per_cell"] if r["feature"] == name]
        per_cell.append(
            {
                "feature": name,
                **flatten(
                    {
                        m: condition_summary(rows, m)
                        for m in ("effective_rank", "relative_variance", "variance", "mean_norm")
                    }
                ),
            }
        )
    tables["per_cell_summary"] = per_cell
    for key, rows in tables.items():
        csv_write(root / f"{key}.csv", rows)
    figures = []

    def save(name, fig, caption):
        fig.tight_layout()
        fig.savefig(root / name, dpi=150)
        plt.close(fig)
        figures.append((name, caption))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(
        ["Full rows", "Deep rows", "Gradient rows"],
        [
            summary["rows"],
            summary["deep_rows"],
            sum(g.get("performed", False) * 4 for g in gradients),
        ],
    )
    ax.set(
        ylabel="Cell records",
        title=f"{summary['arm']}: diagnostic coverage (not independent view count)",
    )
    save(
        "05_coverage.png",
        fig,
        "全量输出覆盖全部训练行；深度探针覆盖全部条件，梯度仅固定条件子集。视图不增加独立细胞数，实际去重control数见清单。",
    )
    names = list(summary["global_heads"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, field in zip(axes, ("H_sample", "JS"), strict=True):
        ax.bar(range(len(names)), [summary["global_heads"][k].get(field, 0) for k in names])
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set(ylabel="nats", title=field)
    save(
        "06_full_prototypes.png",
        fig,
        "逐头、角色、视图统计。H_sample反映单样本软硬程度；JS反映样本间差异。SSL1按条件或遮蔽图节点计数，SSL2按细胞或遮蔽表达token计数；这些数量不能混称细胞数。JS由全体累计概率计算，不是条件JS均值。",
    )
    fs = summary["global_features"]
    names = list(fs)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(names) * 0.14), 9))
    for ax, field in zip(axes, ("effective_rank", "relative_variance"), strict=True):
        ax.bar(range(len(names)), [fs[k].get(field) or 0 for k in names])
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
        ax.set(title=field)
    save(
        "07_full_representations.png",
        fig,
        "这是混合样本轴的全局协方差汇总，不能替代同细胞基因轴的秩或同基因跨control的方差。各层主视图、最终各视图和投影头分开。协方差为float64流式中心化，mHC组件输出取stream均值。",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    with (args.input / "outputs_full.csv").open() as f:
        outputs = list(csv.DictReader(f))
    for ax, field in zip(axes, ("prediction_mse", "effect_scale_slope_no_intercept"), strict=True):
        x = [float(r[field]) for r in outputs if r[field] not in ("", "None")]
        ax.hist(x, bins=60)
        ax.set(title=field, ylabel="Truth rows")
    save(
        "08_full_outputs.png",
        fig,
        "固定训练query子集的逐细胞误差和相对control效应幅度；非正式测试Pearson。斜率1表示投影到真实效应方向上的尺度相当；不能据此单独判断异质性或因果响应。四种输出负值/零值/复制误差等完整数值见表。",
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    gs = []
    for g in gradients:
        for group, value in g.get("groups", {}).items():
            for pair, cos in value["cosine"].items():
                if cos is not None:
                    gs.append((group + ":" + pair, cos))
    labels = sorted({k for k, v in gs})
    if labels:
        ax.boxplot([[v for k, v in gs if k == label] for label in labels], tick_labels=labels)
    ax.tick_params(axis="x", rotation=40)
    ax.set(ylabel="Gradient cosine", ylim=(-1, 1))
    save(
        "09_grouped_gradients.png",
        fig,
        "固定64个条件、每条件最多4个细胞，在共享图参数上比较加权预测与SSL梯度。没有optimizer/EMA/center更新，也没有梯度写入param.grad。负余弦表示该probe中的局部方向冲突，不等于蒸馏无效。",
    )
    sections = [
        f"# {summary['arm']}：cap40 全量与分组补测报告\n",
        "## 1. 结论与证据范围\n仅重载第4轮last.pt，无重训、无选参。全量指cap40训练细胞全集，不是未经下采样的原始数据，也不是所有control×truth的笛卡尔积。\n",
        "## 2. 版本与校验\n"
        + table(
            [
                {
                    k: v
                    for k, v in summary.items()
                    if k
                    in (
                        "arm",
                        "method_sha",
                        "checkpoint_sha256",
                        "checkpoint_epoch",
                        "manifest_sha256",
                        "config_sha256",
                        "resolved_config_sha256",
                    )
                },
                {"analysis_commit": summary["analysis"]["commit"]},
            ]
        ),
        "## 3. 实验改动与配置\n"
        + table(
            [
                {
                    "arm": summary["arm"],
                    "method": {
                        "DP": "DINO式weight_norm与首轮原型冻结",
                        "FN": "统一角色MLP Pre-LayerNorm",
                        "FG": "统一GELU、主干参数匹配",
                        "FB": "角色FFN Linear去bias",
                        "PO": "Softplus绝对非负输出",
                    }[summary["arm"]],
                }
            ]
        ),
        "冻结配置全文见原始resolved_config；补测不改变架构、数值表达剔除、视图比例、权重或教师统计。\n",
        "## 4. 共同样本清单与覆盖\n"
        + table([{k: summary[k] for k in ("rows", "conditions", "deep_rows", "scope")}]),
        "同一扰动/不同扰动分开；context=cell_type::batch。只有Jurkat一个cell_line，不能计算跨细胞系效应。原始清单及真实row/control/gene/view/mask对应关系留在服务器，hash绑定所有实验。\n",
        "## 5. 统计定义、分组与不确定性\n有效秩=exp(−Σλ̃ logλ̃)，λ来自中心化协方差。relative variance=tr(C)/(tr(C)+‖mean‖²)。同细胞基因秩采样最多128个按ID哈希选择的基因，秩上限127；不采用前缀。CLI评估模式固定原有eval顺序，不能将此当作随机顺序鲁棒性实验。条件均值和95%区间以扰动条件为bootstrap单位（1000次），不将view或token当独立细胞。\n",
        "## 6. 历史训练过程\n历史全量训练日志与四细胞旧probe原样保留，不能用当前last诊断补写过去epoch。\n",
    ]
    old_csvs = []
    for filename in (
        "training.csv",
        "heads.csv",
        "pairs.csv",
        "features.csv",
        "outputs.csv",
        "gradients.csv",
        "metrics.csv",
    ):
        path = args.original / filename
        with path.open() as f:
            old_rows = list(csv.DictReader(f))
        old_csvs.append((filename, old_rows))
        if filename == "training.csv":
            sections.append(table(old_rows))
    sections.extend(
        [
            "## 7. 四头原型、CE/H/KL与center\n"
            + table(tables["heads"])
            + "\n"
            + table(tables["pair_summary"])
            + table(tables["centers"]),
            "CE与Teacher熵使用同一真实Student–Teacher配对；iBOT先token后有效cell平均，raw_token列另保留token均值；KL=CE−H。iBOT内部同一次forward的重复CE求值不重复计数。下表为条件等权汇总，原型全局表则按该头有效行累计，聚合轴不同。center原值/历史轨迹只引用原有记录，不制造新更新。\n",
            "## 8. 分组表征与跨视图\n"
            + table(tables["features"])
            + table(tables["per_cell_summary"])
            + table(tables["cls_decomposition"]),
            "within-condition部分固定condition和context并去重control；between部分单列条件均值。全局秩包含基因身份与条件差异，不能直接视为细胞区分能力；同基因跨control与同细胞跨view详表分别见same_gene.csv/cross_view.csv。\n",
            "## 9. 输出与效应校准\n" + table(tables["outputs"]),
            "## 10. 梯度与效率\n"
            + table([{k: summary[k] for k in ("elapsed_seconds", "peak_allocated_mib", "updates")}])
            + table(tables["gradients"]),
            "## 11. 原有正式last测试18项\n" + table(dict(old_csvs)["metrics.csv"]),
            "## 12. 解释与限制\n表征是否区分：结合全局、同细胞基因、同基因跨control、同条件CLS和跨view共同判断。低loss来源：结合H与KL，不能只看CE。输出泛化：以原有all/seen/unseen测试结果为准；本训练probe保留表达剔除，不能冒称补测了未见表达的预测标签。梯度：仅局部诊断，无长期更新证明。\n"
            + "\n".join("- " + s for s in summary["limits"])
            + "\n",
            "## 13. 全部图与分析数据\n",
        ]
    )
    # Existing four plots retain bytes and remain explicitly historical.
    import shutil

    for filename, caption in (
        ("01_losses.png", "历史训练损失与旧小probe，具体样本量见原始报告。"),
        ("02_prototypes.png", "历史四细胞/前缀原型探针，不能当成新全量结果。"),
        ("03_representations.png", "历史小probe表征，横向比较须看真实坐标与采样轴。"),
        ("04_outputs_gradients.png", "历史输出/梯度固定probe，不是全量数据。"),
    ):
        shutil.copyfile(args.original / filename, root / filename)
        sections.append(f"\n![{filename}]({filename})\n\n{caption}\n")
    for filename, caption in figures:
        sections.append(f"\n![{filename}]({filename})\n\n{caption}\n")
    historical = root / "historical"
    historical.mkdir()
    for filename, rows in old_csvs:
        shutil.copyfile(args.original / filename, historical / filename)
        sections.append("\n### 历史已保存明细：" + filename + "\n" + table(rows))
    for filename in ("summary.json", "receipt.json"):
        shutil.copyfile(args.original / filename, historical / filename)
    shutil.copyfile(args.input / "summary.json", root / "analysis_summary.json")
    # Copy scalar result tables only. Never pull raw tensors or checkpoints.
    for filename in ("outputs_full.csv", "cls_within_condition.csv"):
        shutil.copyfile(args.input / filename, root / filename)
    files = [
        {"file": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    sections.append(
        "\n所有图均实际嵌入；CSV包含完整明细，正文为共同汇总表。逐细胞标量不是原始表达/激活矩阵。\n"
        + table(files)
    )
    (root / "report.md").write_text("\n\n".join(sections))
    write(
        root / "receipt.json",
        {
            "schema": "cap40-posthoc-report-1",
            "status": "completed",
            "input_complete_sha256": sha(args.input / "COMPLETE.json"),
            "manifest_sha256": summary["manifest_sha256"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "analysis": summary["analysis"],
            "report_script_sha256": sha(__file__),
            "outputs": {
                str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                for p in sorted(root.rglob("*"))
                if p.is_file()
            },
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("input", "original", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    render(p.parse_args())
