# ruff: noqa: RUF001, E501
"""Render complete Chinese reports from sealed small aggregate artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cap40_posthoc import csv_write, read, sha, write
from cap40_posthoc_stats import condition_summary

ARM_DESCRIPTIONS = {
    "R11-READOUT": "R9原prediction MLP额外300步适配；主干/Teacher/centers冻结，SSL停用",
    "PT2": "前2轮SSL预训练＋后4轮MSE全量微调；两端点共用中等诊断清单",
    "PT1": "前2轮仅SSL1/SSL2预训练，后2轮仅MSE全量微调；边界重建优化器",
    "DP": "DINO式weight_norm与首轮原型冻结",
    "FN": "统一角色MLP Pre-LayerNorm",
    "FG": "统一GELU、主干参数匹配",
    "FB": "角色FFN Linear去bias",
    "PO": "Softplus绝对非负输出",
    "B0": "新锚点：FG GELU＋PO非负预测，SSL1 node关闭",
    "B0-DP": "B0＋原型weight_norm及首轮零学习率冻结；保留历史Adam动量累积",
    "DP2": "B0-DP＋固定原型尺度、首轮去梯度冻结、Teacher降温及更大EMA",
    "D0": "B0同时关闭SSL1和SSL2，仅预测MSE；未调用蒸馏头不作坍塌分析",
    "VH": "两套蒸馏Global80%–100%／Local40%–65%",
    "S12-L4": "SSL1两GO＋两STRING Local；SSL2四随机Local",
    "CG1": "control摘要只调节图第四层的来源key门",
    "U24": "共享GenePT映射＋Softplus之前的基因特异读出修正",
    "MR1": "隐藏25% control表达后预测扰动表达，辅助权重1",
    "P1": "同条件群体均值MSE＋MMD，替代随机逐行MSE",
    "C1": "仅入口注入ep，保留control cross和输出拼接",
    "O1": "所有KDA固定真实基因ID顺序、最终状态读取",
    "K2-fixed-prefix": "所有KDA固定真实基因ID顺序并读取对应前缀状态",
}


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


def gradient_caption(gradients):
    if gradients and all(not g.get("performed", False) for g in gradients):
        reasons = sorted({g.get("reason", "not_recorded") for g in gradients})
        return (
            "图梯度不可用（"
            + ",".join(reasons)
            + "）：主干冻结；不构造SSL梯度、范数或余弦，NA不替换为0。"
        )
    return (
        f"固定{len(gradients)}个条件、每条件最多4个细胞，在共享图参数上比较加权预测与SSL梯度。"
        "没有optimizer/EMA/center更新，也没有梯度写入param.grad。"
        "负余弦表示该probe中的局部方向冲突，不等于蒸馏无效。"
    )


def render(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = args.output
    if root.exists():
        raise ValueError("report requires fresh destination")
    completion = read(args.input / "COMPLETE.json")
    if (
        completion["scope"] not in ("full_cap40_train", "medium_cap40_stratified")
        or not completion["state_unchanged"]
    ):
        raise ValueError("report requires completed preregistered immutable analysis")
    for name, entry in completion["outputs"].items():
        if sha(args.input / name) != entry["sha256"]:
            raise ValueError("analysis output checksum mismatch")
    summary, deep = read(args.input / "summary.json"), read(args.input / "deep.json")
    scope_name = "中等规模" if summary["scope"] == "medium_cap40_stratified" else "全量"
    scope_text = summary.get("manifest_scope", "cap40训练全集；深度覆盖所有条件")
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
        for key in ("pairs", "same_gene", "cross_view", "per_cell", "mask_eligibility"):
            tables.setdefault(key, []).extend({"condition_id": p, **r} for r in g.get(key, []))
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
    grouped_path = args.input / "grouped_views.json"
    grouped = read(grouped_path) if grouped_path.exists() else None
    if grouped is not None:
        tables["grouped_view_pairs"] = grouped["pairs"]
        tables["grouped_view_features"] = grouped["features"]
    for key, rows in tables.items():
        csv_write(root / f"{key}.csv", rows)
    figures = []

    def save(name, fig, caption):
        fig.tight_layout()
        fig.savefig(root / name, dpi=150)
        plt.close(fig)
        figures.append((name, caption))

    if grouped is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, field in zip(axes, ("cosine", "centered_cosine"), strict=True):
            rows = [
                r
                for r in grouped["pairs"]
                if r["role"] == "student" and r["branch"] == "ssl2_response_cls" and r["pair_count"]
            ]
            labels = sorted({r["stratum"] + ":" + r["view_family"] for r in rows})
            values = [
                [
                    r[field + "_condition_balanced_mean"]
                    for r in rows
                    if r["stratum"] + ":" + r["view_family"] == label
                    and r[field + "_condition_balanced_mean"] is not None
                ]
                for label in labels
            ]
            keep = [(label, v) for label, v in zip(labels, values, strict=True) if v]
            if keep:
                ax.boxplot([v for _, v in keep], tick_labels=[k for k, _ in keep])
                ax.tick_params(axis="x", rotation=90, labelsize=7)
            ax.set(title=field, ylabel="Cosine similarity (unitless)")
        save(
            "14_condition_global_local.png",
            fig,
            "同一样本、同条件不同control、异条件不同control、异条件同control分别比较GG/GL/LL。"
            "每个箱内值为一个context×视图对的条件等权均值，仅作描述；真实样本/条件/配对数见CSV。"
            "右图减去同分支同角色同context的Global共同中心，帮助识别共同偏移造成的高余弦；不是质量评分。",
        )

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(
        ["Basic rows", "Deep rows", "Gradient rows"],
        [
            summary["rows"],
            summary["deep_rows"],
            sum(g.get("cells", 0) for g in gradients if g.get("performed", False)),
        ],
    )
    ax.set(
        ylabel="Cell records",
        title=f"{summary['arm']}: diagnostic coverage (not independent view count)",
    )
    save(
        "05_coverage.png",
        fig,
        f"{scope_name}覆盖：基础{summary['rows']}行/{summary['conditions']}条件；深度{summary['deep_rows']}行/{summary.get('deep_conditions', len(deep))}条件。视图不增加独立细胞数。{scope_text}",
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
        gradient_caption(gradients),
    )
    sections = [
        f"# {summary['arm']}：cap40 {scope_name}与分组补测报告\n",
        f"## 1. 结论与证据范围\n仅重载第{summary['checkpoint_epoch']}轮"
        f"{summary.get('checkpoint_role', 'last')} checkpoint，无重训、无选参。"
        + (
            "此端点预测头尚未训练；数值输出仅为未训练读出诊断，不是受监督预测成绩。"
            if summary.get("checkpoint_role") == "pretrain"
            else ""
        )
        + scope_text
        + "。中等规模不是全部训练细胞；深度统计只描述固定分层条件子集，不冒称全体条件总体估计。\n",
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
                    "method": ARM_DESCRIPTIONS[summary["arm"]],
                }
            ]
        ),
        "冻结配置全文见原始resolved_config；补测不改变架构、数值表达剔除、视图比例、权重或教师统计。\n",
        "图全基因轴："
        + summary.get("graph_full_axis_semantics", "静态H4图表示")
        + "。CG1的全图共享H3不能当作其他组的H4；其control条件化H4在深度视图中另列。\n"
        + "实际SSL权重："
        + json.dumps(summary.get("loss_weights", {}), ensure_ascii=False)
        + "。未调用的node头不填造分布或CE。\n",
        "## 4. 共同样本清单与覆盖\n"
        + table([{k: summary[k] for k in ("rows", "conditions", "deep_rows", "scope")}]),
        "同一扰动/不同扰动分开；context=cell_type::batch。只有Jurkat一个cell_line，不能计算跨细胞系效应。原始清单及真实row/control/gene/view/mask对应关系留在服务器，hash绑定所有实验。\n",
        "## 5. 统计定义、分组与不确定性\n有效秩=exp(−Σλ̃ logλ̃)，λ来自中心化协方差。relative variance=tr(C)/(tr(C)+‖mean‖²)。同细胞基因秩采样最多128个按ID哈希选择的基因，秩上限127；不采用前缀。CLI评估模式固定原有eval顺序，不能将此当作随机顺序鲁棒性实验。条件均值和95%区间以扰动条件为bootstrap单位（1000次），不将view或token当独立细胞。\n",
        "## 6. 历史训练过程\n历史训练日志与四细胞旧probe原样保留；"
        "只展示所选端点之前的记录，不用当前checkpoint补写过去epoch。\n",
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
            "within-condition部分固定condition和context并去重control；CLS的between部分固定query panel和context再比较条件均值，避免不同查询基因组成造成混淆。control_treatment仍为control，paired_perturbation只是配对标签。全局秩包含基因身份与条件差异，不能直接视为细胞区分能力；同基因跨control与同细胞跨view详表分别见same_gene.csv/cross_view.csv。每条件最多4细胞时CLS条件内秩上限很低，应看有效N。\n",
            "## 9. 输出与效应校准\n" + table(tables["outputs"]),
            "## 10. 梯度与效率\n"
            + table([{k: summary[k] for k in ("elapsed_seconds", "peak_allocated_mib", "updates")}])
            + table(tables["gradients"]),
            "## 11. 原有正式last测试18项\n"
            + (
                "预训练端点不做正式预测测试，不复用最终轮的指标。\n"
                if summary.get("checkpoint_role") == "pretrain"
                else table(dict(old_csvs)["metrics.csv"])
            ),
            "## 12. 解释与限制\n表征是否区分：结合全局、同细胞基因、同基因跨control、同条件CLS和跨view共同判断。低loss来源：结合H与KL，不能只看CE。输出泛化：以原有all/seen/unseen测试结果为准；本训练probe保留表达剔除，不能冒称补测了未见表达的预测标签。梯度：仅局部诊断，无长期更新证明。\n"
            + "\n".join("- " + s for s in summary["limits"])
            + "\n",
            "## 13. 全部图与分析数据\n",
        ]
    )
    if grouped is not None:
        sections.insert(
            -1,
            "## 12a. 同/异扰动条件的Global与Local共同分析\n"
            "SSL1分析扰动条件向量ep，静态图每条件只计一个样本；SSL2分析response CLS，"
            "同一condition内重复control只保留冻结清单第一行。先固定细胞系、细胞类型和context=batch，"
            "再分同一样本/同条件不同control/异条件不同control/异条件同control。"
            "各类均列GG、GL、LL和每个实际view ID；空组或零范数不可定义，不补零。"
            "表中pair_mean是配对均值，condition_balanced_mean先在条件或条件对内平均再等权。"
            "配对共享细胞，不能用配对数充当独立N或进行独立样本显著性检验。"
            "MSE-only阶段的Student跨视图仅为只读表征探针，不伪造Teacher目标、熵或蒸馏CE。\n"
            + table(grouped["features"])
            + "\n完整分组配对数据：[grouped_view_pairs.csv](grouped_view_pairs.csv)；"
            "按视图的有效秩/相对方差：[grouped_view_features.csv](grouped_view_features.csv)。\n",
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
