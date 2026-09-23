"""Build the figure and aggregate-data appendix from sealed server summaries."""

# ruff: noqa: RUF001, E501 - Chinese prose and table labels are intentional.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "docs/experiments/data/report-aggregate-current-ecdf8f1.json"
DEFAULT_OUTPUT = ROOT / "docs/experiments/PERTURBATION_DATASET_ANALYSIS_APPENDIX_ZH.md"

DATASETS = [
    ("replogle_k562_essential", "Replogle K562", "K562"),
    ("replogle_rpe1_essential", "Replogle RPE1", "RPE1"),
    ("nadig_jurkat", "Nadig Jurkat", "jurkat"),
    ("nadig_hepg2", "Nadig HepG2", "hepg2"),
    ("norman", "Norman", "K562"),
]
CONTEXTS = [
    *DATASETS,
    *[("crosscell", f"固定轴 {line}", line) for line in ("K562", "RPE1", "jurkat", "hepg2")],
]
PROGRAMS = [
    ("dna_replication", "DNA 复制"),
    ("dna_repair", "DNA 修复"),
    ("g1s_transition", "G1/S"),
    ("g2m_transition", "G2/M"),
    ("apoptotic_process", "细胞凋亡"),
    ("oxidative_stress", "氧化应激"),
    ("ribosome_biogenesis", "核糖体生成"),
    ("translation", "翻译"),
]


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list[object]]) -> list[str]:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    result.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return [*result, ""]


def _line(data: dict, source: str, dataset: str, line: str) -> dict:
    return data[source][dataset]["cell_lines"][line]


def build(data_path: Path, output: Path) -> None:
    data = json.loads(data_path.read_text())
    checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
    lines = [
        "# 数据附录：全部图与聚合统计",
        "",
        "本附录补齐本轮五项观察性分析实际生成的全部 12 张图，以及其聚合数值。图 A1-A10 为五个独立数据集各自的 PCA/UMAP 和条件拆半图；正文图 1-2 为景观图与跨细胞热图。后续表格按数据来源分别列出统计量，不把固定轴缓存当作独立基准。逐条件 JSON 包含数千条记录，仍保存在服务器收据所列目录，未压成不可读的 PDF 表格。",
        "",
        f"聚合源文件：`docs/experiments/data/{data_path.name}`；SHA256 `{checksum}`。同一 JSON 也内嵌在 PDF 附件中。它由五项完成的服务器运行导出，只移除了逐条件明细和每批次 control 计数；各运行收据哈希写在源文件的 `sources` 字段。图 A1-A10 是原始分析 PNG 的等内容、印刷分辨率 JPEG 副本。",
        "",
        "## A. EDA：细胞组成、PCA 与测试集拆半",
        "",
    ]
    eda = {row["dataset_id"]: row for row in data["eda"]}
    rows = []
    for dataset, label, _ in DATASETS:
        d = eda[dataset]
        c = d["class_cells"]
        rows.append(
            [
                label,
                d["n_batches"],
                f"{c['control']:,}/{c['single']:,}/{c['double']:,}",
                _fmt(d["pca_variance_explained_2"]),
                d["n_genes"],
            ]
        )
    lines += _table(["数据集", "批次数", "control／单／双细胞", "前两 PC 方差比", "表达基因"], rows)
    rows = []
    for dataset, label, _ in DATASETS:
        d = eda[dataset]
        key = "double" if dataset == "norman" else "single"
        delta = d["split_half"][f"{key}_pearson_delta"]
        expression = d["split_half"][f"{key}_pearson_expression"]
        rows.append(
            [
                label,
                delta["n"],
                _fmt(delta["median"]),
                f"{_fmt(delta['p05'])}/{_fmt(delta['p95'])}",
                _fmt(expression["median"]),
            ]
        )
    lines += _table(
        ["数据集", "条件×重复数", "效应 r 中位", "效应 r P05／P95", "表达 r 中位"], rows
    )
    lines.append(
        "这两表对应图 A1-A10；拆半仅在冻结测试条件计算，重复五次。Norman 用双扰动条件，其他四项用单扰动条件。前两 PC 方差是可视化抽样的描述，不用于模型评价。"
    )
    lines.append("")
    lines += ["## B. 景观：批次、响应与检索", ""]
    rows = []
    for dataset, label, line in CONTEXTS:
        v = _line(data, "landscape", dataset, line)
        batch, ctrl = v["batch_information"], v["control_correlation"]
        rows.append(
            [
                label,
                _fmt(batch["batch_information_fraction"], 4),
                _fmt(batch["permuted_null_fraction"], 4),
                _fmt(batch["excess_batch_information_fraction"], 4),
                _fmt(ctrl["within_batch_median"], 4),
                _fmt(ctrl["across_batch_median"], 4),
            ]
        )
    lines += _table(
        ["数据来源", "标签-批次关联", "置换基线", "超额关联", "同批 control r", "跨批 control r"],
        rows,
    )
    rows = []
    for dataset, label, line in CONTEXTS:
        v = _line(data, "landscape", dataset, line)
        retrieval = v["replicate_retrieval"]
        rows.append(
            [
                label,
                _fmt(v["effect_rms_median"]),
                _fmt(v["effect_rms_p90"]),
                _fmt(v["general_response_pearson_median"]),
                retrieval["conditions"],
                _fmt(retrieval["top1"]),
                _fmt(retrieval["top10"]),
            ]
        )
    lines += _table(
        ["数据来源", "RMS 中位", "RMS P90", "共同响应 r", "检索条件", "Top-1", "Top-10"], rows
    )
    lines.append(
        "Norman 只有一个 canonical 批次，其跨批统计为 NA。批次关联需与置换基线一起读；检索使用观测拆半效应，不是模型推理。"
    )
    lines.append("")
    lines += ["## C. Systema 启发的参照敏感性", ""]
    rows = []
    for dataset, label, line in DATASETS:
        v = _line(data, "systema", dataset, line)
        rows.append(
            [
                label,
                v["train_conditions"],
                v["test_conditions"],
                _fmt(v["systematic_variation_mean_cosine"]),
                _fmt(v["train_perturbed_mean_baseline_control_pearson_delta_median"]),
                _fmt(v["train_perturbed_mean_baseline_systema_pearson_delta_median"]),
                _fmt(v["oracle_split_half_control_pearson_delta_median"]),
                _fmt(v["oracle_split_half_train_perturbed_reference_pearson_delta_median"]),
            ]
        )
    lines += _table(
        [
            "数据集",
            "训练条件",
            "测试条件",
            "共同余弦",
            "常数基线 control r",
            "常数基线扰动中心 r",
            "拆半 control r",
            "拆半扰动中心 r",
        ],
        rows,
    )
    rows = []
    for dataset, label, line in DATASETS:
        v = _line(data, "systema", dataset, line)
        accuracy = v["perturbed_mean_baseline_centroid_accuracy"]
        bins = v["test_count_bins"]
        rows.append(
            [
                label,
                f"{_fmt(accuracy['mean'])}/{_fmt(accuracy['p10'])}/{_fmt(accuracy['p90'])}",
                *[
                    f"{d['conditions']}/{_fmt(d['split_half_pearson_delta_median'])}"
                    for d in (bins[k] for k in ("0-20", "20-50", "50-100", "100-plus"))
                ],
            ]
        )
    lines += _table(
        [
            "数据集",
            "常数基线 centroid 均值/P10/P90",
            "0-19 n/r",
            "20-49 n/r",
            "50-99 n/r",
            "≥100 n/r",
        ],
        rows,
    )
    lines.append(
        "cell-count 分层是测试条件的观察性分组；常数基线的 centroid accuracy 均值约 0.5 是排序恒等性质，不能理解为预测成功率。"
    )
    lines.append("")
    lines += ["## D. 条件效应图谱、guide 与 GO 程序", ""]
    rows = []
    for dataset, label, line in CONTEXTS:
        v = _line(data, "atlas", dataset, line)
        rows.append(
            [
                label,
                v["conditions"],
                v["conditions_below_20_cells"],
                _fmt(v["effect_rms_median"]),
                _fmt(v["specific_residual_rms_median"]),
                _fmt(v["common_alignment_cosine_mean"]),
                _fmt(v["effective_rank_of_condition_effects_max500"]),
                _fmt(v["split_half_top20_jaccard_median"]),
            ]
        )
    lines += _table(
        [
            "数据来源",
            "条件",
            "<20 细胞",
            "效应 RMS",
            "特异残差 RMS",
            "共同余弦",
            "有效秩",
            "Top20 Jaccard",
        ],
        rows,
    )
    rows = []
    for dataset, label, line in DATASETS:
        guide = _line(data, "atlas", dataset, line).get("guide_agreement")
        if guide:
            rows.append(
                [
                    label,
                    guide["guide_column"],
                    guide["eligible_conditions_with_two_guides"],
                    guide["guide_pairs"],
                    _fmt(guide["guide_pair_pearson_delta_median"]),
                ]
            )
    lines += _table(["数据集", "guide 列", "≥2 guide 条件", "guide 对", "效应相关中位"], rows)
    lines.append(
        "GO 表每格依次为 *覆盖基因数／每条件绝对变化中位数／共同变化*；变化来自观测均值，NA 表示覆盖太低或统计未定义，不是 GO 富集显著性。"
    )
    lines.append("")
    for group in (PROGRAMS[:4], PROGRAMS[4:]):
        rows = []
        for dataset, label, line in CONTEXTS:
            programs = _line(data, "atlas", dataset, line)["programs"]
            values = []
            for key, _ in group:
                program = programs[key]
                values.append(
                    f"{program['covered_genes']}/{_fmt(program['median_abs_condition_mean_shift'])}/{_fmt(program['common_mean_shift'])}"
                )
            rows.append([label, *values])
        lines += _table(["数据来源", *[label for _, label in group]], rows)
    lines += ["## E. 跨细胞效应与 Norman 双基因结构", ""]
    cross = data["atlas"]["crosscell"]["four_line_transfer"]
    rows = []
    for key, value in cross.items():
        rows.append(
            [
                key,
                value["shared_conditions"],
                _fmt(value["median_pearson_delta"]),
                _fmt(value["median_target_over_source_effect_norm"]),
                _fmt(value["median_top100_target_gene_sign_agreement"]),
            ]
        )
    lines += _table(
        ["目标 <- 来源", "共同条件", "效应 r 中位", "目标/来源范数中位", "Top100 符号一致"], rows
    )
    rows = []
    for label, value in data["landscape"]["crosscell"]["crosscell_transfer"].items():
        rows.append(
            [
                label,
                value["source_average_shared_conditions"],
                _fmt(value["source_average_pearson_delta_median"]),
            ]
        )
    lines += _table(["留出细胞系", "源均值可比条件", "观测效应与源均值 r 中位"], rows)
    norman = _line(data, "atlas", "norman", "K562")["double_interactions"]
    lines.append(
        f"Norman：拟合双扰动 **{norman['fitted_double_conditions']}** 个；单扰动系数中位数 **{_fmt(norman['coefficient_a_median'])}/{_fmt(norman['coefficient_b_median'])}**；单位可加残差 **{_fmt(norman['unit_additive_relative_residual_median'])}**；拟合残差 **{_fmt(norman['fitted_relative_residual_median'])}**；残差拆半相关 **{_fmt(norman['fitted_residual_split_half_pearson_median'])}**（{norman['fitted_residual_split_half_conditions']} 条件）。"
    )
    lines.append("")
    lines += ["## F. 同批次细胞分布检验与样本量曲线", ""]
    rows = []
    for dataset, label, line in CONTEXTS:
        v = _line(data, "distribution", dataset, line)
        rows.append(
            [
                label,
                f"{v['eligible_conditions']}/{v['all_noncontrol_conditions']}",
                v["analyzed_conditions"],
                _fmt(v["median_energy_rp96"]),
                _fmt(v["fraction_bh_q_below_0_1"]),
                _fmt(v["median_log2_variance_trace_ratio"]),
            ]
        )
    lines += _table(
        [
            "数据来源",
            "合格/全部条件",
            "分析条件",
            "RP96 energy 中位",
            "样本内 q<0.1 比例",
            "log2 方差迹比中位",
        ],
        rows,
    )
    rows = []
    for dataset, label, line in CONTEXTS:
        curve = _line(data, "distribution", dataset, line)["equal_cell_split_half_curve"]
        rows.append(
            [
                label,
                *[
                    f"{curve[str(n)]['comparisons']}/{_fmt(curve[str(n)]['median_pearson_delta_rp96'])}"
                    if str(n) in curve
                    else "NA"
                    for n in (10, 20, 40)
                ],
            ]
        )
    lines += _table(["数据来源", "每半 10：比较/r", "每半 20：比较/r", "每半 40：比较/r"], rows)
    lines.append(
        "同批次合格条件极少时，q 比例只是被选子集的统计，不能推广全数据集。随机投影每细胞系独立，energy 大小不得跨系比较。"
    )
    lines.append("")
    lines += ["## G. 五数据集 PCA/UMAP 与拆半原图", ""]
    lines.append(
        "每个数据集单列一页。PCA/UMAP 依次按扰动类别、冻结划分和批次上色；拆半图给出测试条件 Pearson Δ 分布及与条件细胞数的关系。不同嵌入分别拟合，坐标不能跨数据集对齐。"
    )
    lines.append("")
    for i, (dataset, label, _) in enumerate(DATASETS, 1):
        lines += ["<!-- pagebreak -->", "", f"## 图 A{2 * i - 1}-A{2 * i}｜{label}", ""]
        lines += [
            f"![{label} PCA 和 UMAP](figures/{dataset}-embedding.jpg)",
            "",
            f"图 A{2 * i - 1}｜{label}：PCA/UMAP 按 control／单／双、条件划分、批次上色。",
            "",
            f"![{label} 测试条件拆半](figures/{dataset}-split_half.jpg)",
            "",
            f"图 A{2 * i}｜{label}：测试条件拆半 Pearson Δ 的分布与每条件细胞数。",
            "",
        ]
    output.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.data, args.output)


if __name__ == "__main__":
    main()
