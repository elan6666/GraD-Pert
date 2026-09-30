# ruff: noqa: RUF001
"""Render server-materialized sampling statistics; performs no scientific analysis."""

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--font", type=Path, default=Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
)
args = parser.parse_args()
sources = json.loads((args.results / "sources.json").read_text())
complete = json.loads((args.results / "COMPLETE.json").read_text())
if complete["status"] != "complete" or set(sources) != {"cap40", "proportion50", "proportion25"}:
    raise ValueError("Complete compatible comparison required")
source = sources["proportion50"]["source_commit"]
if source != sources["proportion25"]["source_commit"] or source != complete["source_commit"]:
    raise ValueError("Source mismatch")
summary = {k: v["summary"] for k, v in sources.items()}
original = summary["cap40"]["original"]
args.output.parent.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont("CJK", str(args.font)))
c = canvas.Canvas(str(args.output), pagesize=A4)
c.setTitle("Jurkat 三种下采样：数据量与重复性的权衡")
c.setAuthor("GraD-Pert project")
W, H = A4
M, gap = 36, 24
cw = (W - 2 * M - gap) / 2
rx = M + cw + gap
style = ParagraphStyle(
    "body",
    fontName="CJK",
    fontSize=9,
    leading=14,
    wordWrap="CJK",
    textColor=colors.HexColor("#263647"),
)
small = ParagraphStyle("small", parent=style, fontSize=7.5, leading=11)
heading = ParagraphStyle(
    "heading", parent=style, fontSize=12, leading=17, textColor=colors.HexColor("#235f9a")
)
title = ParagraphStyle("title", parent=heading, fontSize=17, leading=22)


def para(text, x, y, width=cw, sty=style):
    p = Paragraph(text, sty)
    _, height = p.wrap(width, H)
    if y - height < 48:
        raise RuntimeError("text overflow")
    p.drawOn(c, x, y - height)
    return y - height - 8


def table(rows, x, y, widths):
    t = Table(rows, colWidths=widths)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "CJK"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f6")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ccd6de")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    _, height = t.wrap(W, H)
    if y - height < 48:
        raise RuntimeError("table overflow")
    t.drawOn(c, x, y - height)
    return y - height - 10


def footer(page):
    c.setStrokeColor(colors.HexColor("#d6dfe7"))
    c.line(M, 35, W - M, 35)
    c.setFont("CJK", 7)
    c.setFillColor(colors.HexColor("#667887"))
    c.drawString(M, 23, "GraD-Pert | Jurkat 数据分析 | 2026-09-30 | 非模型评估")
    c.drawRightString(W - M, 23, f"{page} / 2")


methods = [("cap40", "最多40"), ("proportion50", "50%"), ("proportion25", "25%")]
y = para("Jurkat：三种下采样与原数据的重复性比较", M, H - 37, W - 2 * M, title)
y = para(
    "仅采训练扰动细胞；每种固定样本重复100次分层拆半，统一control与5000个基因。", M, y, W - 2 * M
)
rows = [
    ["版本", "训练细胞", "减少比例", "Pearson Δ", "表达Pearson"],
    [
        "原始",
        "128,266",
        "0%",
        f"{original['mean']:.4f}",
        f"{summary['cap40']['raw_expression']['original']['mean']:.4f}",
    ],
]
for method, label in methods:
    s = summary[method]
    rows.append(
        [
            label,
            f"{s['sampled_train_cells']:,}",
            f"{s['train_cells_removed_fraction']:.1%}",
            f"{s[method]['mean']:.4f}",
            f"{s['raw_expression']['cap']['mean']:.4f}",
        ]
    )
y = table(rows, M, y, [95, 110, 100, 105, W - 2 * M - 410])
image_height = (W - 2 * M) * 8 / 12
c.bookmarkHorizontalAbsolute("fig1", y)
c.drawImage(
    str(args.results / "comparison.png"),
    M,
    y - image_height,
    width=W - 2 * M,
    height=image_height,
    mask="auto",
)
y -= image_height + 6
y = para(
    "图1 | 分布、训练细胞量—重复性、50%对cap40逐条件对照、原细胞数分层。全部图数据见第2页链接。",
    M,
    y,
    W - 2 * M,
    small,
)
left = right = y - 2
left = para("主要观察", M, left, sty=heading)
left = para(
    "50%保留更多细胞，但总体Pearson Δ与cap40接近；"
    "比例采样减少了小条件的细胞，而cap40完整保留原本≤40个细胞的条件。"
    "大条件中50%更有优势，详见分层表。",
    M,
    left,
)
right = para("解释边界", rx, right, sty=heading)
right = para(
    "25%重复性下降更明显。此结果衡量观测数据稳定性，不是模型分数或理论预测上限；单次seed42采样也不能证明哪种方案训练效果更好。未启动模型训练或切换数据默认。",
    rx,
    right,
)
footer(1)
c.showPage()
y = para("统计定义、完整产物与版本验收", M, H - 37, W - 2 * M, title)
left = right = y - 10
left = para("统一采样与Pearson定义", M, left, sty=heading)
left = para(
    (
        "对条件p的n个训练细胞，50%或25%保留 m=min(n,max(min(n,"
        "2),floor(fn+0.5)))。batch名额按原比例用最大余数法分配，再"
        "无放回随机抽样。单细胞保留1；原有至少2个细胞的条件至少保留2。"
    ),
    M,
    left,
)
left = para(
    "所有1335个训练条件、12013个control、41235个验证行、54658个测试行、2805个排除行保持；保留6506变量及原表达。统计使用冻结5000表达基因轴。",
    M,
    left,
)
left = para(
    (
        "每个batch及条件整体两组细胞数相差≤1，随机处理奇数余行。减去原条件batc"
        "h上下文中的同一个完整control池均值c[p]，在基因轴计算：<br/>r["
        "p,k]=corr(μ[A,k]−c[p], μ[B,k]−c[p])<br/>"
        "先平均100次拆半，再对有效条件等权平均。"
    ),
    M,
    left,
)
left = para(
    "1331个有效条件。BIRC5、RACGAP1、ECT2、SEC13各仅1细胞，无法拆半，记缺失而非0。这是同一固定样本的100次拆半；未生成100份采样数据。",
    M,
    left,
)
left = para("原条件细胞数分层", M, left, sty=heading)
strata = ["1-40", "41-80", "81-160", ">160"]
rows = [["原细胞数", "原始", "40", "50%", "25%"]]
for st in strata:
    row = [st, f"{summary['cap40']['cell_count_strata'][st]['original']['mean']:.3f}"]
    row += [f"{summary[k]['cell_count_strata'][st]['cap']['mean']:.3f}" for k, _ in methods]
    rows.append(row)
left = table(rows, M, left, [cw - 4 * 42, 42, 42, 42, 42])
left = para(
    (
        "共有866/1309/1319个条件分别在cap40/50%/25%中失去至少一"
        "个原batch覆盖。按比例分配不能保证稀有batch都得到1行。所有版本原始参考"
        "与分组身份已核对一致。"
    ),
    M,
    left,
)
right = para("产物与验收", rx, right, sty=heading)
right = para(
    (
        "两种新分析均exit0/COMPLETE，CPU-only、4线程，耗时分别89"
        ".42/82.55秒。全部输出哈希、行顺序、obs/var、配额及各267000"
        "条逐次记录验收通过；所有保留矩阵值精确不变，原始数据/manifest哈希不变。"
        "12项本地及服务器检查通过。"
    ),
    rx,
    right,
)
right = para(
    (
        "每个采样run：<br/>• proportion50.h5ad / propo"
        "rtion25.h5ad：衍生H5AD，仍为analysis-only。<br/"
        ">• selection.json：固定采样行ID与条件清单。<br/>• re"
        "peat_scores.csv：条件×版本×100次记录。<br/>• cond"
        "itions.json、summary.json：逐条件与汇总。<br/>• c"
        "omparison.png/pdf、receipt、COMPLETE、exit："
        "图和身份收据。前三类只留服务器。"
    ),
    rx,
    right,
)
url = (
    "https://github.com/elan6666/GraD-Pert/blob/main/docs/experiments/data/"
    + args.results.name
    + "/"
)
right = para(
    '四版本比较：<br/>• <link color="#235f9a" href="'
    + url
    + (
        'conditions.json">conditions.json：图1全部逐条件'
        "数据</link>。<br/>• sources.json：三个来源和原统计。<"
        "br/>• comparison.png/pdf、COMPLETE。<br/>•"
        " artifact-index.json：全部图/数据与大小/hash。<br/"
        ">• small-transfer及queue plan/publication"
        "/exit/acceptance/tests：安全传输及独立验收。"
    ),
    rx,
    right,
)
right = para(
    '新分析/比较源码：<link color="#235f9a" href="https://github.com/elan6666/GraD-Pert/commit/'
    + source
    + '">'
    + source
    + (
        '</link><br/>cap40历史源码：<link color="#235f'
        '9a" href="https://github.com/elan6666/Gr'
        "aD-Pert/commit/"
    )
    + sources["cap40"]["source_commit"]
    + '">'
    + sources["cap40"]["source_commit"]
    + "</link>。各配置、H5AD、control/基因/划分hash均在索引所链接的收据中。",
    rx,
    right,
    sty=small,
)
right = para(
    (
        '借鉴<link color="#235f9a" href="https://gi'
        "thub.com/valence-labs/TxPert/tree/08d82e"
        "ea86746b044cf7531f4ec8c5f60e1cb73f#exper"
        'imental-reproducibility">TxPert真实子集重复性基线'
        "</link>；分层100次拆半是项目适配，不声称原样复现。共同control参"
        "考噪声和单次采样限制仍存在；总表达高相关不能代替扰动效应稳定性。"
    ),
    rx,
    right,
)
right = para(
    (
        '<link color="#235f9a" href="#fig1">返回图1<'
        '/link> | <link color="#235f9a" href="htt'
        "ps://github.com/elan6666/GraD-Pert/blob/"
        "main/docs/experiments/JURKAT_SAMPLING_CO"
        'MPARISON_20260930.md">完整中文结果与所有产物链接</lin'
        "k>"
    ),
    rx,
    right,
    sty=small,
)
footer(2)
c.save()
print(args.output)
