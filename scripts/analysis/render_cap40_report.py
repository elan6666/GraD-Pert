# ruff: noqa: RUF001
# Chinese report text intentionally uses Chinese punctuation.
"Render the frozen 2026-09-30 cap40 report; no data analysis or training."

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

parser = argparse.ArgumentParser(
    description=("Render the small, server-materialized cap40 results as a Chinese PDF.")
)
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--font", type=Path, default=Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
)
args = parser.parse_args()
root = args.results
s = json.loads((root / "summary.json").read_text())
a = json.loads((root / ("jurkat-cap40-5a8a7bb-20260930T094437Z-acceptance.json")).read_text())
if s["source_commit"] != "5a8a7bb6712bc4facdfeb13d8dd629bd0f250880":
    raise ValueError("This renderer is for the frozen 2026-09-30 result only")
out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont("CJK", str(args.font)))
W, H = A4
c = canvas.Canvas(str(out), pagesize=A4)
c.setTitle("Jurkat cap40 下采样与拆半重复性分析")
c.setAuthor("GraD-Pert project")
M = 36
gap = 24
cw = (W - 2 * M - gap) / 2
style = ParagraphStyle(
    "body",
    fontName="CJK",
    fontSize=9,
    leading=14,
    wordWrap="CJK",
    spaceAfter=6,
    textColor=colors.HexColor("#263647"),
)
small = ParagraphStyle("small", parent=style, fontSize=7.5, leading=11)
heading = ParagraphStyle(
    "head", parent=style, fontSize=12, leading=17, textColor=colors.HexColor("#235f9a")
)


def para(text, x, y, w=cw, sty=style):
    p = Paragraph(text, sty)
    _pw, ph = p.wrap(w, H)
    p.drawOn(c, x, y - ph)
    if y - ph < 48:
        raise RuntimeError("text overflow")
    return y - ph - 8


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
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    _, th = t.wrap(W, H)
    t.drawOn(c, x, y - th)
    if y - th < 48:
        raise RuntimeError("table overflow")
    return y - th - 10


def footer(page):
    c.setStrokeColor(colors.HexColor("#d6dfe7"))
    c.line(M, 35, W - M, 35)
    c.setFont("CJK", 7)
    c.setFillColor(colors.HexColor("#667887"))
    c.drawString(M, 23, "GraD-Pert | 数据集分析 | 2026-09-30 | 非模型评估")
    c.drawRightString(W - M, 23, f"{page} / 2")


y = H - 37
c.bookmarkPage("result")
y = para(
    "Jurkat cap40：数据量与扰动效应重复性的权衡",
    M,
    y,
    W - 2 * M,
    ParagraphStyle("title", parent=heading, fontSize=17, leading=22),
)
y = para("固定训练样本，100次分层拆半；control与验证/测试保持原样。", M, y, W - 2 * M)
y = table(
    [
        ["主要结果", "原始", "cap40"],
        ["训练扰动细胞", "128,266", "47,836 (-62.7%)"],
        ["拆半 Pearson Δ 条件均值", "0.3739", "0.2618"],
        ["有效条件 / 全部条件", "1331 / 1335", "1331 / 1335"],
    ],
    M,
    y,
    [W - 2 * M - 196, 98, 98],
)
image_h = (W - 2 * M) * 720 / 2520
c.bookmarkHorizontalAbsolute("fig1", y)
c.drawImage(
    str(root / "comparison.png"), M, y - image_h, width=W - 2 * M, height=image_h, mask="auto"
)
y -= image_h + 7
y = para(
    "图1 | 左：条件均值分布；中：受影响条件原/采样对照；右：重复性差值与原细胞数。",
    M,
    y,
    W - 2 * M,
    small,
)
left = right = y - 5
left = para("采样和计算", M, left, sty=heading)
left = para(
    (
        "只采样训练扰动细胞，每条件保留 min(40,Np)，种子42；按ba"
        "tch比例配额，无放回抽样。保留全部1335条件和6506个变量，统计"
        "使用原5000个表达基因。"
    ),
    M,
    left,
)
left = para(
    (
        "每个条件拆成互不重叠的两组，各batch及整体两侧行数相差不超过1。两组先计算表达均值，再"
        "减去同一原始上下文control池均值，最后在基因轴算Pearson Δ。重复100次后，"
        "先条件内平均，再对条件等权平均。"
    ),
    M,
    left,
)
left = para(
    (
        "r[p,k] = corr(μ[A] - c[p], μ[B] - c[p])<br/>r"
        "[p] = mean(k=1..100) r[p,k]<br/>总体结果 = 对有效条件 "
        "r[p] 等权平均。"
    ),
    M,
    left,
    sty=small,
)
left = para(
    (
        "317个条件原细胞数≤40，结果完全不变；1018个条件下采样。四个单"
        "细胞条件无法拆半，记缺失而非0。100次是同一固定样本的重复拆半，并非"
        "100份下采样数据。"
    ),
    M,
    left,
)
rx = M + cw + gap
right = para("结果与解释", rx, right, sty=heading)
right = table(
    [
        ["原细胞数", "条件", "原始", "cap40"],
        ["1-40", "317", "0.2687", "0.2687"],
        ["41-80", "411", "0.3704", "0.3022"],
        ["81-160", "452", "0.4028", "0.2384"],
        [">160", "155", "0.5117", "0.2089"],
    ],
    rx,
    right,
    [68, 41, 58, 58],
)
right = para(
    (
        "原/采样配对差均值 -0.1122；条件bootstrap 95%描述"
        "区间 [-0.1188,-0.1058]。这是重复性统计，不是理论预测"
        "上限，也不是生物重复置信区间。"
    ),
    rx,
    right,
)
right = para(
    (
        "保留40行 vs 完整原均值为0.7991，但两者重叠；保留行 vs "
        "删除行仅0.3736。这两种样本量也不同，不能混为同一估计量。"
    ),
    rx,
    right,
)
right = para(
    (
        "总表达相关性仍达0.9630，不能代替扰动变化稳定性。866个条件丢失"
        "至少一个原batch的细胞覆盖。cap40减少训练数据，也同时改变ba"
        "tch覆盖及row_mean条件权重。"
    ),
    rx,
    right,
)
footer(1)
c.showPage()

y = H - 37
y = para(
    "复现、完整产物清单与验收",
    M,
    y,
    W - 2 * M,
    ParagraphStyle("title2", parent=heading, fontSize=17, leading=22),
)
y = para("没有启动训练，没有替换canonical数据，没有切换B0默认数据。", M, y, W - 2 * M)
left = right = y - 10
left = para("保留数据与验收", M, left, sty=heading)
left = table(
    [
        ["分区", "原始", "采样后"],
        ["训练扰动", "128,266", "47,836"],
        ["control", "12,013", "12,013"],
        ["验证扰动", "41,235", "41,235"],
        ["测试扰动", "54,658", "54,658"],
        ["其他排除行", "2,805", "2,805"],
        ["总行数", "238,977", "158,547"],
    ],
    M,
    left,
    [79, 73, 73],
)
left = para(
    (
        "衍生H5AD保留所有基因与保留行的原始表达，逐值及dtype精确一致，obs/var分类字"
        "典严格一致。原H5AD和四份冻结manifest哈希前后不变。主会话独立核对行集合、顺序、"
        "分区、每条件数量与267,000条统计。exit0、COMPLETE、zero-PKL均通"
        "过。"
    ),
    M,
    left,
)
left = para(
    (
        "成功运行耗时90.85秒；仅CPU，数值库4线程。训练行数减少62.7"
        "%，但验证与终态测试未减少，因此不能直接推断总训练时长。尚无cap40"
        "训练性能或模型效果证据。"
    ),
    M,
    left,
)
left = para(
    (
        "首次基因身份核对、第二次类别字典检查的失败目录均保留。未放宽校验、未覆"
        "盖旧run。最后本地与服务器定向测试均8项通过。"
    ),
    M,
    left,
)
right = para("所有产物及图数据", rx, right, sty=heading)
right = para(
    (
        "服务器保留：<br/>• cap40.h5ad：158547×6506，4.143GB。<"
        "br/>• selection.json：固定行ID、条件与哈希，2.72MB。<br/>"
        "• repeat_scores.csv：267000条逐次记录，16.84MB。"
    ),
    rx,
    right,
)
url = (
    "https://github.com/elan6666/GraD-Pert/blob/ma"
    "in/docs/experiments/data/jurkat-cap40-5a8a7bb"
    "-20260930T094437Z/"
)
right = para(
    '小型产物：<br/>• <link color="#235f9a" href="'
    + url
    + (
        'conditions.json">conditions.json：图1的全部作图数据</l'
        "ink>。<br/>• summary.json：汇总与分层统计。<br/>• compa"
        "rison.png / comparison.pdf：同一图。<br/>• artifac"
        "t-index.json：全部文件大小与SHA256。<br/>• receipt / C"
        "OMPLETE / launch / exit / publication / tests"
        " / acceptance：版本及终态证据。"
    ),
    rx,
    right,
)
right = para(
    (
        "服务器根（所有大文件留在服务器）：<br/>/data/yilangliu/GraD-Pe"
        "rt/development/<br/>jurkat-cap40-5a8a7bb-2026"
        "0930T094437Z"
    ),
    rx,
    right,
    sty=small,
)
right = para(
    ('分析源码：<link color="#235f9a" href="https://github.com/elan6666/GraD-Pert/commit/')
    + s["source_commit"]
    + '">'
    + s["source_commit"]
    + "</link><br/>配置SHA256：<br/>"
    + a["configuration_sha256"]
    + "<br/>衍生H5AD SHA256：<br/>"
    + s["derived_h5ad"]["sha256"],
    rx,
    right,
    sty=small,
)
right = para("来源与边界", rx, right, sty=heading)
right = para(
    (
        '借鉴 <link color="#235f9a" href="https://github'
        ".com/valence-labs/TxPert/tree/08d82eea86746b0"
        "44cf7531f4ec8c5f60e1cb73f#experimental-reprod"
        'ucibility">TxPert实验重复性基线</link> 的真实子集与未使用细胞比较'
        "思想。本次分层100次拆半是项目适配，不是原样复现，也不是raw-count multin"
        "omial采样。共同control参考的噪声也可能影响相关性。"
    ),
    rx,
    right,
)
right = para(
    (
        '<link color="#235f9a" href="#fig1">返回图1</link'
        '> | <link color="#235f9a" href="https://githu'
        "b.com/elan6666/GraD-Pert/blob/main/docs/exper"
        'iments/JURKAT_CAP40_RESULT_20260930.md">完整中文结'
        "果、协议及产物索引</link>"
    ),
    rx,
    right,
    sty=small,
)
footer(2)
c.save()
print(out)
