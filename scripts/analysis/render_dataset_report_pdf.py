"""Render the Chinese observed-data report as a paginated PDF with figures."""

from __future__ import annotations

import argparse
import html
import os
import re
from pathlib import Path

from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    TableStyle,
)

DEFAULT_REPORT = (
    Path(__file__).resolve().parents[2] / "docs/experiments/PERTURBATION_DATASET_ANALYSIS_ZH.md"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "output/pdf/gradpert_dataset_analysis_zh.pdf"
DEFAULT_APPENDIX = (
    Path(__file__).resolve().parents[2]
    / "docs/experiments/PERTURBATION_DATASET_ANALYSIS_APPENDIX_ZH.md"
)
DEFAULT_DATA_ATTACHMENT = (
    Path(__file__).resolve().parents[2]
    / "docs/experiments/data/report-aggregate-current-ecdf8f1.json"
)
FONT = "Songti-Embedded"
FONT_FILE = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
INK = colors.HexColor("#193047")
BLUE = colors.HexColor("#1A607C")
PALE = colors.HexColor("#EAF3F7")
GRID = colors.HexColor("#D7E3E8")


def _inline(source: str) -> str:
    source = html.escape(source)
    source = source.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    source = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", source)
    source = re.sub(r"`([^`]+)`", r'<font color="#326C82">\1</font>', source)

    def link(match: re.Match[str]) -> str:
        label, target = match.groups()
        if target.startswith(("https://", "http://", "#")):
            return f'<link href="{html.escape(target, quote=True)}" color="#1A607C">{label}</link>'
        if target.endswith(".md") and "/" not in target:
            source_url = (
                f"https://github.com/elan6666/GraD-Pert/blob/main/docs/experiments/{target}"
            )
            return f'<link href="{source_url}" color="#1A607C">{label}</link>'
        return label

    return re.sub(r"\[([^]]+)\]\(([^)]+)\)", link, source)


def _styles() -> dict[str, ParagraphStyle]:
    if not FONT_FILE.exists():
        raise FileNotFoundError(f"required Chinese font is unavailable: {FONT_FILE}")
    pdfmetrics.registerFont(TTFont(FONT, str(FONT_FILE), subfontIndex=0))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT, italic=FONT, boldItalic=FONT)
    base = getSampleStyleSheet()
    common = dict(fontName=FONT, textColor=INK, wordWrap="CJK", allowWidows=0, allowOrphans=0)
    return {
        "title": ParagraphStyle(
            "report_title",
            parent=base["Title"],
            fontSize=19,
            leading=27,
            textColor=BLUE,
            spaceAfter=12,
            alignment=TA_LEFT,
            **{k: v for k, v in common.items() if k != "textColor"},
        ),
        "section": ParagraphStyle(
            "report_section",
            parent=base["Heading2"],
            fontSize=12.5,
            leading=18,
            textColor=BLUE,
            spaceBefore=17,
            spaceAfter=7,
            **{k: v for k, v in common.items() if k != "textColor"},
        ),
        "body": ParagraphStyle(
            "report_body",
            parent=base["BodyText"],
            fontSize=8.9,
            leading=14.1,
            spaceAfter=7,
            **common,
        ),
        "small": ParagraphStyle(
            "report_small",
            parent=base["BodyText"],
            fontSize=8.0,
            leading=12.5,
            spaceAfter=7,
            **common,
        ),
        "caption": ParagraphStyle(
            "report_caption",
            parent=base["BodyText"],
            fontSize=8.4,
            leading=12.4,
            textColor=colors.HexColor("#536675"),
            spaceBefore=5,
            spaceAfter=12,
            **{k: v for k, v in common.items() if k != "textColor"},
        ),
        "table": ParagraphStyle(
            "report_table",
            parent=base["BodyText"],
            fontSize=7.9,
            leading=10.9,
            alignment=TA_CENTER,
            **common,
        ),
    }


def _table(lines: list[str], styles: dict[str, ParagraphStyle], width: float) -> LongTable:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [rows[0], *rows[2:]]
    count = len(rows[0])
    if count == 6:
        weights = [1.18, 0.82, 1.05, 1.10, 1.50, 1.05]
    elif count == 3:
        weights = [1.20, 1.15, 2.55]
    else:
        weights = [1.0] * count
    unit = width / sum(weights)
    table_style = styles["table"]
    if count >= 7:
        table_style = ParagraphStyle(
            "report_table_compact", parent=styles["table"], fontSize=7.0, leading=9.6
        )
    cells = [[Paragraph(_inline(cell), table_style) for cell in row] for row in rows]
    table = LongTable(
        cells, colWidths=[unit * item for item in weights], repeatRows=1, hAlign="LEFT"
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FBFC")]),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _footer(canvas: object, doc: BaseDocTemplate) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.line(42, 38, A4[0] - 42, 38)
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#617585"))
    canvas.drawString(42, 25, "GraD-Pert | 数据集观察性分析与方法综述 | 2026-09-23")
    canvas.drawRightString(A4[0] - 42, 25, str(doc.page))
    canvas.restoreState()


def render(
    source: Path,
    output: Path,
    appendix: Path = DEFAULT_APPENDIX,
    data_attachment: Path | None = None,
) -> None:
    styles = _styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    width = A4[0] - 84
    gutter = 17
    column_width = (width - gutter) / 2
    doc = BaseDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=47,
        bottomMargin=52,
        title="GraD-Pert 扰动数据集观察性分析与方法综述",
        author="GraD-Pert research project",
    )

    def frame(x: float, y: float, w: float, h: float) -> Frame:
        return Frame(x, y, w, h, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    main_height = A4[1] - 99
    header_height = 155
    doc.addPageTemplates(
        [
            PageTemplate(
                id="opening",
                frames=[
                    frame(42, 52 + main_height - header_height, width, header_height),
                    frame(42, 52, column_width, main_height - header_height - 8),
                    frame(
                        42 + column_width + gutter,
                        52,
                        column_width,
                        main_height - header_height - 8,
                    ),
                ],
                onPage=_footer,
                autoNextPageTemplate="columns",
            ),
            PageTemplate(
                id="columns",
                frames=[
                    frame(42, 52, column_width, main_height),
                    frame(42 + column_width + gutter, 52, column_width, main_height),
                ],
                onPage=_footer,
            ),
            PageTemplate(
                id="full",
                frames=[frame(42, 52, width, main_height)],
                onPage=_footer,
            ),
        ]
    )
    lines = source.read_text().splitlines()
    main_count = len(lines)
    lines += ["", "<!-- pagebreak -->", "", *appendix.read_text().splitlines()]
    story: list[object] = []
    plates: list[object] = []
    main_table_number = 0
    index = 0
    opening_done = False
    while index < len(lines):
        line = lines[index].strip()
        in_appendix = index >= main_count + 3
        if not line:
            index += 1
            continue
        if line == "<!-- pagebreak -->":
            if index == main_count + 1:
                story.append(NextPageTemplate("full"))
                story.append(PageBreak())
                story.append(Paragraph("图版与正文主表", styles["section"]))
                story.extend(plates)
                story.append(PageBreak())
            else:
                story.append(PageBreak())
        elif line.startswith("# "):
            title = _inline(line[2:])
            if line.startswith("# 数据附录"):
                title = '<a name="appendix"/>' + title
            story.append(Paragraph(title, styles["title"]))
        elif line.startswith("## "):
            title = _inline(line[3:])
            if line.startswith("## 数据附录"):
                title = '<a name="appendix"/>' + title
            story.append(Paragraph(title, styles["section"]))
        elif line.startswith("| "):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            if not in_appendix:
                main_table_number += 1
                plates.extend(
                    [
                        Paragraph(
                            f'<a name="table{main_table_number}"/>'
                            f"表 {main_table_number} · 正文数据汇总",
                            styles["section"],
                        ),
                        _table(table_lines, styles, width),
                        Spacer(1, 12),
                    ]
                )
            else:
                story.extend([_table(table_lines, styles, width), Spacer(1, 9)])
            continue
        elif line.startswith("!["):
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if match is None:
                raise ValueError(f"invalid image: {line}")
            path = source.parent / match.group(2)
            figure_width = width * (0.77 if "crosscell-transfer" in path.name else 1.0)
            with PILImage.open(path) as picture:
                figure_height = figure_width * picture.height / picture.width
            figure = Image(str(path), width=figure_width, height=figure_height, hAlign="CENTER")
            if index + 2 < len(lines) and lines[index + 2].startswith("图 "):
                caption_text = lines[index + 2]
                number = re.match(r"图 (A?\d+)", caption_text)
                anchor = f'<a name="fig{number.group(1).lower()}"/>' if number else ""
                caption = Paragraph(anchor + _inline(caption_text), styles["caption"])
                target = story if in_appendix else plates
                if not in_appendix:
                    target.append(PageBreak())
                target.append(KeepTogether([figure, caption]))
                index += 2
            else:
                target = story if in_appendix else plates
                target.append(figure)
        elif line.startswith("- "):
            story.append(Paragraph("- " + _inline(line[2:]), styles["small"]))
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph(_inline(line), styles["small"]))
        else:
            paragraph = [line]
            while (
                index + 1 < len(lines)
                and lines[index + 1].strip()
                and not lines[index + 1].startswith(("#", "|", "![", "- "))
            ):
                index += 1
                paragraph.append(lines[index].strip())
            style = styles["small"] if line.startswith(("¹", "²")) else styles["body"]
            story.append(Paragraph(_inline(" ".join(paragraph)), style))
            if not opening_done:
                story.append(FrameBreak())
                opening_done = True
        index += 1
    doc.build(story)
    if data_attachment is not None:
        writer = PdfWriter()
        writer.clone_document_from_reader(PdfReader(output))
        writer.add_attachment(filename=data_attachment.name, data=data_attachment.read_bytes())
        staged = output.with_suffix(".attached.pdf")
        with staged.open("wb") as handle:
            writer.write(handle)
        os.replace(staged, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--appendix", type=Path, default=DEFAULT_APPENDIX)
    parser.add_argument("--data-attachment", type=Path, default=DEFAULT_DATA_ATTACHMENT)
    args = parser.parse_args()
    render(args.source, args.output, args.appendix, args.data_attachment)


if __name__ == "__main__":
    main()
