#!/usr/bin/env python3
"""Render the canonical Markdown manuscript as a reviewable academic PDF."""

from __future__ import annotations

import html
import re
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".vendor"))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


SOURCE = ROOT / "manuscript.md"
OUTPUT = ROOT / "MedTAWS_GRPO_manuscript.pdf"
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_X = 18 * mm
MARGIN_TOP = 17 * mm
MARGIN_BOTTOM = 18 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X


def _register_fonts() -> tuple[str, str, str, str]:
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    regular = font_dir / "DejaVuSerif.ttf"
    bold = font_dir / "DejaVuSerif-Bold.ttf"
    italic = font_dir / "DejaVuSerif-Italic.ttf"
    mono = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    if all(path.exists() for path in (regular, bold, italic, mono)):
        pdfmetrics.registerFont(TTFont("PaperSerif", str(regular)))
        pdfmetrics.registerFont(TTFont("PaperSerif-Bold", str(bold)))
        pdfmetrics.registerFont(TTFont("PaperSerif-Italic", str(italic)))
        pdfmetrics.registerFont(TTFont("PaperMono", str(mono)))
        pdfmetrics.registerFontFamily(
            "PaperSerif",
            normal="PaperSerif",
            bold="PaperSerif-Bold",
            italic="PaperSerif-Italic",
        )
        return "PaperSerif", "PaperSerif-Bold", "PaperSerif-Italic", "PaperMono"
    return "Times-Roman", "Times-Bold", "Times-Italic", "Courier"


SERIF, SERIF_BOLD, SERIF_ITALIC, MONO = _register_fonts()


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PaperTitle",
            parent=base["Title"],
            fontName=SERIF_BOLD,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            spaceAfter=8,
        ),
        "author": ParagraphStyle(
            "Author",
            parent=base["Normal"],
            fontName=SERIF,
            fontSize=9.5,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#374151"),
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=SERIF_BOLD,
            fontSize=13.2,
            leading=16,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
            textColor=colors.HexColor("#111827"),
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=SERIF_BOLD,
            fontSize=10.8,
            leading=13,
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=True,
            textColor=colors.HexColor("#1f2937"),
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=SERIF,
            fontSize=9.05,
            leading=12.15,
            alignment=TA_JUSTIFY,
            firstLineIndent=4.5 * mm,
            spaceAfter=4.5,
            splitLongWords=True,
            textColor=colors.HexColor("#111827"),
        ),
        "abstract": ParagraphStyle(
            "AbstractBody",
            parent=base["BodyText"],
            fontName=SERIF,
            fontSize=8.75,
            leading=11.8,
            alignment=TA_JUSTIFY,
            leftIndent=7 * mm,
            rightIndent=7 * mm,
            spaceAfter=5,
        ),
        "keywords": ParagraphStyle(
            "Keywords",
            parent=base["BodyText"],
            fontName=SERIF,
            fontSize=8.6,
            leading=11,
            leftIndent=7 * mm,
            rightIndent=7 * mm,
            spaceAfter=6,
        ),
        "list": ParagraphStyle(
            "List",
            parent=base["BodyText"],
            fontName=SERIF,
            fontSize=9.0,
            leading=12,
            leftIndent=8 * mm,
            firstLineIndent=-4 * mm,
            spaceAfter=2.5,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName=SERIF,
            fontSize=7.8,
            leading=9.8,
            alignment=TA_LEFT,
            spaceBefore=2,
            spaceAfter=4,
            textColor=colors.HexColor("#374151"),
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName=SERIF,
            fontSize=6.9,
            leading=8.4,
            alignment=TA_LEFT,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=base["Code"],
            fontName=MONO,
            fontSize=7.2,
            leading=9.5,
            leftIndent=7 * mm,
            rightIndent=7 * mm,
            spaceBefore=2,
            spaceAfter=5,
            textColor=colors.HexColor("#1f2937"),
        ),
    }


STYLES = _styles()


def inline_markup(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", rf'<font name="{MONO}">\1</font>', escaped)
    return escaped


def paragraph(text: str, style: str = "body") -> Paragraph:
    return Paragraph(inline_markup(text), STYLES[style])


def table_flowable(rows: list[list[str]], caption: str | None):
    formatted = []
    for row_index, row in enumerate(rows):
        formatted.append(
            [
                Paragraph(
                    ("<b>" + inline_markup(cell) + "</b>") if row_index == 0 else inline_markup(cell),
                    STYLES["table"],
                )
                for cell in row
            ]
        )
    columns = len(rows[0])
    first_weight = 2.3 if columns >= 4 else 1.8
    other_weight = 1.0
    total = first_weight + (columns - 1) * other_weight
    widths = [CONTENT_WIDTH * first_weight / total] + [CONTENT_WIDTH * other_weight / total] * (columns - 1)
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#374151")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor("#374151")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    output = []
    if caption:
        output.append(paragraph(caption, "caption"))
    output.extend([table, Spacer(1, 4)])
    return output


def figure(path: Path, caption: str, width: float):
    image = Image(str(path), width=width, height=width * 0.43)
    image.hAlign = "CENTER"
    return [Spacer(1, 4), image, paragraph(caption, "caption")]


def parse_manuscript():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = [Spacer(1, 8), paragraph(lines[0][2:], "title")]
    for line in lines[1:4]:
        cleaned = line.replace("**", "").rstrip("  ")
        story.append(paragraph(cleaned, "author"))
    story.extend([Spacer(1, 5), HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#9ca3af")), Spacer(1, 5)])

    index = 4
    paragraph_lines: list[str] = []
    pending_caption: str | None = None
    in_abstract = False
    inserted_framework = False
    inserted_results = False

    def flush():
        nonlocal paragraph_lines
        if paragraph_lines:
            text = " ".join(item.strip() for item in paragraph_lines)
            style = "abstract" if in_abstract else "keywords" if text.startswith("**Keywords:**") else "body"
            story.append(paragraph(text, style))
            paragraph_lines = []

    while index < len(lines):
        line = lines[index]
        if line.startswith("## "):
            flush()
            heading = line[3:]
            in_abstract = heading == "Abstract"
            story.append(paragraph(heading, "h1"))
            if heading.startswith("2. Related Work") and not inserted_framework:
                story.extend(
                    figure(
                        ROOT / "figures" / "framework_overview.png",
                        "Figure 1. Overview of the two-level MedTAWS-GRPO framework. Discrete workflow search and continuous prompt-sensitive adaptation are evaluated separately before a frozen combined arm.",
                        CONTENT_WIDTH * 0.94,
                    )
                )
                inserted_framework = True
            if heading.startswith("6. Discussion") and not inserted_results:
                story.extend(
                    figure(
                        ROOT / "figures" / "development_results.png",
                        "Figure 2. Retrospective RC development evidence. Dots in the left panel are the two training seeds. The right panel changes data size and training length and is not a single-variable ablation.",
                        CONTENT_WIDTH * 0.90,
                    )
                )
                inserted_results = True
            index += 1
            continue
        if line.startswith("### "):
            flush()
            in_abstract = False
            story.append(paragraph(line[4:], "h2"))
            index += 1
            continue
        if line.startswith("**Table ") and line.endswith("**"):
            flush()
            pending_caption = line[2:-2]
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[index + 1]):
            flush()
            rows = [[cell.strip() for cell in line.strip("|").split("|")]]
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip("|").split("|")])
                index += 1
            story.extend(table_flowable(rows, pending_caption))
            pending_caption = None
            continue
        if line == r"\[":
            flush()
            equation = []
            index += 1
            while index < len(lines) and lines[index] != r"\]":
                raw = lines[index]
                equation.extend(textwrap.wrap(raw, width=92, subsequent_indent="  ") or [""])
                index += 1
            index += 1
            story.append(Preformatted("\n".join(equation), STYLES["equation"]))
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)", line)
        bullet = re.match(r"^-\s+(.*)", line)
        if numbered or bullet:
            flush()
            marker = f"{numbered.group(1)}." if numbered else "-"
            content = numbered.group(2) if numbered else bullet.group(1)
            story.append(Paragraph(f"{marker}&nbsp;&nbsp;{inline_markup(content)}", STYLES["list"]))
            index += 1
            continue
        if not line.strip():
            flush()
        else:
            paragraph_lines.append(line)
        index += 1
    flush()
    return story


def page_header_footer(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    canvas.setFont(SERIF, 7.5)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    if page > 1:
        canvas.drawString(MARGIN_X, PAGE_HEIGHT - 10.5 * mm, "MedTAWS-GRPO")
        canvas.drawRightString(PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 10.5 * mm, "Development manuscript")
    canvas.drawCentredString(PAGE_WIDTH / 2, 9 * mm, str(page))
    canvas.restoreState()


def build():
    frame = Frame(
        MARGIN_X,
        MARGIN_BOTTOM,
        CONTENT_WIDTH,
        PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    document = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        title="MedTAWS-GRPO",
        author="Anonymous Author(s)",
        subject="Interactive 3D medical image segmentation workflow search and group-relative alignment",
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
    )
    document.addPageTemplates(PageTemplate(id="Academic", frames=[frame], onPage=page_header_footer))
    document.build(parse_manuscript())


if __name__ == "__main__":
    build()
