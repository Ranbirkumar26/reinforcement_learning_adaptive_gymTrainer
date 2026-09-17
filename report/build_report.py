from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "report" / "report.md"
OUTPUT_PDF = ROOT / "report" / "Adaptive_RL_Gym_Coach_Report.pdf"


def _split_sections(markdown: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "Report"
    current_lines: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            current_title = line[2:].strip()
        elif line.startswith("## "):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    return sections


def _paragraphs(lines: list[str], styles) -> list:
    flowables: list = []
    bullet_items: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if bullet_items:
                flowables.append(
                    ListFlowable(
                        [ListItem(Paragraph(item, styles["Body"])) for item in bullet_items],
                        bulletType="bullet",
                        leftIndent=18,
                    )
                )
                bullet_items = []
            flowables.append(Spacer(1, 0.08 * inch))
        elif stripped.startswith("- "):
            bullet_items.append(stripped[2:])
        elif stripped[0:2].isdigit() or stripped.startswith(tuple(f"{i}." for i in range(1, 10))):
            bullet_items.append(stripped)
        else:
            if bullet_items:
                flowables.append(
                    ListFlowable(
                        [ListItem(Paragraph(item, styles["Body"])) for item in bullet_items],
                        bulletType="bullet",
                        leftIndent=18,
                    )
                )
                bullet_items = []
            flowables.append(Paragraph(stripped, styles["Body"]))
    if bullet_items:
        flowables.append(
            ListFlowable(
                [ListItem(Paragraph(item, styles["Body"])) for item in bullet_items],
                bulletType="bullet",
                leftIndent=18,
            )
        )
    return flowables


def build_report(output_pdf: Path = OUTPUT_PDF) -> Path:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleMain",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#172033"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1f4e79"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.4,
            leading=14,
            spaceAfter=5,
        )
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_pdf),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Adaptive RL Gym Coach Report",
    )
    markdown = REPORT_MD.read_text(encoding="utf-8")
    title = markdown.splitlines()[0].replace("# ", "")
    flowables: list = [Paragraph(title, styles["TitleMain"])]
    summary_table = Table(
        [
            ["Prototype scope", "Squat-only adaptive coaching MVP"],
            ["Core stack", "Python, Streamlit, MediaPipe, Gymnasium, Stable-Baselines3"],
            ["Outputs", "Landmarks, rep features, coaching JSON, charts, report, PPT"],
        ],
        colWidths=[1.55 * inch, 4.95 * inch],
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eef7")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#172033")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c4d6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    flowables.extend([summary_table, Spacer(1, 0.18 * inch)])
    for section_title, lines in _split_sections(markdown):
        if section_title == title:
            continue
        flowables.append(Paragraph(section_title, styles["Heading"]))
        flowables.extend(_paragraphs(lines, styles))
    document.build(flowables)
    return output_pdf


if __name__ == "__main__":
    print(build_report())
