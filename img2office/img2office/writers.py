"""Writers: Analysis -> editable .docx / .xlsx"""
from __future__ import annotations

import math
import os
import re
from typing import Optional

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .engine import Analysis, Paragraph, Table

_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _safe(s: str) -> str:
    return _ILLEGAL.sub("", s)


# --------------------------------------------------------------------------- #
# Word
# --------------------------------------------------------------------------- #

_ALIGN = {
    "left": WD_ALIGN_PARAGRAPH.LEFT,
    "center": WD_ALIGN_PARAGRAPH.CENTER,
    "right": WD_ALIGN_PARAGRAPH.RIGHT,
    "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
}


def _set_font(run, size=None, bold=None, color_black=False):
    run.font.name = "Calibri"
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), "Calibri")
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color_black:
        run.font.color.rgb = RGBColor(0, 0, 0)


def write_docx(an: Analysis, path: str) -> str:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)  # A4
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    usable_cm = 21.0 - 4.4

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08

    last_was_table = False
    for el in an.elements:
        if isinstance(el, Paragraph):
            _add_paragraph(doc, el)
            last_was_table = False
        else:
            if last_was_table:
                doc.add_paragraph()  # keep two adjacent tables from fusing together
            _add_table(doc, el, usable_cm)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            last_was_table = True
    if not an.elements:
        doc.add_paragraph("")
    # python-docx always starts with an empty paragraph? (no) - nothing to remove
    doc.save(path)
    return path


def _add_paragraph(doc, p: Paragraph):
    if p.kind == "bullet":
        para = doc.add_paragraph(style="List Bullet")
    elif p.kind == "heading":
        para = doc.add_paragraph(style=f"Heading {p.heading_level}")
    else:
        para = doc.add_paragraph()
    run = para.add_run(_safe(p.text))
    if p.kind == "heading":
        _set_font(run, size=round(11 * p.size_ratio * 2) / 2, bold=True, color_black=True)
        para.paragraph_format.space_before = Pt(10)
        para.paragraph_format.space_after = Pt(6)
    else:
        _set_font(run, bold=True if p.bold else None)
    para.alignment = _ALIGN.get(p.align, WD_ALIGN_PARAGRAPH.LEFT)
    if p.indent_pt and p.kind != "bullet":
        para.paragraph_format.left_indent = Pt(min(p.indent_pt, 250))


def _add_table(doc, t: Table, usable_cm: float):
    R, C = t.nrows, t.ncols
    tbl = doc.add_table(rows=R, cols=C)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    if t.ruled:
        tbl.style = "Table Grid"
    else:
        tbl.style = "Table Grid"  # editing is easier with visible cells; users can remove borders in one click
    tbl.autofit = False
    # column widths proportional to what was seen in the image
    if t.col_widths_px and len(t.col_widths_px) == C and sum(t.col_widths_px) > 0:
        tot = float(sum(t.col_widths_px))
        widths = [max(1.0, usable_cm * w / tot) for w in t.col_widths_px]
        scale = min(1.0, usable_cm / sum(widths))
        widths = [w * scale for w in widths]
        for c, w in enumerate(widths):
            for r in range(R):
                tbl.cell(r, c).width = Cm(w)
    merged_children = set()
    for (r0, c0, r1, c1) in t.merges:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) != (r0, c0):
                    merged_children.add((r, c))
    for r in range(R):
        for c in range(C):
            if (r, c) in merged_children:
                continue
            _fill_cell(tbl.cell(r, c), t.cells[r][c], bold=_is_header(t, r))
    for (r0, c0, r1, c1) in t.merges:
        a, b = tbl.cell(r0, c0), tbl.cell(r1, c1)
        txt = t.cells[r0][c0]
        m = a.merge(b)
        for p in list(m.paragraphs)[1:]:  # merging concatenates empty paragraphs - clean them
            if not p.text:
                p._element.getparent().remove(p._element)
        # re-write the text so only the real content remains
        for p in m.paragraphs:
            for run in list(p.runs):
                run._element.getparent().remove(run._element)
        _fill_cell(m, txt, bold=_is_header(t, r0))


def _fill_cell(cell, text, bold=False):
    para = cell.paragraphs[0]
    for run in list(para.runs):
        run._element.getparent().remove(run._element)
    lines = _safe(text).split("\n")
    run = para.add_run(lines[0])
    _set_font(run, size=10.5, bold=bold or None)
    para.paragraph_format.space_after = Pt(0)
    for extra in lines[1:]:
        p2 = cell.add_paragraph()
        r2 = p2.add_run(extra)
        _set_font(r2, size=10.5, bold=bold or None)
        p2.paragraph_format.space_after = Pt(0)
    if _parse_number(text)[0] is not None:
        para.alignment = WD_ALIGN_PARAGRAPH.RIGHT


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #

_CUR = "$€£₹¥"


def _parse_number(s: str):
    """-> (value | None, number_format). Conservative: IDs with leading zeros stay text."""
    raw = s.strip().replace("−", "-").replace("–", "-").replace(" ", "")
    if not raw or len(raw) > 22:
        return None, None
    neg = False
    if raw.startswith("(") and raw.endswith(")"):
        neg, raw = True, raw[1:-1]
    sym = ""
    if raw[:1] == "-":
        neg, raw = True, raw[1:]
    if raw[:1] == "+":
        raw = raw[1:]
    if raw[:1] in _CUR and len(raw) > 1:
        sym, raw = raw[0], raw[1:]
    if raw[:1] == "-":
        neg, raw = True, raw[1:]
    pct = raw.endswith("%")
    if pct:
        raw = raw[:-1]
    if not re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?|\d+(\.\d+)?", raw):
        return None, None
    comma = "," in raw
    digits = raw.replace(",", "")
    intpart = digits.split(".")[0]
    if len(intpart) > 1 and intpart.startswith("0"):
        return None, None  # 007, 0123 ... identifiers
    if len(intpart) > 15:
        return None, None
    dec = len(digits.split(".")[1]) if "." in digits else 0
    val = float(digits)
    if dec == 0 and not pct:
        val = int(digits)
    if neg:
        val = -val
    if pct:
        val = val / 100.0
        fmt = "0" + ("." + "0" * dec if dec else "") + "%"
    else:
        base = ("#,##0" if comma else "0") + ("." + "0" * dec if dec else "")
        fmt = f'"{sym}"{base}' if sym else base
    return val, fmt


def _is_header(t: Table, r: int) -> bool:
    """First row is a header if it has no numbers while later rows do."""
    if r != 0 or t.nrows < 2:
        return False
    first = [c for c in t.cells[0] if c]
    if not first or any(_parse_number(c)[0] is not None for c in first):
        return False
    body = [c for row in t.cells[1:] for c in row if c]
    return any(_parse_number(c)[0] is not None for c in body) or len(first) >= max(2, t.ncols // 2 + 1)


def write_xlsx(an: Analysis, path: str) -> str:
    wb = Workbook()
    wb.remove(wb.active)
    thin = Side(style="thin", color="999999")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    flag = PatternFill("solid", fgColor="FFF2CC")

    elements = an.elements
    tables = [e for e in elements if isinstance(e, Table)]
    if not tables:  # text only (explicit Excel request): one text sheet
        ws = wb.create_sheet("Text")
        for i, p in enumerate([e for e in elements if isinstance(e, Paragraph)], 1):
            c = ws.cell(row=i, column=1, value=_safe(p.text))
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if p.bold or p.kind == "heading":
                c.font = Font(name="Calibri", bold=True)
        ws.column_dimensions["A"].width = 110
        wb.save(path)
        return path

    # paragraphs go on the sheet of the nearest table: before-text above it, trailing text below the last table
    sheets, pending = [], []
    for e in elements:
        if isinstance(e, Paragraph):
            pending.append(e)
        else:
            sheets.append({"table": e, "above": pending, "below": []})
            pending = []
    sheets[-1]["below"] = pending

    for si, s in enumerate(sheets, 1):
        t: Table = s["table"]
        ws = wb.create_sheet(f"Table {si}" if len(sheets) > 1 else "Table")
        row = 1
        for p in s["above"]:
            c = ws.cell(row=row, column=1, value=_safe(p.text))
            c.font = Font(name="Calibri", bold=bool(p.bold or p.kind == "heading"), size=14 if p.kind == "heading" else 11)
            row += 1
        if s["above"]:
            row += 1
        top = row
        merged_children = set()
        for (r0, c0, r1, c1) in t.merges:
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if (r, c) != (r0, c0):
                        merged_children.add((r, c))
        widths = [8.0] * t.ncols
        for r in range(t.nrows):
            for c in range(t.ncols):
                cell = ws.cell(row=top + r, column=c + 1)
                cell.border = border
                cell.alignment = Alignment(wrap_text=True, vertical="center")
                if (r, c) in merged_children:
                    continue
                txt = _safe(t.cells[r][c])
                val, fmt = _parse_number(txt)
                if val is not None:
                    cell.value = val
                    cell.number_format = fmt
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                else:
                    cell.value = txt if txt else None
                if _is_header(t, r):
                    cell.font = Font(name="Calibri", bold=True)
                    cell.fill = PatternFill("solid", fgColor="E7EEF8")
                if (r, c) in t.low_conf:
                    cell.fill = flag
                longest = max((len(x) for x in txt.split("\n")), default=0)
                if not any(m[0] == r and m[1] == c for m in t.merges):
                    widths[c] = max(widths[c], min(60.0, longest * 1.15 + 2))
        for (r0, c0, r1, c1) in t.merges:
            ws.merge_cells(start_row=top + r0, start_column=c0 + 1, end_row=top + r1, end_column=c1 + 1)
            for r in range(r0, r1 + 1):  # border on every merged slot
                for c in range(c0, c1 + 1):
                    ws.cell(row=top + r, column=c + 1).border = border
        for c, w in enumerate(widths):
            ws.column_dimensions[get_column_letter(c + 1)].width = w
        row = top + t.nrows + 1
        for p in s["below"]:
            c = ws.cell(row=row, column=1, value=_safe(p.text))
            c.font = Font(name="Calibri", bold=bool(p.bold or p.kind == "heading"))
            row += 1
        if _is_header(t, 0):
            ws.freeze_panes = f"A{top + 1}"
    wb.save(path)
    return path
