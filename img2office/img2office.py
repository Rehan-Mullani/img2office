#!/usr/bin/env python3
"""
img2office - turn an image into an EDITABLE Word (.docx) or Excel (.xlsx) file.

Pipeline
  1. Pre-process   : EXIF/OSD orientation, de-skew, adaptive up-scaling, background
                     normalisation (shadows, tinted paper, dark mode).
  2. Detect tables : (a) ruled tables  -> grid of ruling lines, merged cells recovered
                     (b) un-ruled tables -> column "gutters" found from word positions.
  3. OCR           : Tesseract LSTM.  Ruled tables are OCR'd cell-by-cell, everything
                     else page-wise with a confidence-driven fallback.
  4. Rebuild layout: paragraphs, headings (size), bold lines (stroke width), bullets,
                     numbered items, alignment, tables.
  5. Decide format : only paragraphs -> Word,  mostly/only tables -> Excel,
                     mixed -> whichever dominates (Word keeps real Word tables,
                     Excel keeps the text rows above/below the table).
  6. Write         : real Word paragraphs / tables, real Excel cells (numbers become
                     numeric cells, merged cells preserved, headers styled).

Usage:  python img2office.py scan.png [more.jpg ...] -o out_dir [--format auto|word|excel|both]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # we parallelise ourselves

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

__all__ = ["convert", "analyze", "ConversionResult"]

TARGET_CC_HEIGHT = 24          # glyph height (px) Tesseract is happiest with
_ILLEGAL_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Word:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float
    key: tuple = (0, 0, 0)

    @property
    def cx(self): return (self.x0 + self.x1) / 2

    @property
    def cy(self): return (self.y0 + self.y1) / 2

    @property
    def h(self): return self.y1 - self.y0


@dataclass
class Cell:
    r: int
    c: int
    rs: int
    cs: int
    text: str
    conf: float = 100.0


@dataclass
class Table:
    kind: str                       # "ruled" | "unruled"
    bbox: tuple                     # x0, y0, x1, y1
    nrows: int
    ncols: int
    cells: list = field(default_factory=list)

    @property
    def n_words(self):
        return sum(len(c.text.split()) for c in self.cells)


@dataclass
class Para:
    text: str
    bbox: tuple
    kind: str = "normal"            # normal | heading | bullet | numbered
    level: int = 0
    bold: bool = False
    align: str = "left"             # left | center | right
    conf: float = 100.0

    @property
    def n_words(self):
        return len(self.text.split())


@dataclass
class ConversionResult:
    input: str
    output: str                     # path of main output
    format: str                     # "docx" | "xlsx"
    layout: str                     # "text" | "table" | "mixed"
    reason: str
    n_tables: int
    n_paragraphs: int
    n_words: int
    avg_confidence: float
    extra_outputs: list = field(default_factory=list)
    debug_image: Optional[str] = None
    elements: list = field(default_factory=list, repr=False)

    def summary(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "elements"}
        return d


def clean_text(s: str) -> str:
    for k, v in _LIGATURES.items():
        s = s.replace(k, v)
    s = _ILLEGAL_XML.sub("", s)
    return re.sub(r"[ \t ]+", " ", s).strip()


# --------------------------------------------------------------------------- #
# 1. Pre-processing
# --------------------------------------------------------------------------- #
def load_gray(src) -> np.ndarray:
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, "white")
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    return np.array(img.convert("L"))


def median_cc_height(gray: np.ndarray) -> float:
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return 20.0
    h, w, a = st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_AREA]
    H = gray.shape[0]
    keep = (h >= 5) & (h <= 0.25 * H) & (a >= 8) & (w <= 8 * h) & (h <= 8 * w)
    if keep.sum() < 3:
        return 20.0
    return float(np.median(h[keep]))


def auto_orient(gray: np.ndarray) -> np.ndarray:
    """Fix 90/180/270 degree rotations using Tesseract OSD (only if confident)."""
    try:
        osd = pytesseract.image_to_osd(Image.fromarray(gray), config="--psm 0 -c min_characters_to_try=20",
                                       output_type=Output.DICT)
        rot, conf = int(osd.get("rotate", 0)), float(osd.get("orientation_conf", 0))
        if rot and conf >= 3.0:
            code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot]
            return cv2.rotate(gray, code)
    except Exception:
        pass
    return gray


def deskew(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    f = min(1.0, 1000.0 / max(h, w))
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if (bw > 0).sum() < 50:
        return gray
    sh, sw = bw.shape

    def score(a):
        M = cv2.getRotationMatrix2D((sw / 2, sh / 2), a, 1)
        r = cv2.warpAffine(bw, M, (sw, sh), flags=cv2.INTER_NEAREST)
        p = r.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(p) ** 2))

    best = max(np.arange(-10, 10.01, 1.0), key=score)
    best = max(np.arange(best - 1, best + 1.01, 0.1), key=score)
    if abs(best) < 0.2:
        return gray
    pad = int(0.03 * max(h, w))
    g = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    M = cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), best, 1)
    return cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def normalize_background(gray: np.ndarray, med_h: float) -> np.ndarray:
    k = int(max(15, 1.3 * med_h)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    bg = cv2.GaussianBlur(bg, (0, 0), k)
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    norm[norm >= 235] = 255
    return norm


def preprocess(gray: np.ndarray):
    if np.median(gray) < 110:                      # light text on dark background
        gray = 255 - gray
    gray = auto_orient(gray)
    gray = deskew(gray)
    med = median_cc_height(gray)
    scale = float(np.clip(TARGET_CC_HEIGHT / med, 0.5, 3.5))
    scale = min(scale, 6500.0 / max(gray.shape))
    if abs(scale - 1) > 0.05:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    med_h = median_cc_height(gray)
    norm = normalize_background(gray, med_h)
    return norm, med_h


# --------------------------------------------------------------------------- #
# OCR helpers
# --------------------------------------------------------------------------- #
def ocr_words(img: np.ndarray, psm: int, lang: str) -> list:
    data = pytesseract.image_to_data(Image.fromarray(img), lang=lang, config=f"--oem 1 --psm {psm}",
                                     output_type=Output.DICT)
    out = []
    for i, t in enumerate(data["text"]):
        t = clean_text(t or "")
        if not t:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append(Word(t, x, y, x + w, y + h, conf, (data["block_num"][i], data["par_num"][i], data["line_num"][i])))
    return out


def mean_conf(words) -> float:
    return float(np.mean([w.conf for w in words])) if words else 0.0


def fix_numeric(t: str) -> str:
    """Repair classic OCR confusions (O->0, l->1 ...) inside clearly numeric cells."""
    s = t.strip()
    if not s or len(s) > 25:
        return t
    core = re.sub(r"[\s,.\-+%$€£₹()/:]", "", s)
    if len(core) < 2:
        return t
    digits = sum(ch.isdigit() for ch in core)
    if digits < 2 or digits < 2 * len(core) / 3:
        return t
    if any(not (ch.isdigit() or ch in "OolI|") for ch in core):
        return t
    return s.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"}))


# --------------------------------------------------------------------------- #
# 2a. Ruled tables
# --------------------------------------------------------------------------- #
def _filter_lines(mask: np.ndarray, horizontal: bool, min_aspect: float = 12.0) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        length, thick = (w, h) if horizontal else (h, w)
        keep[i] = length / max(thick, 1) >= min_aspect
    return (keep[lab] * 255).astype(np.uint8)


def _cluster(vals, tol=7):
    vals = sorted(vals)
    groups = []
    for v in vals:
        if groups and v - np.mean(groups[-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def detect_ruled_tables(norm: np.ndarray, med_h: float):
    """Return (list_of_grids, line_mask).  Each grid = dict(xs, ys, hm, vm, bbox)."""
    H, W = norm.shape
    ink = ((norm < 200) * 255).astype(np.uint8)
    minlen = int(max(2.2 * med_h, 0.02 * W))
    hm = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (minlen, 1)))
    vm = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, minlen)))
    hm, vm = _filter_lines(hm, True), _filter_lines(vm, False)
    line_mask = cv2.dilate(cv2.bitwise_or(hm, vm), np.ones((3, 3), np.uint8), iterations=2)

    grid = cv2.dilate(cv2.bitwise_or(hm, vm), np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    grids = []
    for i in range(1, n):
        x, y, w, h = (int(st[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        if w < 3 * med_h or h < 1.8 * med_h:
            continue
        region = lab[y:y + h, x:x + w] == i
        sub_h = hm[y:y + h, x:x + w] * region
        sub_v = vm[y:y + h, x:x + w] * region
        ys, xs = [], []
        nh, _, sth, _ = cv2.connectedComponentsWithStats(sub_h, connectivity=8)
        for j in range(1, nh):
            ys.append(sth[j, cv2.CC_STAT_TOP] + sth[j, cv2.CC_STAT_HEIGHT] / 2)
        nv, _, stv, _ = cv2.connectedComponentsWithStats(sub_v, connectivity=8)
        for j in range(1, nv):
            xs.append(stv[j, cv2.CC_STAT_LEFT] + stv[j, cv2.CC_STAT_WIDTH] / 2)
        ys, xs = _cluster(ys), _cluster(xs)
        if len(ys) < 2 or len(xs) < 2 or (len(ys) - 1) * (len(xs) - 1) < 2:
            continue                                   # a lone box / text frame, not a table
        ys = [v + y for v in ys]
        xs = [v + x for v in xs]
        grids.append(dict(xs=[int(round(v)) for v in xs], ys=[int(round(v)) for v in ys],
                          bbox=(int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1]))))
    return grids, line_mask, hm, vm


def _edge(mask, a0, a1, p, horizontal, band=4):
    """True if a ruling line is present between a0..a1 at position p."""
    if a1 - a0 < 8:
        return True
    if horizontal:
        seg = mask[max(p - band, 0):p + band + 1, a0 + 4:a1 - 3]
        return seg.size > 0 and (seg.max(axis=0) > 0).mean() > 0.5
    seg = mask[a0 + 4:a1 - 3, max(p - band, 0):p + band + 1]
    return seg.size > 0 and (seg.max(axis=1) > 0).mean() > 0.5


def build_ruled_table(g, norm, hm, vm, line_mask, med_h, lang, pool) -> Table:
    xs, ys = g["xs"], g["ys"]
    R, C = len(ys) - 1, len(xs) - 1
    owner = -np.ones((R, C), int)
    spans = []
    for r in range(R):
        for c in range(C):
            if owner[r, c] >= 0:
                continue
            ce = c
            while ce + 1 < C and owner[r, ce + 1] < 0 and not _edge(vm, ys[r], ys[r + 1], xs[ce + 1], False):
                ce += 1
            re_ = r
            while re_ + 1 < R:
                nxt = re_ + 1
                if any(owner[nxt, cc] >= 0 for cc in range(c, ce + 1)):
                    break
                if any(_edge(hm, xs[cc], xs[cc + 1], ys[nxt], True) for cc in range(c, ce + 1)):
                    break
                if any(_edge(vm, ys[nxt], ys[nxt + 1], xs[cc], False) for cc in range(c + 1, ce + 1)):
                    break
                re_ += 1
            owner[r:re_ + 1, c:ce + 1] = len(spans)
            spans.append((r, c, re_ - r + 1, ce - c + 1))

    work = norm.copy()
    work[line_mask > 0] = 255

    def do_cell(span):
        r, c, rs, cs = span
        pad = 5
        x0, x1 = xs[c] + pad, xs[c + cs] - pad
        y0, y1 = ys[r] + pad, ys[r + rs] - pad
        if x1 - x0 < 6 or y1 - y0 < 6:
            return Cell(r, c, rs, cs, "")
        crop = work[y0:y1, x0:x1]
        dark = (crop < 150).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        good = [i for i in range(1, n)
                if st[i, cv2.CC_STAT_AREA] >= 8 and max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) >= 0.25 * med_h]
        if not good:
            return Cell(r, c, rs, cs, "")
        clean = np.full_like(crop, 255)
        keep = np.isin(lab, good)
        clean[keep] = crop[keep]
        clean = cv2.copyMakeBorder(clean, 14, 14, 14, 14, cv2.BORDER_CONSTANT, value=255)
        single = (y1 - y0) <= 3.0 * med_h
        best, best_conf = [], -1
        for psm in ([7, 6, 8] if single else [6, 4]):
            ws = ocr_words(clean, psm, lang)
            cf = mean_conf(ws)
            if ws and cf > best_conf:
                best, best_conf = ws, cf
            if best and best_conf >= 70:
                break
        if not best:
            return Cell(r, c, rs, cs, "", 0)
        lines = {}
        for w in best:
            lines.setdefault(w.key, []).append(w)
        txt = " ".join(" ".join(w.text for w in sorted(v, key=lambda w: w.x0))
                       for _, v in sorted(lines.items()))
        return Cell(r, c, rs, cs, fix_numeric(clean_text(txt)), best_conf)

    cells = list(pool.map(do_cell, spans))
    return Table("ruled", g["bbox"], R, C, cells)


# --------------------------------------------------------------------------- #
# 2b. Un-ruled (borderless) tables from word geometry
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    words: list

    @property
    def y0(self): return min(w.y0 for w in self.words)

    @property
    def y1(self): return max(w.y1 for w in self.words)

    @property
    def cy(self): return (self.y0 + self.y1) / 2


def group_rows(words) -> list:
    rows = []
    for w in sorted(words, key=lambda w: w.cy):
        if rows:
            r = rows[-1]
            ov = min(r.y1, w.y1) - max(r.y0, w.y0)
            if ov > 0.5 * min(r.y1 - r.y0, w.h):
                r.words.append(w)
                continue
        rows.append(Row([w]))
    for r in rows:
        r.words.sort(key=lambda w: w.x0)
    return rows


def segments(row: Row, gap: float) -> list:
    segs, cur = [], [row.words[0]]
    for w in row.words[1:]:
        if w.x0 - cur[-1].x1 > gap:
            segs.append(cur)
            cur = [w]
        else:
            cur.append(w)
    segs.append(cur)
    return segs


_LIST_MARK_CELL = re.compile(r"^(\d{1,3}[.)]|[•·●▪◦○■*\-–—eo«®©¢°])$")


def detect_unruled_tables(words, text_h: float):
    if not words:
        return [], set()
    gap = 1.0 * text_h
    rows = group_rows(words)
    segs = [segments(r, gap) for r in rows]
    n = len(rows)
    tables, consumed = [], set()
    i = 0
    while i < n:
        if len(segs[i]) < 2:
            i += 1
            continue
        last, k = i, i + 1
        while k < n:
            if rows[k].y0 - rows[k - 1].y1 > 2.5 * text_h:
                break
            if len(segs[k]) >= 2:
                last = k
                k += 1
                continue
            if (k + 1 < n and len(segs[k + 1]) >= 2 and rows[k + 1].y0 - rows[k - 1].y1 < 3.0 * text_h
                    and rows[k + 1].y0 - rows[k].y1 <= 2.5 * text_h):
                k += 1
                continue
            break
        run = list(range(i, last + 1))
        i = last + 1
        tab_rows = [j for j in run if len(segs[j]) >= 2]
        avg_segs = np.mean([len(segs[j]) for j in tab_rows])
        if not ((len(tab_rows) >= 3) or (len(tab_rows) >= 2 and avg_segs >= 3)):
            continue
        t = _table_from_rows([rows[j] for j in run], [segs[j] for j in run], gap)
        if t is None:
            continue
        tables.append(t)
        for j in run:
            for w in rows[j].words:
                consumed.add(id(w))
    return tables, consumed


def _table_from_rows(rows, segs, gap) -> Optional[Table]:
    allw = [w for r in rows for w in r.words]
    lo, hi = min(w.x0 for w in allw), max(w.x1 for w in allw)
    occ = np.zeros(hi - lo + 2, int)
    for r in rows:
        cov = np.zeros_like(occ, bool)
        for w in r.words:
            cov[w.x0 - lo:w.x1 - lo + 1] = True
        occ += cov
    tol = int(0.12 * len(rows))
    free = occ <= tol
    cols, start, x = [], None, 0
    # content intervals = maximal runs that are NOT part of a wide-enough gutter
    gutters = []
    run_start = None
    for x in range(len(free)):
        if free[x] and run_start is None:
            run_start = x
        if (not free[x] or x == len(free) - 1) and run_start is not None:
            end = x - 1 if not free[x] else x
            if end - run_start + 1 >= gap and run_start > 0 and end < len(free) - 1:
                gutters.append((run_start, end))
            run_start = None
    prev = 0
    for a, b in gutters:
        if a - prev > 0:
            cols.append((prev + lo, a - 1 + lo))
        prev = b + 1
    if len(free) - prev > 0:
        cols.append((prev + lo, len(free) - 1 + lo))
    if not (2 <= len(cols) <= 40):
        return None

    cells = []
    for r, (row, sg) in enumerate(zip(rows, segs)):
        # recompute segments with the *table's* gap so that cell text is consistent
        last_end = -1
        for seg in sg:
            x0, x1 = seg[0].x0, seg[-1].x1
            hit = [ci for ci, (a, b) in enumerate(cols)
                   if min(x1, b) - max(x0, a) > 0.15 * min(x1 - x0, b - a + 1)]
            if not hit:
                ctr = (x0 + x1) / 2
                hit = [int(np.argmin([abs(ctr - (a + b) / 2) for a, b in cols]))]
            c0, c1 = hit[0], hit[-1]
            c0 = max(c0, last_end + 1)
            text = clean_text(" ".join(w.text for w in seg))
            conf = mean_conf(seg)
            if c0 >= len(cols):
                if cells and cells[-1].r == r:
                    cells[-1].text += " " + text
                continue
            c1 = max(c1, c0)
            cells.append(Cell(r, c0, 1, c1 - c0 + 1, fix_numeric(text), conf))
            last_end = c1
    if not cells:
        return None
    # --- plausibility checks --------------------------------------------------
    words_per_cell = np.mean([len(c.text.split()) for c in cells])
    if words_per_cell > 7:                                  # two-column prose, not a table
        return None
    col0 = [c.text for c in cells if c.c == 0 and c.cs == 1]
    if col0 and all(_LIST_MARK_CELL.match(t) for t in col0):  # bullet / numbered list
        return None
    bbox = (lo, min(r.y0 for r in rows), hi, max(r.y1 for r in rows))
    return Table("unruled", bbox, len(rows), len(cols), cells)


# --------------------------------------------------------------------------- #
# 3. Paragraph reconstruction
# --------------------------------------------------------------------------- #
_BULLET = re.compile(r"^([•·●▪◦○■□◆◇➢➤►▶\-–—*]|[eo«®©¢°])\s+(?=\S)")
_NUMBER = re.compile(r"^(\(?\d{1,3}[.)]|\(?[a-z][.)]|\(?[ivx]{1,4}[.)])\s+(?=\S)")


@dataclass
class TLine:
    words: list
    block: int
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    cc_h: float = 0
    stroke: float = 0

    @property
    def cy(self): return (self.y0 + self.y1) / 2


def _stroke_metrics(ink_gray: np.ndarray, box):
    x0, y0, x1, y1 = box
    crop = ink_gray[max(y0, 0):y1, max(x0, 0):x1]
    if crop.size == 0:
        return 0.0, 0.0
    bw = (cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0).astype(np.uint8)
    area = int(bw.sum())
    if area < 40:
        return 0.0, 0.0
    er = cv2.erode(bw, np.ones((3, 3), np.uint8))
    perim = max(area - int(er.sum()), 1)
    stroke = 2.0 * area / perim
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    hs = [st[i, cv2.CC_STAT_HEIGHT] for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 10]
    cc_h = float(np.median(hs)) if hs else 0.0
    return stroke, cc_h


def build_lines(words, norm) -> list:
    groups = {}
    for w in words:
        groups.setdefault(w.key, []).append(w)
    lines = []
    for key, ws in groups.items():
        ws.sort(key=lambda w: w.x0)
        text = clean_text(" ".join(w.text for w in ws))
        if not text:
            continue
        if len(text) <= 2 and mean_conf(ws) < 50 and not any(ch.isalnum() for ch in text):
            continue                                        # speckle noise
        ln = TLine(ws, key[0], text, min(w.x0 for w in ws), min(w.y0 for w in ws),
                   max(w.x1 for w in ws), max(w.y1 for w in ws))
        ln.stroke, ln.cc_h = _stroke_metrics(norm, (ln.x0, ln.y0, ln.x1, ln.y1))
        lines.append(ln)
    lines.sort(key=lambda l: (min(w.key for w in l.words), l.y0))
    return lines


def build_paragraphs(lines, norm_w: int) -> list:
    if not lines:
        return []
    # block geometry
    blocks = {}
    for l in lines:
        blocks.setdefault(l.block, []).append(l)
    bl = {b: (min(l.x0 for l in ls), max(l.x1 for l in ls)) for b, ls in blocks.items()}
    # typical in-block pitch
    pitches = []
    for ls in blocks.values():
        ls_s = sorted(ls, key=lambda l: l.y0)
        pitches += [b.cy - a.cy for a, b in zip(ls_s, ls_s[1:]) if b.cy - a.cy > 0]
    word_hs = [w.h for l in lines for w in l.words if len(w.text) >= 3]
    text_h = float(np.median(word_hs)) if word_hs else 20.0
    page_pitch = float(np.median(pitches)) if pitches else 1.6 * text_h
    page_pitch = max(page_pitch, 1.1 * text_h)

    groups = []
    for ln in lines:
        mark_bullet = bool(_BULLET.match(ln.text))
        mark_number = bool(_NUMBER.match(ln.text))
        new = True
        if groups:
            g = groups[-1]
            prev = g[-1]
            left, right = bl[ln.block]
            width = max(right - left, 1)
            same_block = prev.block == ln.block
            pitch = ln.cy - prev.cy
            prev_short = prev.x1 < right - 0.25 * width and (right - left) > 12 * text_h
            first_is_list = bool(_BULLET.match(g[0].text) or _NUMBER.match(g[0].text))
            indented = ln.x0 - left > 1.5 * text_h and not first_is_list and not prev_short
            if (same_block and 0 < pitch <= 1.3 * page_pitch and not mark_bullet and not mark_number
                    and not prev_short and not indented):
                new = False
        if new:
            groups.append([ln])
        else:
            groups[-1].append(ln)

    # geometry / style statistics for headings, bold, alignment
    body_cc = float(np.median([l.cc_h for l in lines if l.cc_h > 0] or [1]))
    body_lines = [l for l in lines if len(l.text) >= 12 and l.stroke > 0 and l.cc_h > 0]
    ratios = [l.stroke / l.cc_h for l in body_lines]
    base_ratio = float(np.median(ratios)) if len(ratios) >= 3 else None
    page_x0 = min(l.x0 for l in lines)
    page_x1 = max(l.x1 for l in lines)
    page_cx = (page_x0 + page_x1) / 2
    page_w = max(page_x1 - page_x0, 1)

    paras = []
    for g in groups:
        raw = " ".join(l.text for l in g)
        first = g[0].text
        kind, level = "normal", 0
        m = _BULLET.match(first)
        if m:
            kind = "bullet"
            raw = raw[m.end():] if raw.startswith(first[:m.end()]) else raw
        elif _NUMBER.match(first):
            kind = "numbered"
        text = clean_text(raw)

        ccs = [l.cc_h for l in g if l.cc_h > 0]
        size_ratio = (float(np.median(ccs)) / body_cc) if ccs and body_cc else 1.0
        strokes = [l.stroke / l.cc_h for l in g if l.cc_h > 0 and l.stroke > 0]
        bold = False
        if strokes and len(text) >= 2:
            r = float(np.median(strokes))
            if base_ratio:
                bold = r > 1.3 * base_ratio
            else:
                bold = r > 0.22
        if kind == "normal" and len(g) <= 3 and size_ratio >= 1.25 and len(text.split()) <= 20:
            kind = "heading"
            level = 1 if size_ratio >= 1.7 else 2 if size_ratio >= 1.4 else 3
            bold = True
        # alignment
        x0, x1 = min(l.x0 for l in g), max(l.x1 for l in g)
        align = "left"
        if page_w > 0 and (x1 - x0) < 0.85 * page_w:
            centred = all(abs((l.x0 + l.x1) / 2 - page_cx) < 0.03 * page_w for l in g)
            if centred and (x0 - page_x0) > 0.05 * page_w and (page_x1 - x1) > 0.05 * page_w:
                align = "center"
            elif abs(x1 - page_x1) < 0.02 * page_w and (x0 - page_x0) > 0.15 * page_w:
                align = "right"
        conf = mean_conf([w for l in g for w in l.words])
        paras.append(Para(text, (x0, min(l.y0 for l in g), x1, max(l.y1 for l in g)),
                          kind, level, bold, align, conf))
    return paras


# --------------------------------------------------------------------------- #
# 4. Analysis driver
# --------------------------------------------------------------------------- #
@dataclass
class Analysis:
    elements: list
    tables: list
    paras: list
    n_words: int
    avg_conf: float
    norm: np.ndarray


def analyze(src, lang: str = "eng") -> Analysis:
    gray = load_gray(src)
    norm, med_h = preprocess(gray)
    H, W = norm.shape
    workers = min(8, os.cpu_count() or 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        grids, line_mask, hm, vm = detect_ruled_tables(norm, med_h)
        ruled = [build_ruled_table(g, norm, hm, vm, line_mask, med_h, lang, pool) for g in grids]
        ruled = [t for t in ruled if any(c.text for c in t.cells)]

        text_img = norm.copy()
        text_img[line_mask > 0] = 255
        for t in ruled:
            x0, y0, x1, y1 = t.bbox
            text_img[max(y0 - 8, 0):y1 + 8, max(x0 - 8, 0):x1 + 8] = 255

        has_ink = (text_img < 150).sum() > 30
        words = []
        if has_ink:
            padded = cv2.copyMakeBorder(text_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            words = ocr_words(padded, 3, lang)
            if mean_conf(words) < 70 or not words:
                alt = ocr_words(padded, 6, lang)
                if mean_conf(alt) > mean_conf(words) + 3 and len(alt) >= 0.6 * len(words):
                    words = alt
            for w in words:                                  # undo the 20px padding
                w.x0 -= 20; w.x1 -= 20; w.y0 -= 20; w.y1 -= 20
        words = [w for w in words if not (w.conf < 15 and len(w.text) <= 2 and not w.text.isalnum())]

    text_h = float(np.median([w.h for w in words if len(w.text) >= 3] or [med_h * 1.4]))
    unruled, consumed = detect_unruled_tables(words, text_h)
    rest = [w for w in words if id(w) not in consumed]
    lines = build_lines(rest, norm)
    paras = build_paragraphs(lines, W)

    tables = ruled + unruled
    # merge tables into the paragraph flow by vertical position
    elements, pending = [], sorted(tables, key=lambda t: t.bbox[1])
    for p in paras:
        while pending and pending[0].bbox[1] <= p.bbox[1]:
            elements.append(pending.pop(0))
        elements.append(p)
    elements.extend(pending)

    confs = [w.conf for w in rest] + [c.conf for t in tables for c in t.cells if c.text]
    n_words = len(rest) + sum(t.n_words for t in tables)
    return Analysis(elements, tables, paras, n_words, float(np.mean(confs)) if confs else 0.0, norm)


def decide_format(a: Analysis):
    t_words = sum(t.n_words for t in a.tables)
    p_words = sum(p.n_words for p in a.paras)
    if not a.tables:
        return "docx", "text", "no tables found - paragraph content"
    if p_words <= max(4, 0.05 * (t_words + p_words)) and not any(p.n_words > 8 for p in a.paras):
        return "xlsx", "table", "content is tabular"
    if t_words >= p_words:
        return "xlsx", "mixed", f"mixed content, table-heavy ({t_words} table words vs {p_words} text words)"
    return "docx", "mixed", f"mixed content, text-heavy ({p_words} text words vs {t_words} table words)"


# --------------------------------------------------------------------------- #
# 5. Writers
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"^\(?-?[$€£₹]?\s?(\d{1,3}(,\d{3})+|\d{1,2}(,\d{2})*,\d{3}|\d+)(\.\d+)?\)?%?$")


def to_number(text: str):
    """Return (value, excel_number_format) for clearly numeric text, else None."""
    s = text.strip().replace("Rs.", "").replace("Rs", "").strip()
    if not s or not _NUM.match(s):
        return None
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    pct = s.endswith("%")
    cur = next((ch for ch in "$€£₹" if ch in s), None)
    body = re.sub(r"[()$€£₹%\s-]", "", s)
    digits = body.replace(",", "")
    if len(digits.replace(".", "")) > 15:
        return None
    if re.match(r"^0\d", digits):                          # codes like 007, 0123
        return None
    val = float(digits) if "." in digits else int(digits)
    if neg:
        val = -val
    dec = len(digits.split(".")[1]) if "." in digits else 0
    base = ("#,##0" if "," in body else "0") + ("." + "0" * dec if dec else "")
    if pct:
        return val / 100, "0" + ("." + "0" * dec if dec else "") + "%"
    if cur:
        return val, f'"{cur}"' + ("#,##0" + ("." + "0" * dec if dec else ""))
    return val, base


def write_docx(elements, path: str):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Inches(1)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    normal.paragraph_format.space_after = Pt(8)
    for lvl, size in ((1, 20), (2, 16), (3, 13)):
        st = doc.styles[f"Heading {lvl}"]
        st.font.name, st.font.size, st.font.bold = "Calibri", Pt(size), True
        st.font.color.rgb = RGBColor(0, 0, 0)

    aligns = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
              "right": WD_ALIGN_PARAGRAPH.RIGHT}
    last_was_table = False
    for el in elements:
        if isinstance(el, Para):
            if last_was_table:
                pass
            if el.kind == "heading":
                p = doc.add_heading(level=el.level)
                p.add_run(el.text)
            elif el.kind == "bullet":
                p = doc.add_paragraph(style="List Bullet")
                r = p.add_run(el.text)
                r.bold = el.bold or None
            else:
                p = doc.add_paragraph()
                r = p.add_run(el.text)
                r.bold = el.bold or None
                if el.kind == "numbered":
                    p.paragraph_format.left_indent = Inches(0.35)
                    p.paragraph_format.first_line_indent = Inches(-0.25)
            p.alignment = aligns[el.align]
            last_was_table = False
        else:
            t: Table = el
            tbl = doc.add_table(rows=t.nrows, cols=t.ncols)
            tbl.style = "Table Grid"
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            header = _has_header(t)
            for c in t.cells:
                a = tbl.cell(c.r, c.c)
                if c.rs > 1 or c.cs > 1:
                    a = a.merge(tbl.cell(min(c.r + c.rs - 1, t.nrows - 1), min(c.c + c.cs - 1, t.ncols - 1)))
                a.text = ""
                para = a.paragraphs[0]
                run = para.add_run(c.text)
                if header and c.r == 0:
                    run.bold = True
                if to_number(c.text) is not None:
                    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                para.paragraph_format.space_after = Pt(2)
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
            last_was_table = True
    if not elements:
        doc.add_paragraph("")
    doc.save(path)


def _has_header(t: Table) -> bool:
    """First row looks like a header: text only, while some column below holds numbers."""
    if t.nrows < 2:
        return False
    grid = {}
    for c in t.cells:
        grid[(c.r, c.c)] = c.text
    first = [grid.get((0, c), "") for c in range(t.ncols)]
    filled = [x for x in first if x]
    if not filled or any(to_number(x) is not None for x in filled):
        return False
    for c in range(t.ncols):
        below = [grid.get((r, c), "") for r in range(1, t.nrows)]
        nums = [x for x in below if x and to_number(x) is not None]
        if len(nums) >= max(1, 0.5 * len([x for x in below if x])):
            return True
    return t.nrows >= 3


def write_xlsx(elements, path: str):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    thin = Side(style="thin", color="808080")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # group: text-before + table (+ trailing text on the last group)
    groups, pending = [], []
    for el in elements:
        if isinstance(el, Table):
            groups.append([pending, el, []])
            pending = []
        else:
            pending.append(el)
    if groups:
        groups[-1][2] = pending
    else:
        groups.append([pending, None, []])

    for gi, (before, table, after) in enumerate(groups, 1):
        ws = wb.create_sheet(f"Table {gi}" if len(groups) > 1 else ("Sheet1" if table else "Text"))
        widths = {}
        row = 1

        def put_text(paras, row):
            for p in paras:
                c = ws.cell(row=row, column=1, value=p.text)
                c.data_type = "s"
                c.font = Font(bold=p.bold or p.kind == "heading",
                              size=14 if p.kind == "heading" else 11)
                c.alignment = Alignment(vertical="top")
                row += 1
            return row

        row = put_text(before, row)
        if before and table:
            row += 1
        if table:
            header = _has_header(table)
            r0 = row
            for c in table.cells:
                cell = ws.cell(row=r0 + c.r, column=1 + c.c)
                num = to_number(c.text)
                if num is not None and not (header and c.r == 0):
                    cell.value, cell.number_format = num[0], num[1]
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                else:
                    cell.value = c.text
                    cell.data_type = "s"
                    cell.alignment = Alignment(vertical="center", wrap_text=True,
                                               horizontal="center" if (header and c.r == 0) else None)
                if header and c.r == 0:
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill("solid", fgColor="D9E1F2")
                if c.rs > 1 or c.cs > 1:
                    ws.merge_cells(start_row=r0 + c.r, start_column=1 + c.c,
                                   end_row=r0 + c.r + c.rs - 1, end_column=1 + c.c + c.cs - 1)
                if c.cs == 1:
                    widths[c.c] = max(widths.get(c.c, 0), min(max((len(x) for x in c.text.split("\n")), default=0), 60))
            for r in range(table.nrows):
                for cc in range(table.ncols):
                    ws.cell(row=r0 + r, column=1 + cc).border = border
            row = r0 + table.nrows
            if after:
                row += 1
        put_text(after, row)
        for ci, w in widths.items():
            ws.column_dimensions[get_column_letter(ci + 1)].width = max(8, w + 3)
        if table is None or not widths:
            ws.column_dimensions["A"].width = 100
    wb.save(path)


# --------------------------------------------------------------------------- #
# Debug overlay
# --------------------------------------------------------------------------- #
def draw_debug(a: Analysis, path: str):
    img = cv2.cvtColor(a.norm, cv2.COLOR_GRAY2BGR)
    for t in a.tables:
        x0, y0, x1, y1 = t.bbox
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 160, 0), 3)
        cv2.putText(img, f"{t.kind} table {t.nrows}x{t.ncols}", (x0, max(y0 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 130, 0), 2)
    for p in a.paras:
        x0, y0, x1, y1 = p.bbox
        cv2.rectangle(img, (x0, y0), (x1, y1), (200, 80, 0), 2)
        cv2.putText(img, p.kind + (" bold" if p.bold else ""), (x0, max(y0 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 80, 0), 1)
    cv2.imwrite(path, img)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def convert(src, out_dir: str = ".", fmt: str = "auto", lang: str = "eng", debug: bool = False,
            name: Optional[str] = None) -> ConversionResult:
    """Convert one image to .docx/.xlsx.  fmt: auto | word | excel | both."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = name or (Path(str(src)).stem if isinstance(src, (str, os.PathLike)) else "output")
    a = analyze(src, lang)
    ext, layout, reason = decide_format(a)
    want = {"word": ["docx"], "excel": ["xlsx"], "both": ["docx", "xlsx"], "auto": [ext]}[fmt]
    if fmt != "auto":
        reason = f"forced by user ({fmt}); auto-detection said {ext} ({reason})"
    paths = {}
    for e in want:
        p = str(out / f"{stem}.{e}")
        (write_docx if e == "docx" else write_xlsx)(a.elements, p)
        paths[e] = p
    main = paths.get(ext) or next(iter(paths.values()))
    dbg = None
    if debug:
        dbg = str(out / f"{stem}_layout.png")
        draw_debug(a, dbg)
    return ConversionResult(str(src), main, Path(main).suffix[1:], layout, reason, len(a.tables),
                            len(a.paras), a.n_words, round(a.avg_conf, 1),
                            [p for p in paths.values() if p != main], dbg, a.elements)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert images to editable Word / Excel files.")
    ap.add_argument("images", nargs="+", help="image file(s): png, jpg, bmp, tiff, webp ...")
    ap.add_argument("-o", "--out", default="output", help="output directory (default: ./output)")
    ap.add_argument("-f", "--format", default="auto", choices=["auto", "word", "excel", "both"])
    ap.add_argument("-l", "--lang", default="eng", help="Tesseract language(s), e.g. eng or eng+hin")
    ap.add_argument("--debug", action="store_true", help="also save an annotated layout image")
    ap.add_argument("--json", action="store_true", help="print machine-readable summary")
    args = ap.parse_args(argv)
    rc = 0
    for img in args.images:
        try:
            r = convert(img, args.out, args.format, args.lang, args.debug)
        except Exception as e:                       # keep going on batch input
            print(f"[FAIL] {img}: {e}", file=sys.stderr)
            rc = 1
            continue
        if args.json:
            print(json.dumps(r.summary(), indent=2))
        else:
            print(f"[OK] {img}\n     -> {r.output}  ({r.format.upper()}, layout={r.layout}, "
                  f"{r.n_tables} table(s), {r.n_paragraphs} paragraph(s), "
                  f"confidence {r.avg_confidence}%)\n     reason: {r.reason}")
            for extra in r.extra_outputs:
                print(f"     -> {extra}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
