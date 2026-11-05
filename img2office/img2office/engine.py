"""
img2office.engine
=================
Image -> structured content (paragraphs + tables).

Pipeline
--------
1. load / clean the image  (EXIF rotation, transparency, dark background,
   uneven lighting, 90/180/270 rotation, small-angle de-skew, auto up-scaling)
2. detect *ruled* tables (grid lines)  -> every cell is OCR-ed separately,
   merged cells are recovered from missing borders
3. OCR the rest of the page (ruled tables are masked out)
4. detect *borderless* tables from aligned columns of words
5. rebuild paragraphs / headings / bullets / alignment from the remaining text
6. decide Word vs Excel from what was found
"""
from __future__ import annotations

import math
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Optional, Union

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

TARGET_LH = 56.0  # tesseract is most accurate when a text line is ~50-60 px tall
BOLD_T = 1.2  # relative stroke weight that counts as bold
PAD = 30  # white border added around the page (tesseract dislikes text touching the edge)

# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #


@dataclass
class Word:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int
    block: int = 0
    par: int = 0
    line: int = 0

    @property
    def x1(self):
        return self.x + self.w

    @property
    def y1(self):
        return self.y + self.h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


@dataclass
class TextLine:
    words: list
    block: int = 0

    def __post_init__(self):
        self.words = sorted(self.words, key=lambda w: w.x)
        self.x0 = min(w.x for w in self.words)
        self.y0 = min(w.y for w in self.words)
        self.x1 = max(w.x1 for w in self.words)
        self.y1 = max(w.y1 for w in self.words)
        self.baseline = statistics.median(w.y1 for w in self.words)
        hs = sorted(w.h for w in self.words)
        self.cap_h = hs[min(len(hs) - 1, int(0.75 * len(hs)))]
        self.bold_rel = 1.0

    @property
    def text(self):
        return " ".join(w.text for w in self.words)


@dataclass
class Paragraph:
    lines: list
    text: str = ""
    kind: str = "text"  # text | heading | bullet
    bold: bool = False
    size_ratio: float = 1.0
    heading_level: int = 0
    align: str = "left"  # left | center | right | justify
    indent_pt: float = 0.0
    conf: float = 0.0
    bbox: tuple = (0, 0, 0, 0)


@dataclass
class Table:
    cells: list  # list[list[str]]
    bbox: tuple
    ruled: bool
    merges: list = field(default_factory=list)  # (r0, c0, r1, c1) inclusive
    col_widths_px: list = field(default_factory=list)
    conf: float = 0.0
    low_conf: list = field(default_factory=list)  # [(r, c)]

    @property
    def nrows(self):
        return len(self.cells)

    @property
    def ncols(self):
        return len(self.cells[0]) if self.cells else 0

    def word_count(self):
        return sum(len(c.split()) for row in self.cells for c in row)


@dataclass
class Analysis:
    elements: list  # Paragraph | Table in reading order
    kind: str  # 'word' | 'excel'
    reason: str
    scale: float
    mean_conf: float
    warnings: list
    size: tuple  # (w, h) of the processed page
    debug_image: Optional[np.ndarray] = None

    @property
    def paragraphs(self):
        return [e for e in self.elements if isinstance(e, Paragraph)]

    @property
    def tables(self):
        return [e for e in self.elements if isinstance(e, Table)]


# --------------------------------------------------------------------------- #
# image loading / cleaning
# --------------------------------------------------------------------------- #


def load_image(src) -> np.ndarray:
    """Load path / PIL image / ndarray as a BGR uint8 array."""
    if isinstance(src, np.ndarray):
        arr = src
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr[:, :, :3].copy()
    im = src if isinstance(src, Image.Image) else Image.open(src)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def noise_sigma(gray: np.ndarray) -> float:
    """Robust noise estimate (MAD of the Laplacian); ~0 for clean screenshots."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.median(np.abs(lap)) / 0.6745)


def prep_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sg = noise_sigma(gray)
    if sg > 2.0:  # noisy photo / scan / heavy JPEG
        gray = cv2.fastNlMeansDenoising(gray, None, h=float(min(12, 2 + 0.8 * sg)), templateWindowSize=7, searchWindowSize=21)
    # light text on dark background -> invert
    if np.median(gray) < 110:
        gray = 255 - gray
    # uneven illumination (photos / scans): divide by estimated background
    small = cv2.resize(gray, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA)
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    bg = cv2.medianBlur(bg, 11)
    if np.percentile(bg, 95) - np.percentile(bg, 5) > 45:
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
        gray = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    return gray


def _skew_score(bw, angle):
    h, w = bw.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    r = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST)
    prof = r.sum(axis=1, dtype=np.float64)
    return float(np.var(prof))


def deskew(gray: np.ndarray):
    """Projection-profile de-skew; returns (image, angle)."""
    h, w = gray.shape
    f = 900.0 / max(h, w)
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if cv2.countNonZero(bw) < 50:
        return gray, 0.0
    coarse = max(np.arange(-10, 10.01, 1.0), key=lambda a: _skew_score(bw, a))
    fine = max(np.arange(coarse - 1, coarse + 1.01, 0.1), key=lambda a: _skew_score(bw, a))
    base = _skew_score(bw, 0.0)
    best = _skew_score(bw, fine)
    if abs(fine) < 0.3 or best < base * 1.08:
        return gray, 0.0
    m = cv2.getRotationMatrix2D((w / 2, h / 2), fine, 1.0)
    out = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return out, float(fine)


def _ocr_score(gray, lang):
    try:
        ws = ocr_words(gray, lang)
    except Exception:
        return 0.0
    return sum(w.conf for w in ws if w.conf > 50 and len(w.text) >= 2)


def fix_orientation(gray: np.ndarray, lang: str):
    """Rotate by 90/180/270 when tesseract OSD says so *and* OCR confirms it."""
    try:
        osd = pytesseract.image_to_osd(
            Image.fromarray(gray), config="--psm 0 -c min_characters_to_try=15", output_type=Output.DICT
        )
        rot, conf = int(osd.get("rotate", 0)), float(osd.get("orientation_conf", 0))
    except Exception:
        return gray, 0
    if rot not in (90, 180, 270) or conf < 2.0:
        return gray, 0
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot]
    cand = cv2.rotate(gray, code)
    if _ocr_score(cand, lang) > 1.15 * _ocr_score(gray, lang):
        return cand, rot
    return gray, 0


# --------------------------------------------------------------------------- #
# OCR helpers
# --------------------------------------------------------------------------- #

_LIG = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}


def clean_text(s: str) -> str:
    for k, v in _LIG.items():
        s = s.replace(k, v)
    s = s.replace(" ", " ").replace("​", "").replace("\x0c", "")
    return re.sub(r"[ \t]+", " ", s).strip()


def ocr_words(gray, lang="eng", psm=3) -> list:
    cfg = f"--oem 1 --psm {psm} --dpi 300 -c preserve_interword_spaces=1"
    d = pytesseract.image_to_data(Image.fromarray(gray), lang=lang, config=cfg, output_type=Output.DICT)
    out = []
    for i, t in enumerate(d["text"]):
        t = clean_text(t)
        try:
            conf = float(d["conf"][i])
        except ValueError:
            continue
        if not t or conf < 0:
            continue
        if re.fullmatch(r"[|\\/_~`^]+", t):  # ruling-line noise
            continue
        if conf < 10 and len(t) <= 2:
            continue
        out.append(
            Word(t, conf, int(d["left"][i]), int(d["top"][i]), int(d["width"][i]), int(d["height"][i]),
                 int(d["block_num"][i]), int(d["par_num"][i]), int(d["line_num"][i]))
        )
    return out


def group_tess_lines(words) -> list:
    groups = {}
    for w in words:
        groups.setdefault((w.block, w.par, w.line), []).append(w)
    return [TextLine(g, block=k[0]) for k, g in sorted(groups.items())]


def median_line_height(words) -> float:
    lines = group_tess_lines(words)
    hs = [l.y1 - l.y0 for l in lines if len(l.words) >= 1]
    return float(statistics.median(hs)) if hs else 0.0


# --------------------------------------------------------------------------- #
# ruled tables
# --------------------------------------------------------------------------- #


def _cluster(vals, tol):
    vals = sorted(vals)
    if not vals:
        return []
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(round(float(np.mean(g)))) for g in groups]


def _drop_close(vals, min_gap):
    vals = sorted(vals)
    out = []
    for v in vals:
        if out and v - out[-1] < min_gap:
            out[-1] = int(round((out[-1] + v) / 2))
        else:
            out.append(v)
    return out


def line_masks(gray, lh):
    """Horizontal / vertical rule masks (thin dark/light lines and edges of filled bars)."""
    tmax = max(5, int(0.2 * lh))
    min_len = max(int(1.8 * lh), 24)  # shorter edges are letter shapes, not rules
    k = 2 * tmax + 1
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    ink = (bh > 20).astype(np.uint8) * 255
    sm = cv2.GaussianBlur(gray, (3, 3), 0)
    gy = np.abs(cv2.Sobel(sm, cv2.CV_16S, 0, 1, ksize=3)) > 40
    gx = np.abs(cv2.Sobel(sm, cv2.CV_16S, 1, 0, ksize=3)) > 40
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    h1 = cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk)
    v1 = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)
    h2 = cv2.morphologyEx(gy.astype(np.uint8) * 255, cv2.MORPH_OPEN, hk)
    v2 = cv2.morphologyEx(gx.astype(np.uint8) * 255, cv2.MORPH_OPEN, vk)
    return cv2.bitwise_or(h1, h2), cv2.bitwise_or(v1, v2), tmax


def _border(mask, horizontal, pos, a, b, tol):
    H, W = mask.shape
    a2, b2 = a + tol, b - tol
    if b2 - a2 < 3:
        a2, b2 = a, b
    p0, p1 = max(0, pos - tol), min((H if horizontal else W), pos + tol + 1)
    if horizontal:
        seg = mask[p0:p1, max(0, a2):min(W, b2)]
        if seg.size == 0:
            return False
        return seg.any(axis=0).mean() >= 0.5
    seg = mask[max(0, a2):min(H, b2), p0:p1]
    if seg.size == 0:
        return False
    return seg.any(axis=1).mean() >= 0.5


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def ocr_cell(gray, lines_mask, rect, lh, lang, local_mask=False):
    """OCR one cell. Returns (text, conf)."""
    x0, y0, x1, y1 = rect
    crop = gray[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return "", 100.0, []
    lm = lines_mask if local_mask else lines_mask[y0:y1, x0:x1]
    if np.median(crop) < 110:  # white text on dark fill
        crop = 255 - crop
    crop[lm > 0] = 255
    # blank cell?
    otsu_t, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = (crop < min(otsu_t, 200)).sum() if crop.std() > 12 else 0
    if ink < max(8, 0.0008 * crop.size):
        return "", 100.0, []
    crop = cv2.copyMakeBorder(crop, 14, 14, 14, 14, cv2.BORDER_CONSTANT, value=255)
    single = (y1 - y0) < 1.9 * lh
    for psm in ((7, 6, 10) if single else (6, 7, 10)):
        cfg = f"--oem 1 --psm {psm} --dpi 300 -c preserve_interword_spaces=1"
        d = pytesseract.image_to_data(Image.fromarray(crop), lang=lang, config=cfg, output_type=Output.DICT)
        ws = []
        for i, t in enumerate(d["text"]):
            t = clean_text(t)
            try:
                c = float(d["conf"][i])
            except ValueError:
                continue
            if t and c >= 0 and not re.fullmatch(r"[|\\/_~`^]+", t):
                ws.append(Word(t, c, int(d["left"][i]), int(d["top"][i]), int(d["width"][i]), int(d["height"][i]),
                               int(d["block_num"][i]), int(d["par_num"][i]), int(d["line_num"][i])))
        if ws:
            abs_ws = [Word(w.text, w.conf, w.x - 14 + x0, w.y - 14 + y0, w.w, w.h, w.block, w.par, w.line) for w in ws]
            wl = [max(len(w.text), 1) for w in ws]  # length-weighted: a lone low-confidence "A" is not a problem
            return _join_cell_lines(ws), float(np.average([w.conf for w in ws], weights=wl)), abs_ws
    return "", 40.0, []


def _split_colspan(ws, xs, c0, c1, single_row, lh, tol):
    """If a merged (colspan) cell contains widely separated word groups that line up with grid
    columns, return [(col_a, col_b, words)] else None."""
    if c1 <= c0 or not single_row or len(ws) < 2:
        return None
    ws = sorted(ws, key=lambda w: w.x)
    groups = [[ws[0]]]
    for a, b in zip(ws, ws[1:]):
        if b.x - a.x1 >= 0.9 * lh:
            groups.append([b])
        else:
            groups[-1].append(b)
    if len(groups) < 2:
        return None
    cuts = []
    for ga, gb in zip(groups, groups[1:]):
        lo, hi = max(w.x1 for w in ga), min(w.x for w in gb)
        cand = [c for c in range(c0 + 1, c1 + 1) if lo - tol <= xs[c] <= hi + tol]
        if not cand:
            return None  # the gap does not coincide with a column boundary -> genuine merged cell
        mid = (lo + hi) / 2
        cuts.append(min(cand, key=lambda c: abs(xs[c] - mid)))
    bounds = [c0] + cuts + [c1 + 1]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        left, right = xs[a], xs[b]
        out.append((a, b - 1, [w for w in ws if left <= w.cx < right]))
    return out


def _join_cell_lines(ws) -> str:
    lines = group_tess_lines(ws)
    lines.sort(key=lambda l: l.y0)
    if len(lines) == 1:
        return lines[0].text
    right = max(l.x1 for l in lines)
    left = min(l.x0 for l in lines)
    out = lines[0].text
    for prev, cur in zip(lines, lines[1:]):
        full = (prev.x1 - left) >= 0.85 * (right - left)  # wrapped (full) line -> join with space
        if out.endswith("-") and full and cur.text[:1].islower():
            out = out[:-1] + cur.text
        else:
            out += (" " if full else "\n") + cur.text
    return out


def detect_ruled_tables(gray, lh, lang):
    H, W = gray.shape
    hm, vm, tmax = line_masks(gray, lh)
    tol = max(3, tmax // 2 + 2)
    grid = cv2.dilate(cv2.bitwise_or(hm, vm), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tol + 1, 2 * tol + 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    hn, hlab, hst, _ = cv2.connectedComponentsWithStats(hm, connectivity=8)
    vn, vlab, vst, _ = cv2.connectedComponentsWithStats(vm, connectivity=8)
    rule_mask = np.zeros_like(hm)  # only segments accepted as real rules (never letter edges)
    tables = []
    for i in range(1, n):
        bx, by, bw, bh, _ = stats[i]
        if bw < 2 * lh or bh < 1.4 * lh:
            continue
        ys, xs = [], []
        for j in range(1, hn):
            x, y, w, h, a = hst[j]
            if w >= 3.5 * lh and bx - tol <= x and x + w <= bx + bw + tol and by - tol <= y and y + h <= by + bh + tol:
                ys.append(y + h // 2)
                rule_mask[y:y + h, x:x + w] = 255
        for j in range(1, vn):
            x, y, w, h, a = vst[j]
            if bx - tol <= x and x + w <= bx + bw + tol and by - tol <= y and y + h <= by + bh + tol:
                # a vertical rule must start or end on a horizontal rule (letter stems never do)
                if any(abs(y - yy) <= 3 * tol or abs(y + h - yy) <= 3 * tol for yy in ys):
                    xs.append(x + w // 2)
                    rule_mask[y:y + h, x:x + w] = 255
        gap = max(6, int(0.45 * lh))
        ys = _drop_close(_cluster(ys, tol), gap)
        xs = _drop_close(_cluster(xs, tol), gap)
        if len(ys) < 2 or len(xs) < 2:
            continue
        tbl = _build_grid_table(gray, hm, vm, None, xs, ys, tol, lh, lang, rule_mask)
        if tbl is not None:
            tables.append(tbl)
    tables.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
    return tables


def _build_grid_table(gray, hm, vm, _unused, xs, ys, tol, lh, lang, rule_mask):
    R, C = len(ys) - 1, len(xs) - 1
    if R * C < 2:
        return None
    uf = _UF(R * C)
    idx = lambda r, c: r * C + c
    for r in range(R):
        for c in range(C):
            if c + 1 < C and not _border(vm, False, xs[c + 1], ys[r], ys[r + 1], tol):
                uf.union(idx(r, c), idx(r, c + 1))
            if r + 1 < R and not _border(hm, True, ys[r + 1], xs[c], xs[c + 1], tol):
                uf.union(idx(r, c), idx(r + 1, c))
    groups = {}
    for r in range(R):
        for c in range(C):
            groups.setdefault(uf.find(idx(r, c)), []).append((r, c))
    # non-rectangular groups cannot be merged cells -> split them again
    rects = []
    for cells in groups.values():
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
        if (r1 - r0 + 1) * (c1 - c0 + 1) == len(cells):
            rects.append((r0, c0, r1, c1))
        else:
            rects.extend((r, c, r, c) for r, c in cells)
    if len(rects) < 2:
        return None
    text = [[""] * C for _ in range(R)]
    confs, low = [], []
    inset = tol + 1
    final = []  # (r0, c0, r1, c1)
    for (r0, c0, r1, c1) in rects:
        rect = (xs[c0] + inset, ys[r0] + inset, xs[c1 + 1] - inset, ys[r1 + 1] - inset)
        loc = cv2.dilate(rule_mask[rect[1]:rect[3], rect[0]:rect[2]], np.ones((3, 3), np.uint8))
        t, cf, ws = ocr_cell(gray, loc, rect, lh, lang, local_mask=True)
        parts = _split_colspan(ws, xs, c0, c1, r0 == r1, lh, tol)
        if parts is None:
            final.append((r0, c0, r1, c1))
            text[r0][c0] = t
            if t:
                confs.append(cf)
                if cf < 60:
                    low.append((r0, c0))
        else:  # a faint column border was missed: the "merged" cell really holds several cells
            for (a, b, words_in) in parts:
                final.append((r0, a, r1, b))
                s = " ".join(w.text for w in sorted(words_in, key=lambda w: w.x))
                text[r0][a] = s
                if s:
                    confs.append(cf)
    merges = [m for m in final if m[0] != m[2] or m[1] != m[3]]
    tbl = Table(text, (xs[0], ys[0], xs[-1], ys[-1]), True, merges,
                [xs[i + 1] - xs[i] for i in range(C)], float(np.mean(confs)) if confs else 0.0, low)
    return _trim_table(tbl)


def _trim_table(t: Table) -> Optional[Table]:
    """Drop completely empty outer rows / columns (keeps merged ranges consistent)."""
    cells, merges = t.cells, list(t.merges)
    covered = set()
    for r0, c0, r1, c1 in merges:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                covered.add((r, c))
    R, C = len(cells), len(cells[0])

    def row_empty(r):
        return all(not cells[r][c] for c in range(C))

    def col_empty(c):
        return all(not cells[r][c] for r in range(R))

    top = 0
    while top < R and row_empty(top) and not any(m[0] <= top <= m[2] for m in merges):
        top += 1
    bot = R - 1
    while bot > top and row_empty(bot) and not any(m[0] <= bot <= m[2] for m in merges):
        bot -= 1
    left = 0
    while left < C and col_empty(left) and not any(m[1] <= left <= m[3] for m in merges):
        left += 1
    right = C - 1
    while right > left and col_empty(right) and not any(m[1] <= right <= m[3] for m in merges):
        right -= 1
    if top > bot or left > right:
        return None
    t.cells = [row[left:right + 1] for row in cells[top:bot + 1]]
    t.merges = [(a - top, b - left, c - top, d - left) for a, b, c, d in merges
                if a >= top and c <= bot and b >= left and d <= right]
    t.col_widths_px = t.col_widths_px[left:right + 1]
    t.low_conf = [(r - top, c - left) for r, c in t.low_conf if top <= r <= bot and left <= c <= right]
    if t.word_count() == 0:
        return None
    return t


# --------------------------------------------------------------------------- #
# borderless tables (aligned columns of words)
# --------------------------------------------------------------------------- #


def _geo_lines(words):
    lines = []
    for w in sorted(words, key=lambda w: w.cy):
        placed = False
        for ln in reversed(lines[-6:]):
            top, bot = min(x.y for x in ln), max(x.y1 for x in ln)
            ov = min(bot, w.y1) - max(top, w.y)
            if ov >= 0.55 * min(w.h, bot - top):
                ln.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    for ln in lines:
        ln.sort(key=lambda w: w.x)
    lines.sort(key=lambda ln: statistics.median(w.cy for w in ln))
    return lines


def _segments(line, gap_thr):
    segs = [[line[0]]]
    for a, b in zip(line, line[1:]):
        if b.x - a.x1 > gap_thr:
            segs.append([b])
        else:
            segs[-1].append(b)
    return segs


def detect_borderless_tables(words, lh, gray=None, lang="eng"):
    if not words:
        return [], set()
    lines = _geo_lines(words)
    gap_thr = 1.1 * lh
    segs = [_segments(l, gap_thr) for l in lines]
    multi = [len(s) >= 2 for s in segs]
    cy = [statistics.median(w.cy for w in l) for l in lines]
    ok = lambda a, b: (cy[b] - cy[a]) <= 3.0 * lh
    n = len(lines)
    runs, i = [], 0
    while i < n:
        if not multi[i]:
            i += 1
            continue
        j = i
        while True:
            k = j + 1
            if k < n and multi[k] and ok(j, k):
                j = k
            elif k + 1 < n and not multi[k] and multi[k + 1] and ok(j, k) and ok(k, k + 1) and len(lines[k]) <= 6:
                j = k + 1
            else:
                break
        runs.append((i, j))
        i = j + 1
    tables, used = [], set()
    for a, b in runs:
        rows = list(range(a, b + 1))
        mrows = [r for r in rows if multi[r]]
        if len(mrows) < 3:
            continue
        iv = sorted((w.x, w.x1) for r in mrows for w in lines[r])
        cols = [list(iv[0])]
        for x0, x1 in iv[1:]:
            if x0 - cols[-1][1] > gap_thr:
                cols.append([x0, x1])
            else:
                cols[-1][1] = max(cols[-1][1], x1)
        if len(cols) < 2:
            continue
        grid = []
        for r in rows:
            cells = [[] for _ in cols]
            for w in lines[r]:
                k = next((ci for ci, (x0, x1) in enumerate(cols) if x0 - 1 <= w.cx <= x1 + 1), None)
                if k is None:
                    k = min(range(len(cols)), key=lambda ci: abs(cols[ci][0] - w.x))
                cells[k].append(w)
            grid.append(cells)
        texts = [[" ".join(w.text for w in sorted(c, key=lambda w: w.x)) for c in row] for row in grid]
        if gray is not None:
            texts = _verify_cells(gray, texts, grid, cols, rows, lines, lh, lang, gap_thr)
        filled_rows = sum(1 for row in texts if sum(1 for c in row if c) >= 2)
        if filled_rows < max(3, int(0.6 * len(texts))):
            continue
        wpc = [len(c.split()) for row in texts for c in row if c]
        if statistics.median(wpc) >= 6:  # prose in columns (newspaper layout), not a table
            continue
        allw = [w for r in rows for w in lines[r]]
        bbox = (min(w.x for w in allw), min(w.y for w in allw), max(w.x1 for w in allw), max(w.y1 for w in allw))
        tb = Table(texts, bbox, False, [], [c[1] - c[0] for c in cols],
                   float(np.mean([w.conf for w in allw])))
        tb = _trim_table(tb)
        if tb is None:
            continue
        tables.append(tb)
        used.update(id(w) for w in allw)
    return tables, used


def _verify_cells(gray, texts, grid, cols, rows, lines, lh, lang, gap_thr):
    """Second opinion: re-read every cell of a borderless table from a tight crop.
    Fixes dropped single characters / digits that whole-page OCR tends to lose."""
    H, W = gray.shape
    zero = np.zeros_like(gray)
    bands = []
    for r in rows:
        bands.append((min(w.y for w in lines[r]), max(w.y1 for w in lines[r])))
    pad = int(0.2 * lh)
    bounds = []
    for k, (x0, x1) in enumerate(cols):
        left = (cols[k - 1][1] + x0) / 2 if k else x0 - gap_thr / 2
        right = (x1 + cols[k + 1][0]) / 2 if k + 1 < len(cols) else x1 + gap_thr / 2
        bounds.append((int(max(0, left)), int(min(W, right))))
    out = [row[:] for row in texts]
    for ri, (ya, yb) in enumerate(bands):
        up = min(pad, (ya - bands[ri - 1][1]) // 2) if ri else pad
        dn = min(pad, (bands[ri + 1][0] - yb) // 2) if ri + 1 < len(bands) else pad
        y0, y1 = max(0, ya - max(up, 1)), min(H, yb + max(dn, 1))
        for ci, (xa, xb) in enumerate(bounds):
            page_ws = grid[ri][ci]
            page_conf = float(np.mean([w.conf for w in page_ws])) if page_ws else -1.0
            t, cf, _ = ocr_cell(gray, zero, (xa, y0, xb, y1), lh, lang)
            t = t.replace("\n", " ")
            if not t or t == texts[ri][ci]:
                continue
            if not texts[ri][ci] or cf > page_conf + 3:
                out[ri][ci] = t
    return out


# --------------------------------------------------------------------------- #
# paragraphs
# --------------------------------------------------------------------------- #

_BULLET_CHARS = set("•·●○▪■◦‣∙*►▶–—-»«")
_ENUM_RE = re.compile(r"^(\(?\d{1,3}[.)]|\(?[A-Za-z][.)])\s+\S")


def _is_bullet_line(line: TextLine, lh: float) -> bool:
    """First token is a bullet glyph (real or mis-read, e.g. 'e', 'o', '«') followed by a hanging gap."""
    if len(line.words) < 2:
        return False
    a, b = line.words[0], line.words[1]
    t = a.text
    if len(t) != 1 or t.isdigit() or t in "aAI":
        return False
    gap = b.x - a.x1
    if t in _BULLET_CHARS:
        return gap > 0.3 * lh or t not in "-–—*»«"
    if not t.isalnum():
        return gap > 0.35 * lh
    return gap > 0.55 * lh  # a lone letter (usually a mis-read round bullet) followed by a wide gap


def _stroke_width(gray, line: TextLine):
    pad = 2
    crop = gray[max(0, line.y0 - pad):line.y1 + pad, max(0, line.x0 - pad):line.x1 + pad]
    if crop.size == 0 or crop.std() < 8:
        return 0.0
    _, b = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    area = cv2.countNonZero(b)
    cnts, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    return 2.0 * area / max(per, 1.0)


def _multi_column(lines, lh):
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            ov = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ov > 0.5 * min(a.y1 - a.y0, b.y1 - b.y0) and (b.x0 - a.x1 > 2 * lh or a.x0 - b.x1 > 2 * lh):
                return True
    return False


def build_paragraphs(words, gray, lh, ref_cap=None):
    img_w = gray.shape[1]
    lines = [l for l in group_tess_lines(words) if l.words]
    if not lines:
        return []
    # relative stroke weight (bold detection)
    nw = []
    for l in lines:
        sw = _stroke_width(gray, l)
        hs = statistics.median(w.h for w in l.words)
        nw.append(sw / max(hs, 1))
    ref_nw = statistics.median(nw) or 1.0
    for l, v in zip(lines, nw):
        l.bold_rel = v / ref_nw if ref_nw else 1.0
    ref_cap = ref_cap or statistics.median(l.cap_h for l in lines)

    # --- split lines into paragraphs inside each tesseract block
    multicol = _multi_column(lines, lh)
    by_block = {}
    for l in lines:
        # tesseract often cuts one paragraph into several blocks, so on single-column pages
        # everything is one stream ordered top-to-bottom; only real multi-column pages keep its blocks
        by_block.setdefault(l.block if multicol else 0, []).append(l)
    pitches = []
    for bl in by_block.values():
        bl.sort(key=lambda l: l.baseline)
        pitches += [b.baseline - a.baseline for a, b in zip(bl, bl[1:]) if 0 < b.baseline - a.baseline < 3.2 * lh]
    med_pitch = statistics.median(pitches) if pitches else 1.4 * lh
    paras_lines = []
    for blk in sorted(by_block):
        bl = by_block[blk]
        bx0 = min(l.x0 for l in bl)
        bx1 = max(l.x1 for l in bl)
        cur = [bl[0]]
        for prev, nxt in zip(bl, bl[1:]):
            pitch = nxt.baseline - prev.baseline
            new = pitch > 1.3 * med_pitch
            new |= nxt.cap_h > 1.3 * prev.cap_h or prev.cap_h > 1.3 * nxt.cap_h
            new |= abs(nxt.bold_rel - prev.bold_rel) > BOLD_T - 1 and (len(prev.words) <= 12 or len(nxt.words) <= 12) \
                and (nxt.bold_rel > BOLD_T or prev.bold_rel > BOLD_T)
            short = (prev.x1 - bx0) < 0.62 * (bx1 - bx0) and prev.text.rstrip()[-1:] in ".!?:;" and len(bl) > 2
            if short:  # short line ending a sentence = end of paragraph
                new = True
            if _ENUM_RE.match(nxt.text) or _is_bullet_line(nxt, lh):
                new = True
            if abs(nxt.x0 - prev.x0) > 2.5 * lh:
                new = True
            if (nxt.x0 - bx0) > 1.2 * lh and abs(prev.x0 - bx0) < 0.4 * lh and len(nxt.words) > 2:
                new = True  # first-line indent
            if new:
                paras_lines.append(cur)
                cur = [nxt]
            else:
                cur.append(nxt)
        paras_lines.append(cur)

    all_x0 = [l.x0 for l in lines]
    all_x1 = [l.x1 for l in lines]
    page_x0, page_x1 = min(all_x0), max(all_x1)
    page_w = max(page_x1 - page_x0, 1)
    px_per_pt = max(lh / 14.0, 0.5)

    paras = []
    for pl in paras_lines:
        text = _join_lines([l.text for l in pl])
        kind = "text"
        if _is_bullet_line(pl[0], lh):
            kind = "bullet"
            text = text[len(pl[0].words[0].text):].strip()
        ws = [w for l in pl for w in l.words]
        conf = float(np.mean([w.conf for w in ws]))
        bold_rel = float(np.mean([l.bold_rel for l in pl]))
        size = float(np.mean([l.cap_h for l in pl])) / ref_cap
        bbox = (min(l.x0 for l in pl), min(l.y0 for l in pl), max(l.x1 for l in pl), max(l.y1 for l in pl))
        nwords = len(text.split())
        p = Paragraph(pl, text, kind, bool(bold_rel >= BOLD_T), round(size, 2), 0, "left", 0.0, conf, bbox)
        if len(pl) <= 2 and nwords <= 14 and size >= 1.25 and kind == "text":
            p.kind = "heading"
            p.heading_level = 1 if size >= 1.7 else (2 if size >= 1.4 else 3)
            p.bold = True if p.bold or size >= 1.25 else p.bold
        # alignment
        if not multicol:
            left_gap = bbox[0] - page_x0
            right_gap = page_x1 - bbox[2]
            cx_off = abs((bbox[0] + bbox[2]) / 2 - (page_x0 + page_x1) / 2)
            width = bbox[2] - bbox[0]
            img_off = abs((bbox[0] + bbox[2]) / 2 - img_w / 2)
            if left_gap > 0.08 * page_w and right_gap > 0.08 * page_w and cx_off < 0.025 * page_w:
                p.align = "center"
            elif width < 0.8 * img_w and img_off < 0.02 * img_w and bbox[0] > 0.1 * img_w:
                p.align = "center"  # centred on the page even if the other text is not
            elif right_gap < 0.02 * page_w and left_gap > 0.2 * page_w:
                p.align = "right"
            elif len(pl) >= 3:
                inner = pl[:-1]
                flush = sum(1 for l in inner if page_x1 - l.x1 <= 0.012 * page_w)
                if flush >= 0.8 * len(inner) and bbox[0] - page_x0 < 0.03 * page_w:
                    p.align = "justify"
        # indent
        dx = bbox[0] - page_x0
        if p.align == "left" and dx > 1.5 * lh and not multicol:
            p.indent_pt = round(dx / px_per_pt, 1)
        paras.append(p)
    return paras


def _join_lines(parts):
    out = parts[0]
    for nxt in parts[1:]:
        if out.endswith("-") and nxt[:1].islower() and len(out) > 2 and out[-2].isalpha():
            out = out[:-1] + nxt
        else:
            out += " " + nxt
    return out


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #


def analyze(src, lang: str = "eng", debug: bool = False) -> Analysis:
    warnings = []
    bgr = load_image(src)
    gray = prep_gray(bgr)
    gray, rot = fix_orientation(gray, lang)
    if rot:
        warnings.append(f"Image was rotated by {rot}° automatically.")
    gray, ang = deskew(gray)
    if ang:
        warnings.append(f"Image was straightened by {ang:.1f}°.")
    gray = cv2.copyMakeBorder(gray, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=255)

    # first pass -> text height -> best scale for OCR
    words = ocr_words(gray, lang)
    lh0 = median_line_height(words)
    if lh0 <= 0:
        lh0 = 22.0
    scale = float(np.clip(TARGET_LH / lh0, 0.5, 4.0))
    if 0.85 <= scale <= 1.15:
        scale = 1.0
    scale = min(scale, 6000.0 / max(gray.shape))
    scale = max(scale, 0.3)
    if scale != 1.0:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
        words = ocr_words(gray, lang)
    lh = median_line_height(words) or TARGET_LH

    all_lines = [l for l in group_tess_lines(words) if l.words]
    ref_cap = statistics.median(l.cap_h for l in all_lines) if all_lines else None

    # ---- ruled tables
    ruled = detect_ruled_tables(gray, lh, lang)
    if ruled:
        masked = gray.copy()
        for t in ruled:
            x0, y0, x1, y1 = t.bbox
            masked[max(0, y0 - 4):y1 + 5, max(0, x0 - 4):x1 + 5] = 255
        words = ocr_words(masked, lang)

    # ---- borderless tables
    free, used = detect_borderless_tables(words, lh, gray, lang)
    rest = [w for w in words if id(w) not in used]
    paragraphs = build_paragraphs(rest, gray, lh, ref_cap) if rest else []

    tables = _fix_numeric_columns(ruled + free)
    # ---- reading order: insert each table before the first paragraph that starts below it
    elements = list(paragraphs)
    for t in sorted(tables, key=lambda t: t.bbox[1], reverse=True):
        pos = next((i for i, p in enumerate(elements)
                    if isinstance(p, Paragraph) and p.bbox[1] >= t.bbox[1] - 0.5 * lh), len(elements))
        elements.insert(pos, t)

    # ---- word or excel?
    pw = sum(len(p.text.split()) for p in paragraphs)
    tw = sum(t.word_count() for t in tables)
    if not tables:
        kind, reason = "word", "Only paragraph text was found."
    elif pw == 0:
        kind, reason = "excel", "The image contains only table content."
    else:
        ratio = tw / (tw + pw)
        if ratio >= 0.7:
            kind = "excel"
            reason = f"Mostly tabular content ({ratio:.0%} of the words are inside tables)."
        else:
            kind = "word"
            reason = f"Mixed content (only {ratio:.0%} table words) - paragraphs plus real tables go to Word."

    confs = [w.conf for p in paragraphs for l in p.lines for w in l.words]
    confs += [t.conf for t in tables for _ in range(max(1, t.word_count()))]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    if not elements:
        warnings.append("No text could be recognised in this image.")
    elif mean_conf < 70:
        warnings.append(f"Average OCR confidence is low ({mean_conf:.0f}%). Please check the result carefully.")
    for ti, t in enumerate(tables, 1):
        if t.low_conf:
            warnings.append(f"Table {ti}: {len(t.low_conf)} cell(s) were hard to read - please verify them.")

    dbg = _draw_debug(gray, elements) if debug else None
    return Analysis(elements, kind, reason, scale, mean_conf, warnings, (gray.shape[1], gray.shape[0]), dbg)


# --------------------------------------------------------------------------- #
# numeric clean-up
# --------------------------------------------------------------------------- #

_NUM_RE = re.compile(r"^[\(\-+−]?[$€£₹¥]?\s?\d[\d,]*(\.\d+)?\s?%?\)?$")
_CONFUSE = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8", "Z": "2"})


def _fix_numeric_columns(tables):
    """In columns that are clearly numeric, repair classic OCR mix-ups (O/0, l/1, S/5 ...)."""
    for t in tables:
        R, C = t.nrows, t.ncols
        for c in range(C):
            col = [t.cells[r][c] for r in range(R) if t.cells[r][c]]
            if len(col) < 3:
                continue
            numeric = [x for x in col if _NUM_RE.match(x.replace(" ", ""))]
            if len(numeric) < 0.6 * len(col):
                continue
            for r in range(R):
                s = t.cells[r][c]
                if not s or _NUM_RE.match(s.replace(" ", "")):
                    continue
                if len(s) < 2 or not any(ch.isdigit() for ch in s):
                    continue
                fixed = s.translate(_CONFUSE)
                changed = sum(1 for a, b in zip(s, fixed) if a != b)
                if changed <= 2 and _NUM_RE.match(fixed.replace(" ", "")):
                    t.cells[r][c] = fixed
    return tables


# --------------------------------------------------------------------------- #
# debug overlay
# --------------------------------------------------------------------------- #


def _draw_debug(gray, elements):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for e in elements:
        x0, y0, x1, y1 = e.bbox
        if isinstance(e, Table):
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 140, 255), 3)
            cv2.putText(img, "TABLE" + (" (ruled)" if e.ruled else " (no borders)"), (x0, max(20, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), (200, 120, 0), 2)
            cv2.putText(img, p_label(e), (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 120, 0), 2)
    return img


def p_label(p: Paragraph):
    return f"{p.kind}{' bold' if p.bold else ''} {p.align}"
