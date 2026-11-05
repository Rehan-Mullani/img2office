"""img2office - turn an image into an editable Word (.docx) or Excel (.xlsx) file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .engine import Analysis, Paragraph, Table, analyze
from .writers import write_docx, write_xlsx

__all__ = ["convert", "analyze", "ConvertResult", "Analysis", "Paragraph", "Table"]
__version__ = "1.0.0"


@dataclass
class ConvertResult:
    files: list  # absolute paths written
    kind: str  # 'word' | 'excel' | 'both'
    analysis: Analysis
    debug_path: str = ""


def convert(src, out_dir: str = ".", fmt: str = "auto", lang: str = "eng", debug: bool = False,
            name: str | None = None) -> ConvertResult:
    """
    Convert one image to Word and/or Excel.

    fmt: 'auto' (decide from content) | 'word' | 'excel' | 'both'
    """
    if fmt not in ("auto", "word", "excel", "both"):
        raise ValueError("fmt must be auto, word, excel or both")
    os.makedirs(out_dir, exist_ok=True)
    an = analyze(src, lang=lang, debug=debug)
    if name is None:
        name = os.path.splitext(os.path.basename(src))[0] if isinstance(src, (str, os.PathLike)) else "output"
    base = os.path.join(out_dir, name)
    kind = an.kind if fmt == "auto" else fmt
    files = []
    if kind in ("word", "both"):
        files.append(write_docx(an, base + ".docx"))
    if kind in ("excel", "both"):
        files.append(write_xlsx(an, base + ".xlsx"))
    dbg = ""
    if debug and an.debug_image is not None:
        import cv2
        dbg = base + "_debug.png"
        cv2.imwrite(dbg, an.debug_image)
    return ConvertResult([os.path.abspath(f) for f in files], kind, an, dbg)
