"""Command line:  python -m img2office  photo.png  [-o out_dir] [--format auto|word|excel|both]"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from . import convert

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="img2office", description="Image -> editable Word / Excel file")
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("-o", "--out", default="output", help="output folder (default: ./output)")
    ap.add_argument("-f", "--format", default="auto", choices=["auto", "word", "excel", "both"],
                    help="auto = decide from the image content (default)")
    ap.add_argument("-l", "--lang", default="eng", help="tesseract language(s), e.g. eng or eng+hin")
    ap.add_argument("--debug", action="store_true", help="also save an image showing what was detected")
    a = ap.parse_args(argv)

    files = []
    for p in a.inputs:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(EXTS))
        else:
            files.append(p)
    if not files:
        print("No images found.", file=sys.stderr)
        return 2
    rc = 0
    for f in files:
        try:
            res = convert(f, a.out, a.format, a.lang, a.debug)
        except Exception as e:  # keep batch going
            print(f"[FAIL] {f}: {e}", file=sys.stderr)
            rc = 1
            continue
        an = res.analysis
        print(f"[OK] {f}\n     detected : {an.kind.upper()}  ({an.reason})")
        print(f"     tables   : {len(an.tables)}   paragraphs: {len(an.paragraphs)}   OCR confidence: {an.mean_conf:.0f}%")
        for o in res.files:
            print(f"     written  : {o}")
        for w in an.warnings:
            print(f"     warning  : {w}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
