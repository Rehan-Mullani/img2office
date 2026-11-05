"""Run:  python -m pytest -q   (needs tesseract; samples are generated on first run)"""
import os, sys
import pytest
from docx import Document
from openpyxl import load_workbook

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
S = os.path.join(HERE, "samples")
from img2office import analyze, convert  # noqa: E402
from img2office.engine import Table  # noqa: E402

if not os.path.exists(os.path.join(S, "mixed.png")):
    sys.path.insert(0, HERE)
    import make_samples
    os.makedirs(S, exist_ok=True)
    for k, v in make_samples.PAGES.items():
        make_samples.render(k, v)


def table_of(an):
    return an.tables[0].cells


def test_prompt_image_becomes_word(tmp_path):
    r = convert(os.path.join(S, "task1_prompt.png"), str(tmp_path))
    assert r.kind == "word"
    d = Document(r.files[0])
    text = [p.text for p in d.paragraphs if p.text]
    assert text[0] == "Task-1" and d.paragraphs[0].runs[0].bold
    assert "If image contains tables, then it should create an excel file. Likewise, if image contains" in text[1]
    assert text[2].endswith("with good accuracy of content.")


def test_ruled_table_with_merged_cells_becomes_excel(tmp_path):
    r = convert(os.path.join(S, "ruled_merged.png"), str(tmp_path))
    assert r.kind == "excel"
    ws = load_workbook(r.files[0]).active
    vals = [[c.value for c in row] for row in ws.iter_rows(min_row=3)]
    assert vals[0] == ["Region", "Product", "Units", "Revenue"]
    assert vals[1][2] == 1250 and vals[1][3] == 12500.0       # real numbers, not text
    assert vals[5][2] == 4095 and abs(vals[5][3] - 51376.25) < 1e-9
    assert len(ws.merged_cells.ranges) == 3


def test_borderless_table(tmp_path):
    an = analyze(os.path.join(S, "borderless.png"))
    assert an.kind == "excel" and not an.tables[0].ruled
    assert table_of(an)[3] == ["Stapler", "3", "120.00", "360.00"]


def test_light_grid_table():
    an = analyze(os.path.join(S, "lightgrid.png"))
    assert table_of(an)[2] == ["002", "Priya Nair", "Chemistry", "92"]


def test_mixed_goes_to_word_with_real_table(tmp_path):
    r = convert(os.path.join(S, "mixed.png"), str(tmp_path))
    assert r.kind == "word"
    d = Document(r.files[0])
    assert len(d.tables) == 1 and d.tables[0].cell(2, 2).text == "In progress"


def test_noisy_skewed_scan():
    an = analyze(os.path.join(S, "ruled_scan.jpg"))
    assert table_of(an)[0] == ["Region", "Product", "Units", "Revenue"]
    assert table_of(an)[5][3] == "$51,376.25"


def test_bullets_and_center():
    an = analyze(os.path.join(S, "bullets.png"))
    kinds = [(p.kind, p.align) for p in an.paragraphs]
    assert kinds[0][0] == "heading" and kinds[1][0] == "bullet" and kinds[-1] == ("text", "center")


def test_forced_formats(tmp_path):
    r = convert(os.path.join(S, "mixed.png"), str(tmp_path), fmt="both")
    assert {os.path.splitext(f)[1] for f in r.files} == {".docx", ".xlsx"}
