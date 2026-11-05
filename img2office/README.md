# img2office — image → editable Word / Excel

Detects what an image contains and writes a **fully editable** file:

| Image contains                | Output                                                         |
|-------------------------------|----------------------------------------------------------------|
| tables (with or without lines)| **.xlsx** – one sheet per table, merged cells, real numbers    |
| paragraph text                | **.docx** – paragraphs, headings, bold, bullets, alignment     |
| mixed                         | **.docx** with native Word tables in reading order (or .xlsx when ≥70 % of the words are table words) |

You can always override the decision (`--format word|excel|both`, or the drop-down in the web app).

## Install
1. Install the Tesseract OCR engine (v4 or v5)
   * Windows: https://github.com/UB-Mannheim/tesseract/wiki (set `TESSERACT_CMD` if it is not on PATH)
   * macOS: `brew install tesseract`  ·  Linux: `sudo apt install tesseract-ocr`
   * other languages: `tesseract-ocr-hin`, `-fra`, … then use `--lang hin` / `eng+hin`
2. `pip install -r requirements.txt`

## Use
```bash
python -m img2office photo.png                       # auto-detect, writes ./output/photo.docx or .xlsx
python -m img2office scan1.jpg scans_folder/ -o out  # batch
python -m img2office table.png -f both --debug       # both files + picture of what was detected
python webapp.py                                     # web UI on http://127.0.0.1:5000
```
```python
from img2office import convert
res = convert("photo.png", "out")        # res.files, res.kind, res.analysis
```

## How it works
1. **Clean-up** – EXIF/90°/180° rotation, de-skew, dark-background inversion, shadow removal, denoising (noisy images only), automatic up-scaling to the text size Tesseract likes best.
2. **Ruled tables** – grid lines (dark, light-gray or edges of coloured bars) → cell grid; merged cells are recovered from missing borders; every cell is OCR-ed from its own crop (ruling lines removed, white-on-dark handled).
3. **Borderless tables** – aligned word columns across ≥3 rows; each cell is re-read from a tight crop to catch dropped digits.
4. **Paragraphs** – lines → paragraphs (spacing, indents, bullets), headings (size), bold (stroke width), left/centre/right/justify.
5. **Clean-up of values** – ligatures (ﬁ→fi), numeric columns repaired (O→0, l→1, S→5), numbers written as real Excel numbers (`$1,250.50`, `12%`, `(300)`); IDs such as `007` stay text.
6. Low-confidence cells are listed as warnings (and highlighted yellow in Excel).

## Tests
`python -m pytest -q tests` (generates realistic table images with headless Chromium, incl. a skewed/noisy scan).

## Limits (be aware)
Handwriting, curved/perspective photos, and tables without any outer structure *and* very long wrapped cell text can still need a manual look; always skim the highlighted cells.
