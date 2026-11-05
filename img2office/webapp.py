"""
Web front-end:  python webapp.py   ->  http://127.0.0.1:5000
Drop an image, the app detects tables / paragraphs and offers an editable .docx or .xlsx.
"""
from __future__ import annotations

import base64
import html
import os
import shutil
import tempfile
import time
import uuid

import cv2
from flask import Flask, jsonify, render_template_string, request, send_from_directory

from img2office import convert
from img2office.engine import Paragraph, Table

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
WORK = os.path.join(tempfile.gettempdir(), "img2office_web")
os.makedirs(WORK, exist_ok=True)
ALLOWED = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif")


def _cleanup(max_age=3600):
    now = time.time()
    for d in os.listdir(WORK):
        p = os.path.join(WORK, d)
        if os.path.isdir(p) and now - os.path.getmtime(p) > max_age:
            shutil.rmtree(p, ignore_errors=True)


def _preview(an) -> str:
    out = []
    for el in an.elements:
        if isinstance(el, Paragraph):
            txt = html.escape(el.text)
            if el.kind == "heading":
                out.append(f"<h3>{txt}</h3>")
            elif el.kind == "bullet":
                out.append(f"<p class='b'>• {txt}</p>")
            else:
                st = f"text-align:{el.align};" + ("font-weight:600;" if el.bold else "")
                out.append(f"<p style='{st}'>{txt}</p>")
        else:
            skip, rows = set(), []
            for (r0, c0, r1, c1) in el.merges:
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        if (r, c) != (r0, c0):
                            skip.add((r, c))
            span = {(m[0], m[1]): m for m in el.merges}
            for r in range(el.nrows):
                tds = []
                for c in range(el.ncols):
                    if (r, c) in skip:
                        continue
                    m = span.get((r, c))
                    attr = f" rowspan={m[2]-m[0]+1} colspan={m[3]-m[1]+1}" if m else ""
                    low = " class='low'" if (r, c) in el.low_conf else ""
                    tds.append(f"<td{attr}{low}>{html.escape(el.cells[r][c]).replace(chr(10), '<br>')}</td>")
                rows.append("<tr>" + "".join(tds) + "</tr>")
            out.append("<div class='tw'><table>" + "".join(rows) + "</table></div>")
    return "".join(out) or "<p class='muted'>Nothing recognised.</p>"


@app.post("/convert")
def do_convert():
    _cleanup()
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify(error="Please choose an image."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error=f"Unsupported file type {ext or '?'}. Use PNG, JPG, BMP, TIFF, WEBP or GIF."), 400
    fmt = request.form.get("format", "auto")
    lang = request.form.get("lang", "eng").strip() or "eng"
    if not all(ch.isalnum() or ch in "_+" for ch in lang):
        return jsonify(error="Invalid language code."), 400
    job = uuid.uuid4().hex
    d = os.path.join(WORK, job)
    os.makedirs(d)
    src = os.path.join(d, "input" + ext)
    f.save(src)
    stem = os.path.splitext(os.path.basename(f.filename))[0][:60] or "output"
    stem = "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in stem)
    try:
        res = convert(src, d, fmt, lang, debug=True, name=stem)
    except Exception as e:  # noqa
        return jsonify(error=f"Could not process this image: {e}"), 500
    an = res.analysis
    dbg = ""
    if res.debug_path and os.path.exists(res.debug_path):
        im = cv2.imread(res.debug_path)
        h, w = im.shape[:2]
        if w > 1400:
            im = cv2.resize(im, (1400, int(h * 1400 / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
        dbg = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return jsonify(
        detected=an.kind, reason=an.reason, tables=len(an.tables), paragraphs=len(an.paragraphs),
        confidence=round(an.mean_conf), warnings=an.warnings, preview=_preview(an), debug=dbg,
        files=[{"name": os.path.basename(p), "url": f"/download/{job}/{os.path.basename(p)}",
                "type": "Word" if p.endswith(".docx") else "Excel"} for p in res.files],
    )


@app.get("/download/<job>/<name>")
def download(job, name):
    if not job.isalnum():
        return "bad request", 400
    return send_from_directory(os.path.join(WORK, job), name, as_attachment=True)


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Image to Word / Excel</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2330;--muted:#667085;--line:#e3e6ec;--acc:#2563eb;--ok:#15803d;--warn:#b45309}
@media(prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--ink:#e8ebf0;--muted:#98a2b3;--line:#2a3140;--acc:#6ea0ff;--ok:#4ade80;--warn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 60px}h1{font-size:24px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
#drop{border:2px dashed var(--line);border-radius:12px;padding:36px 16px;text-align:center;cursor:pointer;color:var(--muted)}
#drop.over{border-color:var(--acc);background:color-mix(in srgb,var(--acc) 8%,transparent)}
#drop b{color:var(--ink)}.row{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin-top:14px}
label{font-size:13px;color:var(--muted);display:block;margin-bottom:4px}select,input[type=text]{padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);font:inherit}
button{padding:10px 18px;border:0;border-radius:8px;background:var(--acc);color:#fff;font:inherit;font-weight:600;cursor:pointer}button:disabled{opacity:.5;cursor:default}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;background:color-mix(in srgb,var(--acc) 14%,transparent);color:var(--acc);font-weight:600;font-size:13px}
.muted{color:var(--muted)}.dl a{display:inline-block;margin:6px 8px 0 0;padding:9px 16px;border-radius:8px;border:1px solid var(--acc);color:var(--acc);text-decoration:none;font-weight:600}
.warn{color:var(--warn);margin:6px 0 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
#prev p{margin:0 0 .7em}#prev .b{margin-left:1em}#prev h3{margin:.4em 0}.tw{overflow:auto;margin:8px 0}table{border-collapse:collapse;font-size:14px}td{border:1px solid var(--line);padding:5px 9px;vertical-align:top}td.low{background:#fff2cc55}
#dbg{max-width:100%;border:1px solid var(--line);border-radius:8px}#thumb{max-height:90px;border-radius:6px;margin-top:10px}
</style></head><body><main>
<h1>Image → editable Word / Excel</h1>
<p class=sub>Tables become an Excel sheet, paragraphs become a Word document, mixed pages become Word with real tables.</p>
<div class=card>
 <div id=drop><b>Drop an image here</b> or click to choose<br><span class=muted>PNG · JPG · BMP · TIFF · WEBP</span><br><img id=thumb hidden></div>
 <input id=file type=file accept="image/*" hidden>
 <div class=row>
  <div><label for=fmt>Output</label><select id=fmt><option value=auto>Auto-detect (recommended)</option><option value=word>Word (.docx)</option><option value=excel>Excel (.xlsx)</option><option value=both>Both</option></select></div>
  <div><label for=lang>OCR language</label><input id=lang type=text value=eng size=10 title="Tesseract code, e.g. eng, hin, eng+hin"></div>
  <button id=go disabled>Convert</button><span id=status class=muted></span>
 </div>
</div>
<div id=out hidden>
 <div class=card><span class=pill id=kind></span> <span id=reason class=muted></span>
  <div class=muted style="margin-top:6px"><span id=stats></span></div><div id=warns></div><div class=dl id=dl></div></div>
 <div class=grid><div class=card><b>Extracted content</b><div id=prev style="margin-top:10px"></div></div>
 <div class=card><b>What was detected</b><div style="margin-top:10px"><img id=dbg></div></div></div>
</div>
<script>
const $=id=>document.getElementById(id);let file=null;
const drop=$('drop'),inp=$('file');
function pick(f){if(!f)return;file=f;$('go').disabled=false;const u=URL.createObjectURL(f);$('thumb').src=u;$('thumb').hidden=false;$('status').textContent=f.name}
drop.onclick=()=>inp.click();inp.onchange=()=>pick(inp.files[0]);
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>pick(ev.dataTransfer.files[0]));
document.addEventListener('paste',ev=>{const it=[...ev.clipboardData.items].find(i=>i.type.startsWith('image/'));if(it)pick(it.getAsFile())});
$('go').onclick=async()=>{
 const fd=new FormData();fd.append('image',file,file.name||'pasted.png');fd.append('format',$('fmt').value);fd.append('lang',$('lang').value);
 $('go').disabled=true;$('status').textContent='Reading the image…';
 try{const r=await fetch('/convert',{method:'POST',body:fd});const j=await r.json();
  if(!r.ok){$('status').textContent=j.error||'Failed';return}
  $('status').textContent='';$('out').hidden=false;
  $('kind').textContent=j.detected==='excel'?'Excel':(j.detected==='word'?'Word':'Both');
  $('reason').textContent=j.reason;$('stats').textContent=`${j.tables} table(s) · ${j.paragraphs} paragraph(s) · OCR confidence ${j.confidence}%`;
  $('warns').innerHTML=j.warnings.map(w=>`<div class=warn>⚠ ${w.replace(/</g,'&lt;')}</div>`).join('');
  $('dl').innerHTML=j.files.map(f=>`<a href="${f.url}">Download ${f.type} (${f.name})</a>`).join('');
  $('prev').innerHTML=j.preview;$('dbg').src=j.debug;
 }catch(e){$('status').textContent='Error: '+e}finally{$('go').disabled=false}};
</script></main></body></html>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
