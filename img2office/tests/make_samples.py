"""Generate realistic test images (HTML rendered by headless Chromium)."""
import os, sys
import numpy as np, cv2
from playwright.sync_api import sync_playwright

OUT = os.path.join(os.path.dirname(__file__), "samples")
CSS = "body{font-family:Arial,Helvetica,sans-serif;font-size:16px;margin:30px;width:780px;color:#111}table{border-collapse:collapse}td,th{padding:7px 12px}"

PAGES = {
"ruled_merged": """<h2>Quarterly Sales Report</h2>
<table border=1 style="border-color:#222;width:100%">
<tr style="background:#1f3864;color:#fff"><th>Region</th><th>Product</th><th>Units</th><th>Revenue</th></tr>
<tr><td rowspan=2>North</td><td>Widget A</td><td>1,250</td><td>$12,500.00</td></tr>
<tr><td>Widget B</td><td>980</td><td>$14,700.50</td></tr>
<tr><td rowspan=2>South</td><td>Widget A</td><td>760</td><td>$7,600.00</td></tr>
<tr><td>Widget B</td><td>1,105</td><td>$16,575.75</td></tr>
<tr><td colspan=2><b>Total</b></td><td>4,095</td><td>$51,376.25</td></tr>
</table>""",
"lightgrid": """<table style="width:100%;border:1px solid #ddd">
<tr><th style="border:1px solid #ddd;background:#f2f2f2">ID</th><th style="border:1px solid #ddd;background:#f2f2f2">Name</th><th style="border:1px solid #ddd;background:#f2f2f2">Dept</th><th style="border:1px solid #ddd;background:#f2f2f2">Score</th></tr>
<tr><td style="border:1px solid #ddd">001</td><td style="border:1px solid #ddd">Aarav Sharma</td><td style="border:1px solid #ddd">Physics</td><td style="border:1px solid #ddd">87.5</td></tr>
<tr style="background:#fafafa"><td style="border:1px solid #ddd">002</td><td style="border:1px solid #ddd">Priya Nair</td><td style="border:1px solid #ddd">Chemistry</td><td style="border:1px solid #ddd">92</td></tr>
<tr><td style="border:1px solid #ddd">003</td><td style="border:1px solid #ddd">Rohan Mehta</td><td style="border:1px solid #ddd">Mathematics</td><td style="border:1px solid #ddd">78.25</td></tr>
<tr style="background:#fafafa"><td style="border:1px solid #ddd">004</td><td style="border:1px solid #ddd">Sneha Iyer</td><td style="border:1px solid #ddd">Biology</td><td style="border:1px solid #ddd">65</td></tr>
</table>""",
"borderless": """<table style="width:100%">
<tr><th align=left>Item</th><th align=left>Quantity</th><th align=left>Unit Price</th><th align=left>Total</th></tr>
<tr><td>Notebook</td><td>12</td><td>45.00</td><td>540.00</td></tr>
<tr><td>Ballpoint pen</td><td>50</td><td>8.50</td><td>425.00</td></tr>
<tr><td>Stapler</td><td>3</td><td>120.00</td><td>360.00</td></tr>
<tr><td>Whiteboard marker</td><td>20</td><td>35.00</td><td>700.00</td></tr>
</table>""",
"mixed": """<h2>Project Status</h2>
<p style="line-height:1.5;text-align:justify">The project is progressing according to plan. During the last quarter the team completed the first two milestones and started the integration tests. The numbers below summarise the work packages and their current state.</p>
<table border=1 style="border-color:#444;width:100%">
<tr><th>Work package</th><th>Owner</th><th>Status</th></tr>
<tr><td>Design</td><td>Anita</td><td>Done</td></tr>
<tr><td>Backend</td><td>Ravi</td><td>In progress</td></tr>
<tr><td>Testing</td><td>Meera</td><td>Not started</td></tr>
</table>
<p style="line-height:1.5">Next review meeting is planned for the end of the month.</p>""",
"bullets": """<h1>Meeting Notes</h1><ul style="line-height:1.6"><li>Review the budget proposal</li><li>Assign owners to open issues</li><li>Plan the release for next month</li></ul>
<p style="text-align:center"><i>Thank you</i></p>""",
}

def render(name, html, w=860):
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": w, "height": 400}, device_scale_factor=1)
        pg.set_content(f"<style>{CSS}</style>{html}"); pg.screenshot(path=os.path.join(OUT, name + ".png"), full_page=True); b.close()

def degrade(src, dst, angle=2.0, noise=10, blur=1.0, jpeg=55):
    im = cv2.imread(src); h, w = im.shape[:2]
    m = cv2.getRotationMatrix2D((w/2, h/2), angle, 1); im = cv2.warpAffine(im, m, (w, h), borderValue=(255,255,255))
    im = cv2.GaussianBlur(im, (0,0), blur)
    n = np.random.default_rng(1).normal(0, noise, im.shape); im = np.clip(im + n, 0, 255).astype(np.uint8)
    cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, jpeg])

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for k, v in PAGES.items(): render(k, v)
    degrade(os.path.join(OUT, "ruled_merged.png"), os.path.join(OUT, "ruled_scan.jpg"), angle=1.2, noise=5, blur=0.8, jpeg=70)
    degrade(os.path.join(OUT, "ruled_merged.png"), os.path.join(OUT, "ruled_harsh.jpg"))
    degrade(os.path.join(OUT, "mixed.png"), os.path.join(OUT, "mixed_scan.jpg"), angle=-1.5, noise=5, blur=0.8, jpeg=70)
    print(sorted(os.listdir(OUT)))
