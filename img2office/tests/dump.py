import sys
from img2office import convert
from img2office.engine import Table, Paragraph
for f in sys.argv[1:]:
    r = convert(f, "/tmp/claude-0/out2", debug=True)
    an = r.analysis
    print("=====", f.split('/')[-1], "->", an.kind, "|", an.reason, "| conf %.0f scale %.2f" % (an.mean_conf, an.scale))
    for e in an.elements:
        if isinstance(e, Table):
            print(" TABLE", "ruled" if e.ruled else "free", e.nrows, "x", e.ncols, "merges", e.merges)
            for row in e.cells: print("   ", row)
        else:
            print(" PARA", e.kind, "bold" if e.bold else "", e.align, f"sz{e.size_ratio}", "|", e.text[:90])
    for w in an.warnings: print(" WARN", w)
