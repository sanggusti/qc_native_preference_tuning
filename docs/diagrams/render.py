"""Render a .drawio file to SVG and PNG with the draw.io viewer in headless Chromium."""
import json, sys, pathlib
from html import escape as html_escape
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).parent
VIEWER = (HERE / "viewer-static.min.js").read_text()

def render(drawio_path, out_svg, out_png):
    xml = pathlib.Path(drawio_path).read_text()
    cfg = {"highlight": "#0000ff", "nav": False, "resize": True, "toolbar": None, "xml": xml}
    html = f"""<!doctype html><html><head><meta charset='utf-8'><style>body{{margin:0;background:#fff}}</style></head>
<body><div class='mxgraph' style='max-width:100%;border:0;' data-mxgraph="{html_escape(json.dumps(cfg), quote=True)}"></div>
<script>{VIEWER}</script></body></html>"""
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome" if pathlib.Path("/opt/pw-browsers/chromium-1194/chrome-linux/chrome").exists() else None, args=["--no-sandbox"])
        page = b.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=2)
        page.set_content(html)
        page.wait_for_selector("svg", timeout=20000)
        page.wait_for_timeout(500)
        svg = page.evaluate("() => { const s=document.querySelector('svg'); s.setAttribute('xmlns','http://www.w3.org/2000/svg'); s.setAttribute('xmlns:xlink','http://www.w3.org/1999/xlink'); return s.outerHTML; }")
        pathlib.Path(out_svg).write_text(svg)
        el = page.query_selector("svg")
        el.screenshot(path=out_png)
        b.close()

if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3])
