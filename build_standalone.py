"""build_standalone.py — 生成零依赖的单文件离线版
================================================================
把 index.html、style.css、app.js 与全部 data/*.json 内联为一个 HTML，
双击即可运行，不需要服务器、不需要网络、不依赖任何外部资源。

输出
    reading_efficiency_standalone.html
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

HERE = Path(__file__).parent
DOCS = HERE / "docs"
DATA = DOCS / "data"
OUT = HERE / "reading_efficiency_standalone.html"

DATA_FILES = {
    "materials": "materials.json",
    "norm": "norm.json",
    "material_gaze": "material_gaze.json",
    "ngram": "ngram.json",
    "scoring": "scoring.json",
    "vocab": "vocab.json",
}


def main() -> None:
    index = (DOCS / "index.html").read_text(encoding="utf-8")
    css = (DOCS / "style.css").read_text(encoding="utf-8")
    js = (DOCS / "app.js").read_text(encoding="utf-8")

    payload = {}
    for key, fname in DATA_FILES.items():
        payload[key] = json.loads((DATA / fname).read_text(encoding="utf-8"))
    data_js = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data_js = data_js.replace("</", "<\\/")

    # 示例文件内联为 data URI，避免离线版出现死链
    def data_uri(name: str, mime: str) -> str:
        raw = (DATA / name).read_bytes()
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    sample_txt = data_uri("sample_text.txt", "text/plain;charset=utf-8")
    sample_asc = data_uri("sample_reading.asc", "text/plain;charset=utf-8")

    out = index
    out = out.replace('<link rel="stylesheet" href="style.css">',
                      f"<style>\n{css}\n</style>")
    out = out.replace('<a href="data/sample_text.txt" download>示例文本</a>',
                      f'<a href="{sample_txt}" download="sample_text.txt">示例文本</a>')
    out = out.replace('<a href="data/sample_reading.asc" download>模拟 asc</a>',
                      f'<a href="{sample_asc}" download="sample_reading.asc">模拟 asc</a>')
    # 注意：JS 必须原样内联，任何文案替换都只对上方 HTML 部分生效
    out = out.replace(
        '<script src="app.js"></script>',
        f"<script>window.__DATA__={data_js};</script>\n<script>\n{js}\n</script>")

    OUT.write_text(out, encoding="utf-8")
    print(f"saved: {OUT} ({OUT.stat().st_size/1024/1024:.2f} MB)")

    # 自检：不应再引用任何外部资源
    for bad in ('href="data/', 'src="app.js"', 'href="style.css"', "http://", "https://"):
        if bad in out:
            if bad in ("http://", "https://"):
                # 只允许 SVG 命名空间
                import re
                hits = [m for m in re.findall(r"https?://[^\s\"'<>]+", out)
                        if m != "http://www.w3.org/2000/svg"]
                if hits:
                    print("  [警告] 仍有外部 URL:", hits[:3])
            else:
                print("  [警告] 仍有外部引用:", bad)
    print("self-check done")


if __name__ == "__main__":
    main()
