"""Markdown -> PDF（零安装，Windows 自带 Edge 无头模式）

用法：
    python tools/md_to_pdf.py                # 转换下方 MD_FILES 里的全部文件
    python tools/md_to_pdf.py 提交报告.md     # 只转换指定的某一个

原理：
    Markdown --(markdown 包)--> HTML（带 @page 打印样式）
             --(Edge --headless --print-to-pdf)--> PDF

注意（都是踩过的坑，别改）：
    * 必须指定 --user-data-dir：Edge 正在运行时不指定会静默失败（退出码仍是 0）
    * 判成功要看"文件存在且体积 > 2000 字节"，不要看返回码
    * 本地图片必须换成 file:/// 绝对路径，否则无头浏览器里全是裂图
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# 要转换的文档（相对 docs/ 目录）
# 学习笔记.md 已加入：你补完笔记后跑一次脚本，会自动出 学习笔记.pdf
MD_FILES = ["提交报告.md", "训练过程记录.md", "学习过程记录.md", "概念题.md", "学习笔记.md"]

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
  font-size: 10.5pt; line-height: 1.8; color: #1f2328;
  margin: 0; padding: 0;
}
h1 { font-size: 19pt; border-bottom: 2.5px solid #1f4e79; color: #1f4e79;
     padding-bottom: 6px; margin: 0 0 14px; }
h2 { font-size: 14pt; border-left: 4px solid #2c6cb0; padding-left: 9px;
     color: #1f4e79; margin: 20px 0 10px; }
h3 { font-size: 12pt; color: #23527c; margin: 16px 0 8px; }
h2, h3 { break-after: avoid-page; page-break-after: avoid; }
p, li { orphans: 2; widows: 2; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 10px 0; }
th { background: #eef3f9; font-weight: 600; }
th, td { border: 1px solid #c9d3dd; padding: 5px 9px; text-align: left;
         vertical-align: top; }
table, pre, tr { page-break-inside: avoid; break-inside: avoid; }
img { max-width: 76%; display: block; margin: 10px auto;
      break-inside: avoid; page-break-inside: avoid; }
code { background: #f2f4f7; padding: 1px 5px; border-radius: 3px;
       font-family: Consolas, "Courier New", monospace; font-size: 9.5pt; }
pre { background: #f7f9fb; border-left: 3px solid #9bb4cc; padding: 9px 12px;
      overflow-x: auto; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #c9d3dd; margin: 10px 0; padding: 2px 14px;
             color: #4a5568; background: #fafbfc; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 18px 0; }
"""


def find_browser() -> str:
    for p in EDGE_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("找不到 Edge / Chrome，请手动确认安装路径")


def md_to_html(md_path: Path) -> str:
    import markdown

    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
    )

    base = md_path.parent

    def fix_img(m):
        src = m.group(1)
        if src.startswith(("http://", "https://", "file://", "data:")):
            return f'src="{src}"'
        abs_path = (base / src).resolve()
        return 'src="' + str(abs_path).replace("\\", "/") + '"'

    body = re.sub(r'src="([^"]+)"', fix_img, body)
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def convert(md_path: Path, edge: str) -> bool:
    pdf_path = md_path.with_suffix(".pdf")
    html_path = md_path.with_suffix(".tmp.html")

    html_path.write_text(md_to_html(md_path), encoding="utf-8")

    udd = os.path.join(tempfile.gettempdir(), "_edge_pdf_profile")
    cmd = [
        edge,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-pdf-header-footer",
        "--virtual-time-budget=8000",
        f"--user-data-dir={udd}",
        f"--print-to-pdf={pdf_path}",
        "file:///" + str(html_path).replace("\\", "/"),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        print(f"  [超时] {md_path.name}")
        return False
    finally:
        html_path.unlink(missing_ok=True)

    ok = pdf_path.is_file() and pdf_path.stat().st_size > 2000
    if ok:
        kb = pdf_path.stat().st_size / 1024
        print(f"  [OK] {pdf_path.name}  ({kb:.0f} KB)")
    else:
        print(f"  [失败] {md_path.name}（PDF 未生成或过小）")
    return ok


def main() -> int:
    edge = find_browser()
    print(f"浏览器: {edge}")

    targets = sys.argv[1:] if len(sys.argv) > 1 else MD_FILES
    print(f"待转换 {len(targets)} 个文件 -> {DOCS_DIR}\n")

    failed = []
    for name in targets:
        md_path = DOCS_DIR / name
        if not md_path.is_file():
            print(f"  [跳过] {name}（文件不存在）")
            failed.append(name)
            continue
        if not convert(md_path, edge):
            failed.append(name)

    print()
    if failed:
        print(f"完成，但有 {len(failed)} 个失败: {failed}")
        print("备选：VS Code 里装 Markdown PDF 扩展，右键 Export (pdf)")
        return 1
    print("全部转换完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
