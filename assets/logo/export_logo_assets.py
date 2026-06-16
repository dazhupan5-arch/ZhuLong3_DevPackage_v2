#!/usr/bin/env python3
"""Export 烛龙 logo：Skia 渲染 SVG 矢量源 → PNG / ICO。"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import skia
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SVG = ROOT / "vector" / "zhulong-logo.svg"
PNG_SIZES = (16, 20, 24, 32, 40, 48, 64, 80, 128, 180, 192, 256, 512, 1024)
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def read_app_version() -> str:
    csproj = ROOT.parent.parent / "src" / "ZhuLong.App" / "ZhuLong.App.csproj"
    match = re.search(r"<Version>([\d.]+)</Version>", csproj.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"未在 {csproj} 找到 <Version>")
    return match.group(1)


def render_svg(size: int, svg_path: Path = SVG) -> Image.Image:
    if not svg_path.is_file():
        raise FileNotFoundError(svg_path)
    stream = skia.MemoryStream(svg_path.read_bytes())
    dom = skia.SVGDOM.MakeFromStream(stream)
    surface = skia.Surface(size, size)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color4f(0, 0, 0, 1))
    dom.setContainerSize(skia.Size(size, size))
    dom.render(canvas)
    png_data = surface.makeImageSnapshot().encodeToData(skia.EncodedImageFormat.kPNG, 100)
    return Image.open(io.BytesIO(bytes(png_data))).convert("RGBA")


def save_ico(path: Path, sizes: tuple[int, ...]) -> None:
    """BMP 层 ICO — Windows 快捷方式 / GDI+ / 托盘兼容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    master = render_svg(max(sizes))
    master.save(path, format="ICO", sizes=[(n, n) for n in sizes], bitmap_format="bmp")


def main() -> None:
    import _write_logo_svg as writer

    writer.main()
    if not SVG.is_file():
        raise FileNotFoundError(SVG)

    out_png = ROOT / "png"
    out_mark = out_png / "mark"
    out_fav = ROOT / "favicon"
    for d in (out_png, out_mark, out_fav):
        d.mkdir(parents=True, exist_ok=True)

    (ROOT / "zhulong-logo.svg").write_text(SVG.read_text(encoding="utf-8"), encoding="utf-8")

    master = render_svg(1024)
    master.save(ROOT / "zhulong-logo-preview.png", optimize=True)

    for n in PNG_SIZES:
        img = master.resize((n, n), Image.Resampling.LANCZOS) if n != 1024 else master.copy()
        img.save(out_png / f"icon-{n}x{n}.png", optimize=True)

    mark_svg = ROOT / "vector" / "zhulong-logo-mark-gold.svg"
    for n in (32, 64, 128, 256, 512, 1024):
        render_svg(n, mark_svg).save(out_mark / f"mark-gold-{n}x{n}.png", optimize=True)

    save_ico(out_fav / "favicon.ico", ICO_SIZES)
    (ROOT / "zhulong.ico").write_bytes((out_fav / "favicon.ico").read_bytes())

    for n in (16, 32, 48, 180, 512):
        render_svg(n).save(out_fav / {
            16: "favicon-16x16.png",
            32: "favicon-32x32.png",
            48: "favicon-48x48.png",
            180: "apple-touch-icon.png",
            512: "android-chrome-512x512.png",
        }[n], optimize=True)

    (ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "source": "vector/zhulong-logo.svg",
                "renderer": "skia-svgdom",
                "design": "abstract-day-night-lens",
                "myth": "烛龙睁眼为昼、闭眼为夜",
                "elements": ["split-lens", "vertical-slit"],
                "colors": {
                    "background": "#0A0A0A",
                    "gold": "#D4A820",
                    "night": "#141414",
                },
                "app_version": read_app_version(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Logo PNG/ICO export OK (Skia SVG)")


if __name__ == "__main__":
    main()
