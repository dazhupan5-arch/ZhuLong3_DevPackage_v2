#!/usr/bin/env python3
"""从 SVG 矢量源导出 WinUI / 安装包 / 快捷方式所需的全部图标资源。"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
LOGO = ROOT / "assets" / "logo"
EXPORT = LOGO / "export_logo_assets.py"
ASSETS = ROOT / "src" / "ZhuLong.App" / "Assets"
BG = (10, 10, 10, 255)


def read_app_version() -> str:
    csproj = ROOT / "src" / "ZhuLong.App" / "ZhuLong.App.csproj"
    match = re.search(r"<Version>([\d.]+)</Version>", csproj.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"未在 {csproj} 找到 <Version>")
    return match.group(1)


def export_logo_png() -> None:
    subprocess.run([sys.executable, str(EXPORT)], cwd=ROOT, check=True)


def load_icon(name: str) -> Image.Image:
    path = LOGO / "png" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGBA")


def fit(img: Image.Image, size: int) -> Image.Image:
    if img.size == (size, size):
        return img
    return img.resize((size, size), Image.Resampling.LANCZOS)


def wide_banner(master: Image.Image, icon_size: int = 256) -> Image.Image:
    icon = fit(master, icon_size)
    wide = Image.new("RGBA", (620, 300), BG)
    wide.paste(icon, ((620 - icon_size) // 2, (300 - icon_size) // 2), icon)
    return wide


def save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    print("OK", path.relative_to(ROOT))


def sync_ico() -> None:
    favicon = LOGO / "favicon" / "favicon.ico"
    if not favicon.is_file():
        raise FileNotFoundError(favicon)
    for path in (
        LOGO / "zhulong.ico",
        ASSETS / "zhulong.ico",
        ASSETS / "app.ico",
    ):
        shutil.copy2(favicon, path)
        print("OK", path.relative_to(ROOT))


def main() -> None:
    version = read_app_version()
    print(f"App version: {version}")

    export_logo_png()
    ASSETS.mkdir(parents=True, exist_ok=True)

    master = load_icon("icon-1024x1024.png")
    banner = wide_banner(master, 256)

    square_assets: list[tuple[str, int]] = [
        ("TitleLogo.png", 80),
        ("StoreLogo.png", 50),
        ("Square44x44Logo.png", 44),
        ("Square150x150Logo.png", 150),
        ("Square44x44Logo.targetsize-24_altform-unplated.png", 24),
        ("Square44x44Logo.scale-200.png", 88),
        ("Square150x150Logo.scale-200.png", 300),
        ("LockScreenLogo.scale-200.png", 48),
    ]
    for name, size in square_assets:
        save_png(fit(master, size), ASSETS / name)

    for name in (
        "SplashScreen.scale-200.png",
        "Wide310x150Logo.scale-200.png",
        "SplashScreen.png",
        "Wide310x150Logo.png",
    ):
        save_png(banner, ASSETS / name)

    sync_ico()
    print(f"All brand assets synced for ZhuLong v{version}")


if __name__ == "__main__":
    main()
