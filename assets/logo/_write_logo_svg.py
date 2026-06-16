#!/usr/bin/env python3
"""Generate 烛龙 logo — abstract day/night split lens + vertical slit."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent

# Minimal almond lens (128×128, centered at 64,64)
EYE = (
    "M 36 64 "
    "C 40 50, 52 42, 64 42 "
    "C 76 42, 88 50, 92 64 "
    "C 88 78, 76 86, 64 86 "
    "C 52 86, 40 78, 36 64 Z"
)
EYE_UPPER = (
    "M 36 64 "
    "C 40 50, 52 42, 64 42 "
    "C 76 42, 88 50, 92 64 "
    "L 36 64 Z"
)
EYE_LOWER = (
    "M 36 64 "
    "L 92 64 "
    "C 88 78, 76 86, 64 86 "
    "C 52 86, 40 78, 36 64 Z"
)
PUPIL = "M 62 48 L 66 48 L 66 80 L 62 80 Z"


def svg_defs() -> str:
    return """
    <linearGradient id="gold-bright" x1="36" y1="42" x2="92" y2="86" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#F5DC78"/>
      <stop offset="45%" stop-color="#D4A820"/>
      <stop offset="100%" stop-color="#B8860B"/>
    </linearGradient>
    <linearGradient id="gold-rim" x1="64" y1="42" x2="64" y2="86" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#E8C547"/>
      <stop offset="100%" stop-color="#8A6914"/>
    </linearGradient>
    <clipPath id="icon">
      <rect width="128" height="128" rx="26"/>
    </clipPath>
  """


def svg_logo(bg: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" role="img" aria-label="Zhulong">
  <defs>{svg_defs()}
  </defs>
  <g clip-path="url(#icon)">
    <rect width="128" height="128" fill="{bg}"/>
    <path id="eye-upper" fill="url(#gold-bright)" d="{EYE_UPPER}"/>
    <path id="eye-lower" fill="#141414" d="{EYE_LOWER}"/>
    <path id="eye-rim" fill="none" stroke="url(#gold-rim)" stroke-width="1.6" d="{EYE}"/>
    <path id="pupil" fill="{bg if bg != '#F5F5F7' else '#181818'}" d="{PUPIL}"/>
  </g>
</svg>
"""


def svg_mark_mono(stroke: str, fill_upper: str, fill_lower: str, pupil: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" role="img" aria-label="Zhulong mark">
  <path id="eye-upper" fill="{fill_upper}" d="{EYE_UPPER}"/>
  <path id="eye-lower" fill="{fill_lower}" d="{EYE_LOWER}"/>
  <path id="eye-rim" fill="none" stroke="{stroke}" stroke-width="1.6" d="{EYE}"/>
  <path id="pupil" fill="{pupil}" d="{PUPIL}"/>
</svg>
"""


COLORS = {
    "bg": (10, 10, 10),
    "bg_light": (245, 245, 247),
    "gold_bright": (212, 168, 32),
    "gold_rim": (142, 105, 20),
    "eye_lower": (20, 20, 20),
    "pupil": (10, 10, 10),
    "pupil_light": (24, 24, 24),
}

PATHS = {
    "eye-upper": (EYE_UPPER, 0, "gold_bright", True),
    "eye-lower": (EYE_LOWER, 0, "eye_lower", True),
    "eye-rim": (EYE, 1.6, "gold_rim", False),
    "pupil": (PUPIL, 0, "pupil", True),
}


def main() -> None:
    vec = ROOT / "vector"
    vec.mkdir(parents=True, exist_ok=True)

    (vec / "zhulong-logo.svg").write_text(svg_logo("#0A0A0A"), encoding="utf-8")
    (vec / "zhulong-logo-light.svg").write_text(svg_logo("#F5F5F7"), encoding="utf-8")
    (vec / "zhulong-logo-mark-gold.svg").write_text(
        svg_mark_mono("#B8860B", "#D4A820", "#181818", "#0A0A0A"), encoding="utf-8"
    )
    (vec / "zhulong-logo-mark-white.svg").write_text(
        svg_mark_mono("#F5F5F7", "#F5F5F7", "#181818", "#181818"), encoding="utf-8"
    )
    (vec / "zhulong-logo-mark-dark.svg").write_text(
        svg_mark_mono("#181818", "#181818", "none", "#181818"), encoding="utf-8"
    )
    print("Logo SVG updated")


if __name__ == "__main__":
    main()
