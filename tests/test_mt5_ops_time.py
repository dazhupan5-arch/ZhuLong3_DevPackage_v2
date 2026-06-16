"""MT5 API 时间偏移 — 与 ZhuLongIndicator ServerOffsetSec 对齐。"""

from __future__ import annotations

import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent / "ZhuLong.PythonEngine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from mt5_ops import offset_from_tick_delta  # noqa: E402


def test_offset_from_tick_delta_already_utc() -> None:
    assert offset_from_tick_delta(0) == 0
    assert offset_from_tick_delta(90) == 0
    assert offset_from_tick_delta(-60) == 0


def test_offset_from_tick_delta_wcg_utc_plus3() -> None:
    assert offset_from_tick_delta(10_800) == 10_800
    assert offset_from_tick_delta(10_750) == 10_800


def test_offset_from_tick_delta_out_of_range() -> None:
    assert offset_from_tick_delta(20 * 3600) == 0

