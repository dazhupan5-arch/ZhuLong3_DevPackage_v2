"""宏观日历特征（G9 / macro_events.csv）。"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from zhulong.utils.paths import macro_events_path

logger = logging.getLogger(__name__)

# 全局缓存，MacroCalendarThread 每日刷新
_MACRO_CACHE: dict = {"df": None, "lock": threading.Lock()}


def load_macro_csv(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or macro_events_path()
    if not p.is_file():
        logger.warning("宏观日历不存在: %s", p)
        return pd.DataFrame(columns=["datetime", "event_name", "impact", "currency"])
    df = pd.read_csv(p)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime")


def set_macro_cache(df: pd.DataFrame) -> None:
    with _MACRO_CACHE["lock"]:
        _MACRO_CACHE["df"] = df


def get_macro_cache() -> pd.DataFrame:
    with _MACRO_CACHE["lock"]:
        if _MACRO_CACHE["df"] is None:
            set_macro_cache(load_macro_csv())
        return _MACRO_CACHE["df"].copy()


IMPACT_MAP = {"high": 1.0, "medium": 0.5, "mid": 0.5, "low": 0.25}


def macro_features(now: Optional[datetime] = None, max_types: int = 10) -> np.ndarray:
    """返回宏观特征向量（维度约 8，与 config macro_feature_dim 对齐）。"""
    import numpy as np

    df = get_macro_cache()
    if df.empty:
        return np.zeros(8, dtype=np.float32)

    now = now or datetime.now()
    ts = pd.Timestamp(now)
    future = df[df["datetime"] >= ts]
    past = df[df["datetime"] < ts]

    hours_to_next = 999.0
    next_impact = 0.0
    type_onehot = np.zeros(max_types, dtype=np.float32)
    if not future.empty:
        nxt = future.iloc[0]
        hours_to_next = (nxt["datetime"] - ts).total_seconds() / 3600.0
        next_impact = IMPACT_MAP.get(str(nxt["impact"]).lower(), 0.5)
        type_onehot[0] = 1.0  # 简化：首类占位

    hours_since = 999.0
    if not past.empty:
        hours_since = (ts - past.iloc[-1]["datetime"]).total_seconds() / 3600.0

    just_happened = 1.0 if hours_since <= 1.0 else 0.0

    vec = np.array(
        [
            min(hours_to_next, 999.0),
            next_impact,
            min(hours_since, 999.0),
            just_happened,
            type_onehot[0],
            type_onehot[1] if max_types > 1 else 0.0,
            type_onehot[2] if max_types > 2 else 0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    return vec[:8]


class MacroCalendarThread(threading.Thread):
    def __init__(self, reload_hours: float, stop_event: threading.Event) -> None:
        super().__init__(name="MacroCalendarThread", daemon=True)
        self._reload_hours = reload_hours
        self._stop = stop_event

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                set_macro_cache(load_macro_csv())
                logger.info("宏观日历已重载")
            except Exception as exc:
                logger.exception("宏观日历重载失败: %s", exc)
            self._stop.wait(self._reload_hours * 3600)
