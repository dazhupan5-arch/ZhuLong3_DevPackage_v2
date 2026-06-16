"""M5 / bar 时间索引归一化（实机 C# 管道为 UTC tz-aware，训练/日历多为 naive）。"""

from __future__ import annotations

import pandas as pd


def normalize_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def normalize_datetime_index(index: pd.DatetimeIndex | pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def normalize_m5_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    out.index = normalize_datetime_index(out.index)
    return out.sort_index()
