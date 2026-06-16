"""导入、清洗供应商 M5/M1 CSV。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VENDOR_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]


def load_vendor_csv(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    peek = path.read_text(encoding="utf-8", errors="replace").splitlines()[:2]
    has_header = peek and peek[0].lower().startswith("time")

    if has_header:
        df = pd.read_csv(path, parse_dates=["time"])
        out = df.set_index("time").sort_index()
    else:
        df = pd.read_csv(
            path,
            names=VENDOR_COLUMNS,
            header=None,
            dtype={"date": str, "time": str},
        )
        ts = pd.to_datetime(df["date"].str.replace(".", "-", regex=False) + " " + df["time"])
        out = pd.DataFrame(
            {
                "time": ts,
                "open": pd.to_numeric(df["open"], errors="coerce"),
                "high": pd.to_numeric(df["high"], errors="coerce"),
                "low": pd.to_numeric(df["low"], errors="coerce"),
                "close": pd.to_numeric(df["close"], errors="coerce"),
                "volume": pd.to_numeric(df["volume"], errors="coerce"),
            }
        )
        out = out.set_index("time").sort_index()

    out = out[~out.index.duplicated(keep="last")]
    logger.info("loaded %s rows %s .. %s", len(out), out.index.min(), out.index.max())
    return out


def resample_m1_to_m5(m1: pd.DataFrame) -> pd.DataFrame:
    return (
        m1.resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def clean_m5_gaps(m5: pd.DataFrame, max_gap_minutes: int = 10) -> pd.DataFrame:
    """连续缺失超过 2 根 M5（>10 分钟）的片段整段删除。"""
    if m5.empty:
        return m5
    gap = m5.index.to_series().diff()
    break_ix = np.where(gap > pd.Timedelta(minutes=max_gap_minutes))[0]
    if len(break_ix) == 0:
        return m5

    keep = np.ones(len(m5), dtype=bool)
    starts = np.concatenate([[0], break_ix])
    ends = np.concatenate([break_ix, [len(m5)]])
    for s, e in zip(starts, ends):
        seg = m5.iloc[s:e]
        if len(seg) <= 2:
            keep[s:e] = False
    cleaned = m5.iloc[keep]
    logger.info("gap clean removed %s bars, kept %s", int((~keep).sum()), len(cleaned))
    return cleaned


def remove_low_volume_weekends(m5: pd.DataFrame, min_volume: float = 1.0) -> pd.DataFrame:
    """去掉 volume 极低的 K 线（周末/节假日）。"""
    mask = m5["volume"] >= min_volume
    out = m5.loc[mask]
    logger.info("volume filter removed %s bars", int((~mask).sum()))
    return out


def to_standard_csv(m5: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = m5.reset_index()
    df.to_csv(path, index=False)
    logger.info("wrote %s (%s rows)", path, len(df))


def prepare_m5(raw_path: Path, out_path: Path, from_m1: bool = False) -> pd.DataFrame:
    df = load_vendor_csv(raw_path)
    if from_m1:
        df = resample_m1_to_m5(df)
    df = remove_low_volume_weekends(df)
    df = clean_m5_gaps(df)
    to_standard_csv(df, out_path)
    return df
