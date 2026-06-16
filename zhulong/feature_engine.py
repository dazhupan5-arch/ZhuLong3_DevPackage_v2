"""
特征工程：M1 → M5/M60，计算序列特征与 1 小时背景特征。
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "return", "hl_ratio", "close_pos", "volume_change",
    "rsi", "macd_line", "macd_signal", "macd_hist", "atr",
    "channel_width", "channel_position", "above_upper", "below_lower",
    "ema_diff", "price_ema30", "price_ema60", "ema30_slope", "ema_cross",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


@dataclass
class BarStore:
    """按品种缓存 M1 并合成 M5。"""

    m1: dict[str, Deque[dict]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=5000)))
    m5: dict[str, pd.DataFrame] = field(default_factory=dict)

    def ingest_m1(self, bar: dict) -> Optional[pd.Timestamp]:
        sym = bar["symbol"]
        ts = pd.Timestamp(bar["time"])
        self.m1[sym].append(
            {
                "time": ts,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar.get("volume", 0)),
            }
        )
        df = pd.DataFrame(list(self.m1[sym]))
        if df.empty:
            return None
        df = df.set_index("time").sort_index()
        m5 = (
            df.resample("5min", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        self.m5[sym] = m5
        return ts


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def compute_m5_features(m5: pd.DataFrame, atr_period: int = 14, ema_fast: int = 30, ema_slow: int = 60) -> pd.DataFrame:
    df = m5.copy()
    df["return"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["hl_ratio"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_pos"] = (df["close"] - df["low"]) / hl
    vol_ma = df["volume"].rolling(5).mean().replace(0, np.nan)
    df["volume_change"] = df["volume"] / vol_ma

    df["rsi"] = _rsi(df["close"], 14) / 100.0
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_line"] = macd_line / df["close"]
    df["macd_signal"] = macd_signal / df["close"]
    df["macd_hist"] = (macd_line - macd_signal) / df["close"]

    atr = _atr(df, atr_period)
    df["atr"] = atr / df["close"]
    ema30 = df["close"].ewm(span=ema_fast, adjust=False).mean()
    ema60 = df["close"].ewm(span=ema_slow, adjust=False).mean()
    upper = ema30 + 3 * atr
    lower = ema30 - 3 * atr
    width = (upper - lower).replace(0, np.nan)
    df["channel_width"] = width / df["close"]
    df["channel_position"] = (df["close"] - lower) / width
    df["above_upper"] = (df["close"] > upper).astype(float)
    df["below_lower"] = (df["close"] < lower).astype(float)
    df["ema_diff"] = (ema30 - ema60) / ema60.replace(0, np.nan)
    df["price_ema30"] = (df["close"] - ema30) / ema30.replace(0, np.nan)
    df["price_ema60"] = (df["close"] - ema60) / ema60.replace(0, np.nan)
    ema30_shift = ema30.shift(5)
    df["ema30_slope"] = (ema30 - ema30_shift) / ema30_shift.replace(0, np.nan)
    cross = np.sign(ema30 - ema60) - np.sign(ema30.shift(1) - ema60.shift(1))
    df["ema_cross"] = cross.clip(-1, 1)

    idx = df.index
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    return df[FEATURE_COLUMNS].dropna()


def compute_hourly_background(m5: pd.DataFrame) -> np.ndarray:
    """1 小时背景特征（约 10 维）。"""
    table = _hourly_background_table(m5)
    if len(table) == 0:
        return np.zeros(10, dtype=np.float32)
    return table[-1]


def _hourly_background_table(m5: pd.DataFrame) -> np.ndarray:
    h1 = (
        m5.resample("1h", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    if len(h1) < 5:
        return np.zeros((0, 10), dtype=np.float32)

    ema30 = h1["close"].ewm(span=30, adjust=False).mean()
    ema60 = h1["close"].ewm(span=60, adjust=False).mean()
    atr = _atr(h1, 14)
    close = h1["close"]
    diff = close.diff()
    pct_std = close.pct_change().rolling(4).std().fillna(0.0)
    up_count = (diff > 0).rolling(4).sum().fillna(0.0)
    down_count = (diff < 0).rolling(4).sum().fillna(0.0)
    up_bar = (close > close.shift(1)).astype(float).fillna(0.0)

    return np.column_stack(
        [
            (close - ema30) / ema30,
            (close - ema60) / ema60,
            (3 * atr) / close,
            (close - (ema30 - 3 * atr)) / np.maximum(6 * atr, 1e-9),
            pct_std,
            up_count,
            down_count,
            up_bar,
            np.zeros(len(h1)),
            np.zeros(len(h1)),
        ]
    ).astype(np.float32)


def precompute_hourly_backgrounds(m5: pd.DataFrame, times: pd.DatetimeIndex) -> np.ndarray:
    """批量计算各 M5 时点对应的 1h 背景特征（避免训练 O(n^2) 重采样）。"""
    h1 = (
        m5.resample("1h", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    table = _hourly_background_table(m5)
    out = np.zeros((len(times), 10), dtype=np.float32)
    if len(h1) == 0 or len(table) == 0:
        return out

    idx = h1.index.searchsorted(times, side="right") - 1
    valid = idx >= 4
    out[valid] = table[idx[valid]]
    return out


def latest_sequence(m5_feat: pd.DataFrame, seq_len: int = 60) -> Optional[np.ndarray]:
    if len(m5_feat) < seq_len:
        return None
    return m5_feat.tail(seq_len).values.astype(np.float32)


def current_atr_pct(m5: pd.DataFrame, period: int = 14) -> float:
    atr = _atr(m5, period)
    if atr.empty or np.isnan(atr.iloc[-1]):
        return 0.0
    return float(atr.iloc[-1] / m5["close"].iloc[-1] * 100)


def compute_mtf_trend_features(m5: pd.DataFrame) -> pd.DataFrame:
    """M15/H1 多周期趋势，对齐到 M5 index。"""
    out = pd.DataFrame(index=m5.index)
    m15 = (
        m5.resample("15min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    h1 = (
        m5.resample("1h", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    if len(m15) < 10 or len(h1) < 10:
        out["m15_ema_bias"] = 0.0
        out["m15_trend"] = 0.0
        out["h1_rsi"] = 0.5
        out["h1_macd_sign"] = 0.0
        out["h1_ema_bias"] = 0.0
        out["h1_atr_pct"] = 0.0
        return out

    m15_ema = m15["close"].ewm(span=20, adjust=False).mean()
    m15_bias = ((m15["close"] - m15_ema) / m15_ema.replace(0, np.nan)).rename("m15_ema_bias")
    m15_trend = np.sign(m15["close"].diff(3)).rename("m15_trend")

    h1_rsi = (_rsi(h1["close"], 14) / 100.0).rename("h1_rsi")
    ema12 = h1["close"].ewm(span=12, adjust=False).mean()
    ema26 = h1["close"].ewm(span=26, adjust=False).mean()
    h1_macd = np.sign(ema12 - ema26).rename("h1_macd_sign")
    h1_ema60 = h1["close"].ewm(span=60, adjust=False).mean()
    h1_bias = ((h1["close"] - h1_ema60) / h1_ema60.replace(0, np.nan)).rename("h1_ema_bias")
    h1_atr = (_atr(h1, 14) / h1["close"]).rename("h1_atr_pct")

    for name, series in [
        ("m15_ema_bias", m15_bias),
        ("m15_trend", m15_trend),
        ("h1_rsi", h1_rsi),
        ("h1_macd_sign", h1_macd),
        ("h1_ema_bias", h1_bias),
        ("h1_atr_pct", h1_atr),
    ]:
        idx = series.index.searchsorted(m5.index, side="right") - 1
        valid = idx >= 0
        aligned = np.zeros(len(m5), dtype=np.float32)
        aligned[valid] = series.to_numpy()[idx[valid]]
        out[name] = aligned
    return out


MTF_COLUMNS = [
    "m15_ema_bias", "m15_trend", "h1_rsi", "h1_macd_sign", "h1_ema_bias", "h1_atr_pct",
]


def sequence_stats_from_window(window: np.ndarray) -> np.ndarray:
    """60×22 窗口 → 紧凑统计向量（34 维）。"""
    last = window[-1]
    ret = window[:, 0]
    rsi = window[:, 4]
    atr = window[:, 8]
    parts: list[float] = list(last.astype(float))
    parts.extend([
        float(ret[-5:].mean()), float(ret[-10:].mean()), float(ret[-20:].mean()),
        float(ret[-5:].std()), float(ret[-10:].std()), float(ret[-20:].std()),
        float(np.polyfit(np.arange(10), ret[-10:], 1)[0]) if len(ret) >= 10 else 0.0,
        float(np.polyfit(np.arange(20), ret[-20:], 1)[0]) if len(ret) >= 20 else 0.0,
        float(rsi[-5:].mean()), float(atr[-5:].mean()),
        float(ret.max() - ret.min()),
        float(ret[-1] > ret[-10:].mean()),
    ])
    return np.array(parts, dtype=np.float32)


def fused_feature_dim() -> int:
    """sequence_stats(34) + mtf(6) + hourly(10) = 50"""
    return len(FEATURE_COLUMNS) + 12 + 6 + 10


def build_fused_row(
    window: np.ndarray,
    mtf_row: np.ndarray,
    hourly_row: np.ndarray,
) -> np.ndarray:
    stats = sequence_stats_from_window(window)
    return np.concatenate([stats, mtf_row.astype(np.float32), hourly_row.astype(np.float32)])
