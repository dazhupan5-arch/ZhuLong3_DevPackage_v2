"""M5 技术指标（趋势/状态机/网格共用）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr_series(m5: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = m5["close"].shift(1)
    tr = pd.concat(
        [
            m5["high"] - m5["low"],
            (m5["high"] - prev_close).abs(),
            (m5["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def atr_pct(m5: pd.DataFrame, period: int = 14) -> float:
    atr = atr_series(m5, period)
    close = float(m5["close"].iloc[-1])
    if close <= 0 or atr.empty or np.isnan(atr.iloc[-1]):
        return 0.0
    return float(atr.iloc[-1] / close * 100.0)


def atr_ratio(m5: pd.DataFrame, lookback: int = 100, period: int = 14) -> float:
    atr = atr_series(m5, period).dropna()
    if len(atr) < 2:
        return 1.0
    tail = atr.iloc[-lookback:] if len(atr) >= lookback else atr
    mean = float(tail.mean())
    cur = float(atr.iloc[-1])
    if mean <= 0:
        return 1.0
    return cur / mean


def adx_series(m5: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = m5["high"], m5["low"], m5["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=m5.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=m5.index).rolling(period).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.rolling(period).mean()


def ema_cross_up(fast: pd.Series, slow: pd.Series) -> bool:
    if len(fast) < 2 or len(slow) < 2:
        return False
    return float(fast.iloc[-2]) <= float(slow.iloc[-2]) and float(fast.iloc[-1]) > float(slow.iloc[-1])


def ema_cross_down(fast: pd.Series, slow: pd.Series) -> bool:
    if len(fast) < 2 or len(slow) < 2:
        return False
    return float(fast.iloc[-2]) >= float(slow.iloc[-2]) and float(fast.iloc[-1]) < float(slow.iloc[-1])


def atr_expanding(m5: pd.DataFrame, bars: int = 5, period: int = 14) -> bool:
    atr = atr_series(m5, period).dropna()
    if len(atr) < bars + 1:
        return False
    tail = atr.iloc[-bars:]
    return bool((tail.diff().dropna() > 0).all())
