"""V15 标签：V14 前向收益 + Regime 过滤（兼顾 OOS 与 crash 日）。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from zhulong.training.lgb.labels import generate_direction_labels

logger = logging.getLogger(__name__)

V15_HORIZON = 12
V15_GAIN = 0.0020
V15_REGIME_SMA = 200
V15_DEEP_ATR = 2.0
V15_STRESS_RET_12 = -0.003
V15_STRESS_DAY_RET = -0.01


def _atr_values(m5: pd.DataFrame) -> np.ndarray:
    prev = m5["close"].shift(1)
    tr = pd.concat(
        [
            m5["high"] - m5["low"],
            (m5["high"] - prev).abs(),
            (m5["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean().bfill()
    close = m5["close"].replace(0, np.nan)
    return np.maximum(atr.values.astype(np.float64), close.values * 0.0005)


def _regime_flags(close: np.ndarray, atr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sma200 = pd.Series(close).rolling(V15_REGIME_SMA, min_periods=50).mean().values
    d = np.zeros(len(close), dtype=np.float64)
    valid = (sma200 > 0) & (atr > 0)
    d[valid] = (close[valid] - sma200[valid]) / atr[valid]
    return d > V15_DEEP_ATR, d < -V15_DEEP_ATR


def _stress_mask(close: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    n = len(close)
    ret12 = np.zeros(n, dtype=np.float64)
    if n > V15_HORIZON:
        ret12[: n - V15_HORIZON] = (
            close[V15_HORIZON:] - close[: n - V15_HORIZON]
        ) / np.maximum(close[: n - V15_HORIZON], 1e-9)
    day = pd.Series(close, index=index).groupby(index.normalize()).transform("first").values
    day_ret = (close - day) / np.maximum(day, 1e-9)
    return (ret12 <= V15_STRESS_RET_12) & (day_ret <= V15_STRESS_DAY_RET)


def generate_labels_v15(m5: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    V14 前向 12 根 ±0.2% 标签 + 轻量 regime 过滤。
    stress 仅用于训练样本加权，不改写标签。
    """
    close = m5["close"].values.astype(np.float64)
    atr = _atr_values(m5)
    deep_bull, deep_bear = _regime_flags(close, atr)

    labels = generate_direction_labels(
        m5, horizon=V15_HORIZON, gain_threshold=V15_GAIN
    ).copy()

    for i in range(len(labels)):
        if labels[i] == 1 and deep_bear[i]:
            labels[i] = 0
        elif labels[i] == -1 and deep_bull[i]:
            labels[i] = 0

    stress = _stress_mask(close, m5.index)
    counts = {int(v): int((labels == v).sum()) for v in (-1, 0, 1)}
    logger.info(
        "V15 hybrid labels: short=%d long=%d flat=%d stress=%d",
        counts[-1],
        counts[1],
        counts[0],
        int(stress.sum()),
    )
    return labels, stress


def to_multiclass_v15(labels: np.ndarray) -> np.ndarray:
    out = np.zeros(len(labels), dtype=np.int8)
    out[labels == 1] = 1
    out[labels == -1] = 2
    return out
