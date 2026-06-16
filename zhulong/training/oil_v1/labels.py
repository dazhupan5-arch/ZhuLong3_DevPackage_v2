"""USOIL v1 三分类标签：动态 ATR 阈值 + 90min 预测窗口。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HORIZON = 18  # 90 min @ M5
DEFAULT_GAIN_FIXED = 0.003  # 0.30%
ATR_MULT = 0.8


def _atr(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def generate_oil_triple_labels(
    m5: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    gain_fixed: float = DEFAULT_GAIN_FIXED,
    use_dynamic: bool = True,
    atr_mult: float = ATR_MULT,
) -> pd.DataFrame:
    close = m5["close"]
    ret = (close.shift(-horizon) - close) / close.replace(0, np.nan)

    if use_dynamic:
        atr = _atr(close, m5["high"], m5["low"], 14)
        gain = (atr_mult * atr / close).clip(lower=gain_fixed * 0.5, upper=gain_fixed * 2.0)
        gain = gain.fillna(gain_fixed)
    else:
        gain = gain_fixed

    labels = np.zeros(len(m5), dtype=np.int8)
    labels[ret > gain] = 1
    labels[ret < -gain] = 2
    out = pd.DataFrame({"label": labels, "gain_threshold": gain}, index=m5.index)
    n = max(int((~ret.isna()).sum()), 1)
    logger.info(
        "oil triple h=%s dynamic=%s long=%.1f%% short=%.1f%% flat=%.1f%% avg_gain=%.3f%%",
        horizon,
        use_dynamic,
        100 * (labels == 1).sum() / n,
        100 * (labels == 2).sum() / n,
        100 * (labels == 0).sum() / n,
        100 * float(gain.mean()),
    )
    return out
