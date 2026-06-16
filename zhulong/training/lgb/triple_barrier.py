"""三重屏障标签：学习固定 SL/TP 下的交易机会。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from zhulong.training.lgb.backtest import _atr_series

logger = logging.getLogger(__name__)

DEFAULT_SL_MULT = 1.2
DEFAULT_TP_MULT = 1.8
DEFAULT_MAX_HOLD = 12
DEFAULT_MIN_ATR_PCT = 0.001


def _simulate_barrier(
    direction: int,
    entry: float,
    atr: float,
    highs: np.ndarray,
    lows: np.ndarray,
    sl_mult: float,
    tp_mult: float,
) -> int:
    """返回 1=先止盈, -1=先止损/双触, 0=超时未止盈。"""
    if direction > 0:
        sl, tp = entry - sl_mult * atr, entry + tp_mult * atr
        for h, l in zip(highs, lows):
            hit_sl, hit_tp = l <= sl, h >= tp
            if hit_sl and hit_tp:
                return -1
            if hit_sl:
                return -1
            if hit_tp:
                return 1
        return 0
    sl, tp = entry + sl_mult * atr, entry - tp_mult * atr
    for h, l in zip(highs, lows):
        hit_sl, hit_tp = h >= sl, l <= tp
        if hit_sl and hit_tp:
            return -1
        if hit_sl:
            return -1
        if hit_tp:
            return 1
    return 0


def generate_triple_barrier_labels(
    m5: pd.DataFrame,
    sl_mult: float = DEFAULT_SL_MULT,
    tp_mult: float = DEFAULT_TP_MULT,
    max_hold: int = DEFAULT_MAX_HOLD,
    min_atr_pct: float = DEFAULT_MIN_ATR_PCT,
    trend_filter: bool = False,
) -> pd.DataFrame:
    """
    三分类标签：0=观望, 1=做多机会, 2=做空机会。
    ATR/close < min_atr_pct 的样本 label 为 NaN（训练时丢弃）。
    """
    atr = _atr_series(m5).to_numpy()
    close = m5["close"].to_numpy()
    high = m5["high"].to_numpy()
    low = m5["low"].to_numpy()
    n = len(m5)
    labels = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    h1_close = None
    h1_ema50 = None
    if trend_filter:
        h1 = m5["close"].resample("1h", label="right", closed="right").last().dropna()
        h1_ema50 = h1.ewm(span=50, adjust=False).mean()
        h1_close = h1

    for i in range(n - max_hold):
        if i > 0 and i % 50000 == 0:
            logger.info("triple_barrier progress %s/%s", i, n - max_hold)
        a = atr[i]
        c = close[i]
        if np.isnan(a) or a <= 0 or c <= 0:
            continue
        if (a / c) < min_atr_pct:
            continue
        valid[i] = True

        if trend_filter and h1_close is not None:
            ts = m5.index[i]
            h1_ix = h1_close.index.searchsorted(ts, side="right") - 1
            if h1_ix < 0:
                continue
            hc = float(h1_close.iloc[h1_ix])
            he = float(h1_ema50.iloc[h1_ix])
            long_ok = hc > he
            short_ok = hc < he
        else:
            long_ok = short_ok = True

        hs = high[i + 1 : i + 1 + max_hold]
        ls = low[i + 1 : i + 1 + max_hold]
        long_out = _simulate_barrier(1, c, a, hs, ls, sl_mult, tp_mult) if long_ok else 0
        short_out = _simulate_barrier(-1, c, a, hs, ls, sl_mult, tp_mult) if short_ok else 0

        if long_out == 1 and short_out != 1:
            labels[i] = 1
        elif short_out == 1 and long_out != 1:
            labels[i] = 2
        else:
            labels[i] = 0

    out = pd.DataFrame({"label": labels, "valid": valid}, index=m5.index)
    labeled = out["label"].dropna()
    n_lab = max(len(labeled), 1)
    logger.info(
        "triple_barrier sl=%.1f tp=%.1f hold=%s min_atr=%.3f%% long=%.1f%% short=%.1f%% flat=%.1f%% valid=%s",
        sl_mult,
        tp_mult,
        max_hold,
        min_atr_pct * 100,
        100 * (labeled == 1).sum() / n_lab,
        100 * (labeled == 2).sum() / n_lab,
        100 * (labeled == 0).sum() / n_lab,
        int(valid.sum()),
    )
    return out
