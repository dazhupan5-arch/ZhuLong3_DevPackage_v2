"""v6 盈亏标签：模拟做多 SL/TP 交易结果（与回测规则一致）。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from zhulong.training.lgb.backtest import SL_ATR, TP_ATR, _atr_series, simulate_trade

logger = logging.getLogger(__name__)

DEFAULT_MAX_HOLD_BARS = 12
WIN_R = TP_ATR / SL_ATR


def generate_profit_labels(
    m5: pd.DataFrame,
    atr_period: int = 14,
    sl_mult: float = SL_ATR,
    tp_mult: float = TP_ATR,
    max_hold_bars: int = DEFAULT_MAX_HOLD_BARS,
) -> pd.DataFrame:
    """
    每个 bar 模拟做多：1=在 max_hold 内先触 TP，0=触 SL/超时/未达标。
    与 backtest.simulate_trade 一致（同 bar 双触视为亏）。
    """
    atr = _atr_series(m5, atr_period)
    close = m5["close"].values
    high = m5["high"].values
    low = m5["low"].values
    n = len(m5)
    labels = np.zeros(n, dtype=np.int8)
    win_r = tp_mult / sl_mult

    for i in range(n - max_hold_bars):
        a = float(atr.iloc[i])
        if a <= 0 or np.isnan(a):
            continue
        entry = float(close[i])
        end = i + 1 + max_hold_bars
        hs = high[i + 1 : end]
        ls = low[i + 1 : end]
        if len(hs) < 1:
            continue
        r = simulate_trade(
            1, entry, a, hs, ls, closes=None, max_bars=max_hold_bars,
            sl_mult=sl_mult, tp_mult=tp_mult,
        )
        if r >= win_r - 1e-9:
            labels[i] = 1

    out = pd.DataFrame({"label": labels}, index=m5.index)
    valid = labels[: n - max_hold_bars]
    pos = int(valid.sum())
    logger.info(
        "profit labels max_hold=%s sl=%.1fx tp=%.1fx win=%s (%.2f%%) loss=%s (%.2f%%)",
        max_hold_bars,
        sl_mult,
        tp_mult,
        pos,
        100.0 * pos / max(len(valid), 1),
        len(valid) - pos,
        100.0 * (len(valid) - pos) / max(len(valid), 1),
    )
    return out
