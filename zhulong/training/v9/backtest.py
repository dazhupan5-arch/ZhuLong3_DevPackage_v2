"""v9 回测：移动止损、利润回撤保护、冷却与日限。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zhulong.training.lgb.backtest import SL_ATR, TP_ATR, _atr_series, max_drawdown_r, simulate_trade

V9_COOLDOWN_BARS = 18  # 90 分钟 @ M5
V9_MAX_HOLD = 24
V9_MAX_DAILY_SIGNALS = 10


def simulate_trade_v9(
    direction: int,
    entry: float,
    atr: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    max_bars: int = V9_MAX_HOLD,
    sl_mult: float = SL_ATR,
    tp_mult: float = TP_ATR,
) -> float:
    """移动止损 + 40% 浮盈回撤保护；返回 R 倍数。"""
    risk = sl_mult * atr
    if risk <= 0 or len(highs) < 1:
        return 0.0
    bars = min(len(highs), max_bars)
    hs, ls, cs = highs[:bars], lows[:bars], closes[:bars]
    win_r = tp_mult / sl_mult

    if direction > 0:
        sl = entry - risk
        tp = entry + tp_mult * atr
        peak_profit = 0.0
        for h, l, c in zip(hs, ls, cs):
            peak_profit = max(peak_profit, h - entry)
            if peak_profit >= 1.0 * atr:
                sl = max(sl, entry)
            if peak_profit >= 2.0 * atr:
                sl = max(sl, entry + (peak_profit - 2.0 * atr) * 0.6)
            cur_profit = c - entry
            if peak_profit > 0 and (peak_profit - cur_profit) / peak_profit >= 0.40:
                return float(cur_profit / risk)
            hit_sl, hit_tp = l <= sl, h >= tp
            if hit_sl and hit_tp:
                return -1.0
            if hit_sl:
                return float((sl - entry) / risk)
            if hit_tp:
                return win_r
        return float((cs[-1] - entry) / risk)

    sl = entry + risk
    tp = entry - tp_mult * atr
    peak_profit = 0.0
    for h, l, c in zip(hs, ls, cs):
        float_profit = entry - l
        peak_profit = max(peak_profit, float_profit)
        if float_profit >= atr:
            sl = min(sl, entry)
        if peak_profit >= 2.0 * atr:
            sl = min(sl, entry - (peak_profit - 2.0 * atr) * 0.6)
        if peak_profit > 0 and (peak_profit - (entry - c)) / peak_profit >= 0.40:
            return float((entry - c) / risk)
        hit_sl, hit_tp = h >= sl, l <= tp
        if hit_sl and hit_tp:
            return -1.0
        if hit_sl:
            return float((entry - sl) / risk)
        if hit_tp:
            return win_r
    return float((entry - cs[-1]) / risk)


def h1_trend_flags(m5: pd.DataFrame) -> pd.DataFrame:
    """H1 EMA20/EMA50 趋势：long_ok / short_ok。"""
    h1 = (
        m5["close"]
        .resample("1h", label="right", closed="right")
        .last()
        .dropna()
    )
    ema20 = h1.ewm(span=20, adjust=False).mean()
    ema50 = h1.ewm(span=50, adjust=False).mean()
    flags = pd.DataFrame(
        {
            "long_ok": ((h1 > ema20) & (ema20 > ema50)).astype(np.int8),
            "short_ok": ((h1 < ema20) & (ema20 < ema50)).astype(np.int8),
        },
        index=h1.index,
    )
    return flags.reindex(m5.index, method="ffill").fillna(0)


def backtest_v9(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
    threshold: float = 0.5,
    max_hold: int = V9_MAX_HOLD,
    cooldown_bars: int = V9_COOLDOWN_BARS,
    max_daily_signals: int = V9_MAX_DAILY_SIGNALS,
    use_trend_filter: bool = True,
    use_trailing_stop: bool = True,
) -> dict[str, float]:
    atr = _atr_series(m5)
    close = m5["close"]
    trend = h1_trend_flags(m5) if use_trend_filter else None

    rs: list[float] = []
    trade_times: list[pd.Timestamp] = []
    last_long_idx = -10**9
    last_short_idx = -10**9
    daily_count: dict[object, int] = {}

    for t, d in zip(times, directions):
        if d == 0 or t not in m5.index:
            continue
        day = t.date()
        if daily_count.get(day, 0) >= max_daily_signals:
            continue

        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            continue
        if d > 0 and cooldown_bars > 0 and (idx - last_long_idx) < cooldown_bars:
            continue
        if d < 0 and cooldown_bars > 0 and (idx - last_short_idx) < cooldown_bars:
            continue
        if trend is not None:
            if d > 0 and trend.iloc[idx]["long_ok"] < 1:
                continue
            if d < 0 and trend.iloc[idx]["short_ok"] < 1:
                continue
        if idx + 1 >= len(m5):
            continue
        a = float(atr.iloc[idx])
        if a <= 0 or np.isnan(a):
            continue

        entry = float(close.iloc[idx])
        end = min(idx + 1 + max_hold, len(m5))
        hs = m5["high"].iloc[idx + 1 : end].to_numpy()
        ls = m5["low"].iloc[idx + 1 : end].to_numpy()
        cs = m5["close"].iloc[idx + 1 : end].to_numpy()
        r = (
            simulate_trade_v9(int(d), entry, a, hs, ls, cs, max_hold)
            if use_trailing_stop
            else simulate_trade(int(d), entry, a, hs, ls, cs, max_hold)
        )
        rs.append(r)
        trade_times.append(t)
        daily_count[day] = daily_count.get(day, 0) + 1
        if d > 0:
            last_long_idx = idx
        else:
            last_short_idx = idx

    if not rs:
        return {
            "win_rate": 0.0,
            "avg_rr": 0.0,
            "expectancy": -1.0,
            "n_trades": 0,
            "max_daily_signals": 0,
            "max_drawdown": 0.0,
            "total_pnl_r": 0.0,
        }

    rs_arr = np.array(rs)
    wins = rs_arr[rs_arr > 0]
    losses = rs_arr[rs_arr < 0]
    win_rate = float((rs_arr > 0).mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 1.0
    avg_rr = avg_win / avg_loss if avg_loss > 0 else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    daily = pd.Series(1, index=trade_times).groupby(pd.DatetimeIndex(trade_times).date).sum()

    return {
        "win_rate": win_rate,
        "avg_rr": avg_rr,
        "expectancy": float(expectancy),
        "n_trades": int(len(rs_arr)),
        "max_daily_signals": int(daily.max()) if len(daily) else 0,
        "max_drawdown": max_drawdown_r(rs),
        "total_pnl_r": float(rs_arr.sum()),
    }
