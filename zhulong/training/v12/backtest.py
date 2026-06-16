"""v12 双向优化回测（不对称阈值/止损/冷却 + 做空趋势过滤）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zhulong.training.lgb.backtest import TP_ATR, _atr_series, max_drawdown_r, simulate_trade
from zhulong.training.v10.backtest import MIN_ATR_PCT
from zhulong.training.v11.train import proba_to_directions

V12_LONG_THR = 0.84
V12_SHORT_THR = 0.88
V12_MAX_HOLD = 12
V12_LONG_COOLDOWN = 18   # 90 min
V12_SHORT_COOLDOWN = 24  # 120 min
V12_MAX_DAILY = 10
V12_LONG_SL = 1.2
V12_SHORT_SL = 1.0
V12_BREAKEVEN_ATR = 0.8
V12_USE_BREAKEVEN = False  # M5 上同 bar 保本易误触，默认关闭


def h1_bearish_series(m5: pd.DataFrame) -> pd.Series:
    """H1: close < EMA20 < EMA50。"""
    h1 = m5["close"].resample("1h", label="right", closed="right").last().dropna()
    ema20 = h1.ewm(span=20, adjust=False).mean()
    ema50 = h1.ewm(span=50, adjust=False).mean()
    bear = ((h1 < ema20) & (ema20 < ema50)).astype(np.int8)
    return bear.reindex(m5.index, method="ffill").fillna(0)


def apply_short_trend_filter(
    m5: pd.DataFrame,
    feats: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
) -> np.ndarray:
    """做空需 H1 空头排列且 h1_rsi < 50（特征中为 0.5）。"""
    bear = h1_bearish_series(m5)
    out = directions.copy()
    for i, t in enumerate(times):
        if out[i] != -1 or t not in m5.index:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            continue
        rsi = float(feats.loc[t, "h1_rsi"]) if t in feats.index and "h1_rsi" in feats.columns else 1.0
        if bear.iloc[idx] < 1 or rsi >= 0.5:
            out[i] = 0
    return out


def simulate_trade_v12(
    direction: int,
    entry: float,
    atr: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    max_bars: int = V12_MAX_HOLD,
    sl_mult: float = V12_LONG_SL,
) -> float:
    """R 倍数；保本止损在 K 线收盘后生效（避免同 bar 内假触发）。"""
    risk = sl_mult * atr
    if risk <= 0 or len(highs) < 1:
        return 0.0
    bars = min(len(highs), max_bars)
    hs, ls, cs = highs[:bars], lows[:bars], closes[:bars]
    tp_mult = TP_ATR
    win_r = tp_mult / sl_mult

    if direction > 0:
        sl = entry - risk
        tp = entry + tp_mult * atr
        peak = 0.0
        for h, l, c in zip(hs, ls, cs):
            hit_sl, hit_tp = l <= sl, h >= tp
            if hit_sl and hit_tp:
                return -1.0
            if hit_sl:
                return -1.0 if sl < entry else 0.0
            if hit_tp:
                return win_r
            peak = max(peak, h - entry)
            if peak >= V12_BREAKEVEN_ATR * atr:
                sl = max(sl, entry)
        return float((cs[-1] - entry) / risk)

    sl = entry + risk
    tp = entry - tp_mult * atr
    peak = 0.0
    for h, l, c in zip(hs, ls, cs):
        hit_sl, hit_tp = h >= sl, l <= tp
        if hit_sl and hit_tp:
            return -1.0
        if hit_sl:
            return -1.0 if sl > entry else 0.0
        if hit_tp:
            return win_r
        peak = max(peak, entry - l)
        if peak >= V12_BREAKEVEN_ATR * atr:
            sl = min(sl, entry)
    return float((entry - cs[-1]) / risk)


def backtest_v12(
    m5: pd.DataFrame,
    feats: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
    max_hold: int = V12_MAX_HOLD,
    long_cooldown: int = V12_LONG_COOLDOWN,
    short_cooldown: int = V12_SHORT_COOLDOWN,
    max_daily_signals: int = V12_MAX_DAILY,
) -> dict[str, float]:
    atr = _atr_series(m5)
    close = m5["close"]
    dirs = apply_short_trend_filter(m5, feats, times, directions)

    rs: list[float] = []
    sides: list[int] = []
    trade_times: list[pd.Timestamp] = []
    last_long_idx = -10**9
    last_short_idx = -10**9
    daily_count: dict[object, int] = {}

    for t, d in zip(times, dirs):
        if d == 0 or t not in m5.index:
            continue
        day = t.date()
        if daily_count.get(day, 0) >= max_daily_signals:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            continue
        if d > 0 and (idx - last_long_idx) < long_cooldown:
            continue
        if d < 0 and (idx - last_short_idx) < short_cooldown:
            continue
        if idx + 1 >= len(m5):
            continue
        a = float(atr.iloc[idx])
        c = float(close.iloc[idx])
        if a <= 0 or (a / c) < MIN_ATR_PCT:
            continue

        sl_mult = V12_LONG_SL if d > 0 else V12_SHORT_SL
        entry = c
        end = min(idx + 1 + max_hold, len(m5))
        hs = m5["high"].iloc[idx + 1 : end].to_numpy()
        ls = m5["low"].iloc[idx + 1 : end].to_numpy()
        cs = m5["close"].iloc[idx + 1 : end].to_numpy()
        if V12_USE_BREAKEVEN:
            r = simulate_trade_v12(int(d), entry, a, hs, ls, cs, max_hold, sl_mult)
        else:
            r = simulate_trade(int(d), entry, a, hs, ls, cs, max_hold, sl_mult=sl_mult, tp_mult=TP_ATR)
        rs.append(r)
        sides.append(int(d))
        trade_times.append(t)
        daily_count[day] = daily_count.get(day, 0) + 1
        if d > 0:
            last_long_idx = idx
        else:
            last_short_idx = idx

    if not rs:
        return _empty()

    rs_arr = np.array(rs)
    sides_arr = np.array(sides)
    stats = _stats(rs_arr, trade_times)
    stats["n_long"] = int((sides_arr > 0).sum())
    stats["n_short"] = int((sides_arr < 0).sum())
    long_rs = rs_arr[sides_arr > 0]
    short_rs = rs_arr[sides_arr < 0]
    stats["long_win_rate"] = float((long_rs > 0).mean()) if len(long_rs) else 0.0
    stats["short_win_rate"] = float((short_rs > 0).mean()) if len(short_rs) else 0.0
    return stats


def _empty() -> dict[str, float]:
    return {
        "win_rate": 0.0, "avg_rr": 0.0, "expectancy": -1.0, "n_trades": 0,
        "n_long": 0, "n_short": 0, "long_win_rate": 0.0, "short_win_rate": 0.0,
        "max_daily_signals": 0, "max_drawdown": 0.0, "total_pnl_r": 0.0,
    }


def _stats(rs_arr: np.ndarray, trade_times: list) -> dict[str, float]:
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
        "max_drawdown": max_drawdown_r(rs_arr.tolist()),
        "total_pnl_r": float(rs_arr.sum()),
    }


def val_weighted_precision(
    proba: np.ndarray,
    y_true: np.ndarray,
    times: pd.DatetimeIndex,
    m5: pd.DataFrame,
    feats: pd.DataFrame,
    long_thr: float = V12_LONG_THR,
    short_thr: float = V12_SHORT_THR,
) -> dict[str, float]:
    dirs = proba_to_directions(proba, long_thr, short_thr)
    dirs = apply_short_trend_filter(m5, feats, times, dirs)
    long_m = dirs == 1
    short_m = dirs == -1
    lp = float((y_true[long_m] == 1).mean()) if long_m.any() else 0.0
    sp = float((y_true[short_m] == 2).mean()) if short_m.any() else 0.0
    n = int(long_m.sum() + short_m.sum())
    wprec = (lp * long_m.sum() + sp * short_m.sum()) / max(n, 1)
    days = max((times.max() - times.min()).total_seconds() / 86400.0, 1.0)
    return {
        "precision": wprec,
        "long_precision": lp,
        "short_precision": sp,
        "n_signals": n,
        "signals_per_day": n / days,
    }
