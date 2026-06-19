"""v10 双向信号与回测（对称做空）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from zhulong.analysis.feature_engineering import _adx
from zhulong.training.lgb.backtest import _atr_series, max_drawdown_r, simulate_trade, trade_cost_r
from zhulong.training.v13.trade_sim import simulate_trade_trailing
from zhulong.training.v9.backtest import V9_COOLDOWN_BARS, V9_MAX_HOLD

logger = logging.getLogger(__name__)

MIN_ATR_PCT = 0.001  # 0.1%


def h1_trend_flags_v10(m5: pd.DataFrame) -> pd.DataFrame:
    """H1 趋势：做多多头排列；做空空头排列 + EMA20 向下。"""
    h1 = (
        m5["close"]
        .resample("1h", label="right", closed="right")
        .last()
        .dropna()
    )
    ema20 = h1.ewm(span=20, adjust=False).mean()
    ema50 = h1.ewm(span=50, adjust=False).mean()
    ema20_down = ema20.diff(3) < 0
    flags = pd.DataFrame(
        {
            "long_ok": ((h1 > ema20) & (ema20 > ema50)).astype(np.int8),
            "short_ok": ((h1 < ema20) & (ema20 < ema50) & ema20_down).astype(np.int8),
        },
        index=h1.index,
    )
    return flags.reindex(m5.index, method="ffill").fillna(0)


def build_directions(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    proba: np.ndarray,
    long_thr: float,
    short_thr: float,
    mode: str = "both",
) -> np.ndarray:
    """
    mode: 'long' | 'short' | 'both'
    做多: proba >= long_thr；做空: proba <= short_thr（对称 1-p 近似）
    """
    trend = h1_trend_flags_v10(m5)
    atr = _atr_series(m5)
    close = m5["close"]
    dirs = np.zeros(len(times), dtype=np.int8)

    for i, t in enumerate(times):
        if t not in m5.index:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            continue
        a = float(atr.iloc[idx])
        c = float(close.iloc[idx])
        if a <= 0 or c <= 0 or (a / c) < MIN_ATR_PCT:
            continue
        p = float(proba[i])
        row = trend.iloc[idx]
        if mode in ("long", "both") and p >= long_thr and row["long_ok"] >= 1:
            dirs[i] = 1
        elif mode in ("short", "both") and p <= short_thr and row["short_ok"] >= 1:
            dirs[i] = -1
    return dirs


def apply_adx_filter(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
    adx_min: float = 25.0,
) -> np.ndarray:
    adx = _adx(m5, 14)
    out = directions.copy()
    for i, t in enumerate(times):
        if out[i] == 0 or t not in m5.index:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            out[i] = 0
            continue
        if float(adx.iloc[idx]) < adx_min:
            out[i] = 0
    return out


def backtest_both(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
    max_hold: int = V9_MAX_HOLD,
    cooldown_bars: int = V9_COOLDOWN_BARS,
    max_daily_signals: int = 10,
    sl_mult: float = 1.2,
    tp_mult: float = 2.0,
    trailing: bool = False,
    adx_min: float | None = None,
    *,
    slippage_points: float = 0.0,
    spread_points: float = 0.0,
    commission_per_lot: float = 0.0,
    contract_size: float = 100.0,
) -> dict[str, float]:
    """双向回测（固定或移动止损 SL/TP）。"""
    dirs = directions.copy()
    if adx_min is not None:
        dirs = apply_adx_filter(m5, times, dirs, adx_min)

    atr = _atr_series(m5)
    close = m5["close"]

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
        if d > 0 and (idx - last_long_idx) < cooldown_bars:
            continue
        if d < 0 and (idx - last_short_idx) < cooldown_bars:
            continue
        if idx + 1 >= len(m5):
            continue
        a = float(atr.iloc[idx])
        c = float(close.iloc[idx])
        if a <= 0 or (a / c) < MIN_ATR_PCT:
            continue

        entry = c
        cost_r = trade_cost_r(
            entry,
            a,
            sl_mult,
            slippage_points=slippage_points,
            spread_points=spread_points,
            commission_per_lot=commission_per_lot,
            contract_size=contract_size,
        )
        end = min(idx + 1 + max_hold, len(m5))
        hs = m5["high"].iloc[idx + 1 : end].to_numpy()
        ls = m5["low"].iloc[idx + 1 : end].to_numpy()
        cs = m5["close"].iloc[idx + 1 : end].to_numpy()
        if trailing:
            sim = simulate_trade_trailing(
                int(d), entry, a, hs, ls, cs,
                max_bars=max_hold, sl_mult=sl_mult, tp_mult=tp_mult, trailing=True,
            )
            r = sim.r_multiple - cost_r
        else:
            r = simulate_trade(
                int(d),
                entry,
                a,
                hs,
                ls,
                cs,
                max_hold,
                sl_mult=sl_mult,
                tp_mult=tp_mult,
                cost_r=cost_r,
            )
        rs.append(r)
        sides.append(int(d))
        trade_times.append(t)
        daily_count[day] = daily_count.get(day, 0) + 1
        if d > 0:
            last_long_idx = idx
        else:
            last_short_idx = idx

    if not rs:
        return _empty_stats()

    rs_arr = np.array(rs)
    sides_arr = np.array(sides)
    stats = _aggregate_stats(rs_arr, trade_times)
    stats["n_long"] = int((sides_arr > 0).sum())
    stats["n_short"] = int((sides_arr < 0).sum())
    long_rs = rs_arr[sides_arr > 0]
    short_rs = rs_arr[sides_arr < 0]
    stats["long_win_rate"] = float((long_rs > 0).mean()) if len(long_rs) else 0.0
    stats["short_win_rate"] = float((short_rs > 0).mean()) if len(short_rs) else 0.0
    return stats


def _empty_stats() -> dict[str, float]:
    return {
        "win_rate": 0.0,
        "avg_rr": 0.0,
        "expectancy": -1.0,
        "n_trades": 0,
        "n_long": 0,
        "n_short": 0,
        "long_win_rate": 0.0,
        "short_win_rate": 0.0,
        "max_daily_signals": 0,
        "max_drawdown": 0.0,
        "total_pnl_r": 0.0,
    }


def _aggregate_stats(rs_arr: np.ndarray, trade_times: list) -> dict[str, float]:
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


@dataclass
class ShortThresholdResult:
    threshold: float
    precision: float
    n_signals: int
    signals_per_day: float


def short_ground_truth(m5: pd.DataFrame, times: pd.DatetimeIndex, horizon: int = 12, gain: float = 0.002) -> np.ndarray:
    close = m5["close"]
    fut = (close.shift(-horizon) - close) / close.replace(0, np.nan)
    y = np.zeros(len(times), dtype=int)
    for i, t in enumerate(times):
        if t in fut.index and not np.isnan(fut.loc[t]):
            y[i] = 1 if fut.loc[t] < -gain else 0
    return y


def tune_short_threshold(
    proba: np.ndarray,
    times: pd.DatetimeIndex,
    m5: pd.DataFrame,
    y_short: np.ndarray,
    lo: float = 0.15,
    hi: float = 0.35,
    step: float = 0.02,
    max_signals_per_day: float = 8.0,
) -> tuple[ShortThresholdResult, list[ShortThresholdResult]]:
    days = max((times.max() - times.min()).total_seconds() / 86400.0, 1.0)
    trend = h1_trend_flags_v10(m5)
    atr = _atr_series(m5)
    close = m5["close"]
    rows: list[ShortThresholdResult] = []

    for thr in np.arange(lo, hi + step * 0.5, step):
        pred = np.zeros(len(times), dtype=int)
        for i, t in enumerate(times):
            if t not in m5.index:
                continue
            idx = m5.index.get_loc(t)
            if isinstance(idx, slice):
                continue
            a, c = float(atr.iloc[idx]), float(close.iloc[idx])
            if a <= 0 or (a / c) < MIN_ATR_PCT:
                continue
            if proba[i] <= thr and trend.iloc[idx]["short_ok"] >= 1:
                pred[i] = 1
        n = int(pred.sum())
        if n == 0:
            rows.append(ShortThresholdResult(float(thr), 0.0, 0, 0.0))
            continue
        prec = float((y_short[pred == 1] == 1).mean())
        spd = n / days
        rows.append(ShortThresholdResult(float(round(thr, 4)), prec, n, spd))

    candidates = [r for r in rows if r.precision >= 0.45 and r.signals_per_day <= max_signals_per_day and r.n_signals > 0]
    if candidates:
        best = max(candidates, key=lambda r: (r.precision, -r.signals_per_day))
    elif rows:
        best = max(rows, key=lambda r: (r.precision, -r.signals_per_day))
    else:
        best = ShortThresholdResult(0.20, 0.0, 0, 0.0)
    return best, rows
