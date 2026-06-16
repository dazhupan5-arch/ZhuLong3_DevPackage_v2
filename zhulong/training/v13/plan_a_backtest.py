"""方案 A 回测：移动止损 + 分批止盈 + 统一 SL/TP。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from zhulong.training.lgb.backtest import _atr_series, max_drawdown_r


@dataclass
class PlanAPositionConfig:
    sl_mult: float = 1.2
    tp_mult: float = 2.0
    max_hold: int = 12
    trailing_enabled: bool = True
    trailing_activation_pct: float = 0.15
    trailing_step_pct: float = 0.10
    trailing_tighten: float = 0.8
    partial_enabled: bool = True
    partial_target1_pct: float = 0.25
    partial_ratio1: float = 0.5
    partial_target2_pct: float = 0.40
    partial_ratio2: float = 0.5
    profit_drawdown_ratio: float = 0.4


def _profit_pct(direction: int, entry: float, price: float) -> float:
    if direction > 0:
        return (price - entry) / entry * 100.0
    return (entry - price) / entry * 100.0


def simulate_trade_plan_a(
    direction: int,
    entry: float,
    atr: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    cfg: PlanAPositionConfig,
) -> float:
    risk = cfg.sl_mult * atr
    if risk <= 0 or len(highs) == 0:
        return 0.0

    win_r = cfg.tp_mult / cfg.sl_mult
    if direction > 0:
        sl, tp = entry - risk, entry + cfg.tp_mult * atr
    else:
        sl, tp = entry + risk, entry - cfg.tp_mult * atr

    volume = 1.0
    realized_r = 0.0
    peak_profit = 0.0
    trailing_on = False
    last_trail_price = entry
    partial1 = partial2 = False
    bars = min(len(highs), cfg.max_hold)

    for i in range(bars):
        h, l, c = float(highs[i]), float(lows[i]), float(closes[i])
        mark = c
        profit = _profit_pct(direction, entry, mark)
        peak_profit = max(peak_profit, profit)

        if cfg.partial_enabled and not partial1 and profit >= cfg.partial_target1_pct:
            realized_r += win_r * cfg.partial_ratio1 * volume
            volume *= 1.0 - cfg.partial_ratio1
            partial1 = True
        elif cfg.partial_enabled and partial1 and not partial2 and profit >= cfg.partial_target2_pct:
            realized_r += win_r * cfg.partial_ratio2 * volume
            volume *= 1.0 - cfg.partial_ratio2
            partial2 = True

        if cfg.trailing_enabled and profit >= cfg.trailing_activation_pct and not trailing_on:
            sl = entry
            trailing_on = True
            last_trail_price = entry
        elif trailing_on:
            move = (mark - last_trail_price) * direction
            step = entry * cfg.trailing_step_pct / 100.0
            if move >= step:
                tighten = step * cfg.trailing_tighten
                sl = sl + tighten * direction if direction > 0 else sl - tighten
                last_trail_price = mark

        if peak_profit > 0 and profit < peak_profit * (1.0 - cfg.profit_drawdown_ratio):
            if volume > 0:
                pnl = (mark - entry) * direction / risk
                realized_r += pnl * volume
            return float(realized_r)

        if direction > 0:
            hit_sl, hit_tp = l <= sl, h >= tp
        else:
            hit_sl, hit_tp = h >= sl, l <= tp

        if hit_sl and hit_tp:
            if volume > 0:
                realized_r -= volume
            return float(realized_r)
        if hit_sl:
            if volume > 0:
                pnl = (sl - entry) * direction / risk
                realized_r += pnl * volume
            return float(realized_r)
        if hit_tp:
            if volume > 0:
                realized_r += win_r * volume
            return float(realized_r)

    if volume > 0:
        pnl = (float(closes[bars - 1]) - entry) * direction / risk
        realized_r += pnl * volume
    return float(realized_r)


def backtest_plan_a(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
    cfg: PlanAPositionConfig,
    *,
    cooldown_bars: int = 6,
    max_daily_signals: int = 8,
    min_atr_pct: float = 0.001,
) -> dict[str, float]:
    atr = _atr_series(m5)
    close = m5["close"]
    rs: list[float] = []
    trade_times: list[pd.Timestamp] = []
    last_long = last_short = -10**9
    daily: dict[object, int] = {}

    for t, d in zip(times, directions):
        if d == 0 or t not in m5.index:
            continue
        day = t.date()
        if daily.get(day, 0) >= max_daily_signals:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            continue
        if d > 0 and (idx - last_long) < cooldown_bars:
            continue
        if d < 0 and (idx - last_short) < cooldown_bars:
            continue
        if idx + 1 >= len(m5):
            continue
        a = float(atr.iloc[idx])
        c = float(close.iloc[idx])
        if a <= 0 or (a / c) < min_atr_pct:
            continue

        entry = c
        end = min(idx + 1 + cfg.max_hold, len(m5))
        hs = m5["high"].iloc[idx + 1 : end].to_numpy()
        ls = m5["low"].iloc[idx + 1 : end].to_numpy()
        cs = m5["close"].iloc[idx + 1 : end].to_numpy()
        r = simulate_trade_plan_a(int(d), entry, a, hs, ls, cs, cfg)
        rs.append(r)
        trade_times.append(t)
        daily[day] = daily.get(day, 0) + 1
        if d > 0:
            last_long = idx
        else:
            last_short = idx

    if not rs:
        return {
            "win_rate": 0.0,
            "avg_rr": 0.0,
            "expectancy": -1.0,
            "n_trades": 0,
            "max_daily_signals": 0,
            "max_drawdown": 0.0,
            "total_pnl_r": 0.0,
            "signals_per_day": 0.0,
        }

    arr = np.array(rs)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 1.0
    days = max((times.max() - times.min()).total_seconds() / 86400.0, 1.0)
    daily_s = pd.Series(1, index=trade_times).groupby(pd.DatetimeIndex(trade_times).date).sum()
    return {
        "win_rate": float((arr > 0).mean()),
        "avg_rr": avg_win / avg_loss if avg_loss > 0 else 0.0,
        "expectancy": float((arr > 0).mean() * avg_win - (1 - (arr > 0).mean()) * avg_loss),
        "n_trades": int(len(arr)),
        "max_daily_signals": int(daily_s.max()) if len(daily_s) else 0,
        "max_drawdown": max_drawdown_r(arr.tolist()),
        "total_pnl_r": float(arr.sum()),
        "signals_per_day": len(arr) / days,
    }
