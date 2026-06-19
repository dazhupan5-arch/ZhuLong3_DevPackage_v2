"""固定 SL/TP 回测（1.2×ATR / 2.0×ATR），v4.2 最长持仓 4 小时。"""

from __future__ import annotations

import numpy as np
import pandas as pd

SL_ATR = 1.2
TP_ATR = 2.0
MAX_HOLD_BARS = 48  # 240 分钟 @ M5
DEFAULT_COOLDOWN_BARS = 12  # 60 分钟 @ M5

# XAUUSD 默认交易成本（价格点；commission 美元/手）
DEFAULT_SLIPPAGE_POINTS = 0.3
DEFAULT_SPREAD_POINTS = 0.5
DEFAULT_COMMISSION_PER_LOT = 7.0
DEFAULT_CONTRACT_SIZE = 100.0


def trade_cost_r(
    entry: float,
    atr: float,
    sl_mult: float = SL_ATR,
    *,
    slippage_points: float = 0.0,
    spread_points: float = 0.0,
    commission_per_lot: float = 0.0,
    contract_size: float = DEFAULT_CONTRACT_SIZE,
) -> float:
    """将滑点/点差/手续费折算为 R 倍数成本。"""
    risk = sl_mult * atr
    if risk <= 0 or entry <= 0:
        return 0.0
    point_cost_r = (slippage_points + spread_points) / risk
    commission_r = commission_per_lot / max(risk * contract_size, 1e-9)
    return float(point_cost_r + commission_r)


def _atr_series(m5: pd.DataFrame, period: int = 14) -> pd.Series:
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


def simulate_trade(
    direction: int,
    entry: float,
    atr: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray | None = None,
    max_bars: int = MAX_HOLD_BARS,
    sl_mult: float = SL_ATR,
    tp_mult: float = TP_ATR,
    *,
    cost_r: float = 0.0,
) -> float:
    """返回 R 倍数；超时按最后一根收盘价平仓。"""
    risk = sl_mult * atr
    if risk <= 0:
        return 0.0
    bars = min(len(highs), max_bars)
    if bars <= 0:
        return 0.0
    hs, ls = highs[:bars], lows[:bars]
    cs = closes[:bars] if closes is not None else None
    win_r = tp_mult / sl_mult

    def _apply_cost(r: float) -> float:
        return float(r - cost_r)

    if direction > 0:
        sl, tp = entry - risk, entry + tp_mult * atr
        for i, (h, l) in enumerate(zip(hs, ls)):
            hit_sl, hit_tp = l <= sl, h >= tp
            if hit_sl and hit_tp:
                return _apply_cost(-1.0)
            if hit_sl:
                return _apply_cost(-1.0)
            if hit_tp:
                return _apply_cost(win_r)
        if cs is not None:
            return _apply_cost(float((cs[-1] - entry) / risk))
        return _apply_cost(0.0)

    sl, tp = entry + risk, entry - tp_mult * atr
    for i, (h, l) in enumerate(zip(hs, ls)):
        hit_sl, hit_tp = h >= sl, l <= tp
        if hit_sl and hit_tp:
            return _apply_cost(-1.0)
        if hit_sl:
            return _apply_cost(-1.0)
        if hit_tp:
            return _apply_cost(win_r)
    if cs is not None:
        return _apply_cost(float((entry - cs[-1]) / risk))
    return _apply_cost(0.0)


def backtest_signals(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
    max_hold: int = MAX_HOLD_BARS,
    cooldown_bars: int = 0,
    *,
    sl_mult: float = SL_ATR,
    tp_mult: float = TP_ATR,
    slippage_points: float = 0.0,
    spread_points: float = 0.0,
    commission_per_lot: float = 0.0,
    contract_size: float = DEFAULT_CONTRACT_SIZE,
) -> dict[str, float]:
    """directions: +1 long, -1 short, 0 flat；cooldown_bars 同方向最小间隔。"""
    atr = _atr_series(m5)
    close = m5["close"]
    rs: list[float] = []
    trade_times: list[pd.Timestamp] = []
    last_long_idx = -10**9
    last_short_idx = -10**9

    for t, d in zip(times, directions):
        if d == 0:
            continue
        if t not in m5.index:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            continue
        if d > 0 and cooldown_bars > 0 and (idx - last_long_idx) < cooldown_bars:
            continue
        if d < 0 and cooldown_bars > 0 and (idx - last_short_idx) < cooldown_bars:
            continue
        if idx + 1 >= len(m5):
            continue
        a = float(atr.iloc[idx])
        if a <= 0 or np.isnan(a):
            continue
        entry = float(close.iloc[idx])
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
        rs.append(
            simulate_trade(
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
        )
        trade_times.append(t)
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
    max_daily = int(daily.max()) if len(daily) else 0

    return {
        "win_rate": win_rate,
        "avg_rr": avg_rr,
        "expectancy": float(expectancy),
        "n_trades": int(len(rs_arr)),
        "max_daily_signals": max_daily,
        "max_drawdown": max_drawdown_r(rs),
        "total_pnl_r": float(rs_arr.sum()),
    }


def max_drawdown_r(rs: list[float]) -> float:
    if not rs:
        return 0.0
    eq = np.cumsum(rs)
    peak = np.maximum.accumulate(eq)
    dd_r = peak - eq
    # 以峰值或 10R 为分母，避免初期小权益导致 100% 回撤
    denom = max(float(np.max(np.abs(peak))), 10.0)
    return float(np.clip(dd_r.max() / denom, 0.0, 1.0))
