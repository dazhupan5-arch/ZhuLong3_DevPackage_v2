"""USOIL v1 回测：不对称阈值 + EIA 屏蔽 + 波动率自适应止损。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zhulong.training.lgb.backtest import TP_ATR, _atr_series, max_drawdown_r, simulate_trade
from zhulong.training.v10.backtest import MIN_ATR_PCT
from zhulong.training.v11.train import proba_to_directions
from zhulong.training.oil_v1.inventory import _eia_wednesdays

OIL_LONG_THR = 0.82
OIL_SHORT_THR = 0.78
OIL_MAX_HOLD = 18  # 90 min
OIL_COOLDOWN = 18  # 统一 90 min
OIL_MAX_DAILY = 8
OIL_LONG_SL = 1.5
OIL_SHORT_SL = 1.2
OIL_TP_ATR = 2.5
EIA_BLACKOUT_BEFORE_MIN = 30
EIA_BLACKOUT_AFTER_MIN = 15


def _eia_blackout_mask(m5_index: pd.DatetimeIndex) -> np.ndarray:
    """EIA 公布前 30min 至后 15min 屏蔽信号。"""
    from zhulong.utils.time_index import normalize_datetime_index

    m5_index = normalize_datetime_index(m5_index)
    blocked = np.zeros(len(m5_index), dtype=bool)
    eia_times = _eia_wednesdays(m5_index.min(), m5_index.max() + pd.Timedelta(days=7))
    for eia_t in eia_times:
        start = eia_t - pd.Timedelta(minutes=EIA_BLACKOUT_BEFORE_MIN)
        end = eia_t + pd.Timedelta(minutes=EIA_BLACKOUT_AFTER_MIN)
        blocked |= (m5_index >= start) & (m5_index <= end)
    return blocked


def h1_extreme_trend_filter(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
) -> np.ndarray:
    """仅在极端趋势时过滤逆向信号（原油震荡市多空均可）。"""
    from zhulong.utils.time_index import normalize_datetime_index, normalize_m5_index

    m5 = normalize_m5_index(m5)
    times = normalize_datetime_index(times)
    h1 = m5["close"].resample("1h", label="right", closed="right").last().dropna()
    ema20 = h1.ewm(span=20, adjust=False).mean()
    ema50 = h1.ewm(span=50, adjust=False).mean()
    bias = (h1 - ema50) / ema50.replace(0, np.nan)
    strong_bull = (bias > 0.02) & (ema20 > ema50)
    strong_bear = (bias < -0.02) & (ema20 < ema50)
    bull_m5 = strong_bull.reindex(m5.index, method="ffill").fillna(False)
    bear_m5 = strong_bear.reindex(m5.index, method="ffill").fillna(False)

    out = directions.copy()
    for i, t in enumerate(times):
        if out[i] == 0 or t not in m5.index:
            continue
        if out[i] < 0 and bull_m5.loc[t]:
            out[i] = 0
        elif out[i] > 0 and bear_m5.loc[t]:
            out[i] = 0
    return out


def backtest_oil_v1(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
    max_hold: int = OIL_MAX_HOLD,
    cooldown: int = OIL_COOLDOWN,
    max_daily_signals: int = OIL_MAX_DAILY,
    long_sl: float = OIL_LONG_SL,
    short_sl: float = OIL_SHORT_SL,
    tp_atr: float = OIL_TP_ATR,
) -> dict[str, float]:
    atr = _atr_series(m5)
    close = m5["close"]
    dirs = h1_extreme_trend_filter(m5, times, directions)
    blackout = _eia_blackout_mask(times)

    rs: list[float] = []
    sides: list[int] = []
    trade_times: list[pd.Timestamp] = []
    last_idx = -10**9
    daily_count: dict[object, int] = {}

    for i, (t, d) in enumerate(zip(times, dirs)):
        if d == 0 or t not in m5.index or blackout[i]:
            continue
        day = t.date()
        if daily_count.get(day, 0) >= max_daily_signals:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            continue
        if (idx - last_idx) < cooldown:
            continue
        if idx + 1 >= len(m5):
            continue
        a = float(atr.iloc[idx])
        c = float(close.iloc[idx])
        if a <= 0 or (a / c) < MIN_ATR_PCT:
            continue

        sl_mult = long_sl if d > 0 else short_sl
        end = min(idx + 1 + max_hold, len(m5))
        hs = m5["high"].iloc[idx + 1 : end].to_numpy()
        ls = m5["low"].iloc[idx + 1 : end].to_numpy()
        cs = m5["close"].iloc[idx + 1 : end].to_numpy()
        r = simulate_trade(int(d), c, a, hs, ls, cs, max_hold, sl_mult=sl_mult, tp_mult=tp_atr)
        rs.append(r)
        sides.append(int(d))
        trade_times.append(t)
        daily_count[day] = daily_count.get(day, 0) + 1
        last_idx = idx

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


def val_weighted_precision_oil(
    proba: np.ndarray,
    y_true: np.ndarray,
    times: pd.DatetimeIndex,
    m5: pd.DataFrame,
    long_thr: float = OIL_LONG_THR,
    short_thr: float = OIL_SHORT_THR,
) -> dict[str, float]:
    dirs = proba_to_directions(proba, long_thr, short_thr)
    dirs = h1_extreme_trend_filter(m5, times, dirs)
    blackout = _eia_blackout_mask(times)
    dirs[blackout] = 0
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
