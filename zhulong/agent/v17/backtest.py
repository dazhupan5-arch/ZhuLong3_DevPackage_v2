"""V17 含成本回测模型。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from zhulong.training.lgb.backtest import (
    DEFAULT_CONTRACT_SIZE,
    simulate_trade,
    trade_cost_r,
    _atr_series,
    max_drawdown_r,
)

COST_MODEL: dict[str, dict[str, float]] = {
    "XAUUSD": {
        "spread_points": 0.5,
        "slippage_market": 0.3,
        "slippage_limit": 0.05,
        "commission_per_lot": 7.0,
        "contract_size": 100.0,
    },
    "USOIL": {
        "spread_points": 0.03,
        "slippage_market": 0.02,
        "slippage_limit": 0.005,
        "commission_per_lot": 5.0,
        "contract_size": 1000.0,
    },
}


def resolve_cost(symbol: str, entry_mode: str = "immediate") -> dict[str, float]:
    base = dict(COST_MODEL.get(symbol.upper(), COST_MODEL["XAUUSD"]))
    slip_key = "slippage_limit" if entry_mode == "limit" else "slippage_market"
    return {
        "spread_points": base["spread_points"],
        "slippage_points": base[slip_key],
        "commission_per_lot": base["commission_per_lot"],
        "contract_size": base.get("contract_size", DEFAULT_CONTRACT_SIZE),
    }


def backtest_direction_signals(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
    entry_modes: np.ndarray | None = None,
    *,
    symbol: str = "XAUUSD",
    max_hold: int = 12,
    cooldown_bars: int = 3,
    sl_mult: float = 1.2,
    tp_mult: float = 2.0,
    with_cost: bool = True,
) -> dict[str, Any]:
    """DirectionScorer 方向序列 + 可选 entry_mode 的成本回测。"""
    atr = _atr_series(m5)
    close = m5["close"]
    rs_gross: list[float] = []
    rs_net: list[float] = []
    trade_times: list[pd.Timestamp] = []
    last_long_idx = -10**9
    last_short_idx = -10**9

    for j, (t, d) in enumerate(zip(times, directions)):
        if d == 0 or t not in m5.index:
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
        if a <= 0 or np.isnan(a):
            continue
        entry = float(close.iloc[idx])
        mode = "immediate"
        if entry_modes is not None and j < len(entry_modes):
            mode = str(entry_modes[j] or "immediate")
        cost_cfg = resolve_cost(symbol, mode)
        cost_r = (
            trade_cost_r(
                entry,
                a,
                sl_mult,
                slippage_points=cost_cfg["slippage_points"],
                spread_points=cost_cfg["spread_points"],
                commission_per_lot=cost_cfg["commission_per_lot"],
                contract_size=cost_cfg["contract_size"],
            )
            if with_cost
            else 0.0
        )
        end = min(idx + 1 + max_hold, len(m5))
        hs = m5["high"].iloc[idx + 1 : end].to_numpy()
        ls = m5["low"].iloc[idx + 1 : end].to_numpy()
        cs = m5["close"].iloc[idx + 1 : end].to_numpy()
        gross = simulate_trade(int(d), entry, a, hs, ls, cs, max_hold, sl_mult=sl_mult, tp_mult=tp_mult, cost_r=0.0)
        net = gross - cost_r if with_cost else gross
        rs_gross.append(gross)
        rs_net.append(net)
        trade_times.append(t)
        if d > 0:
            last_long_idx = idx
        else:
            last_short_idx = idx

    if not rs_net:
        return {
            "win_rate": 0.0,
            "win_rate_gross": 0.0,
            "win_rate_net": 0.0,
            "n_trades": 0,
            "total_pnl_r": 0.0,
            "max_drawdown": 0.0,
        }

    gross_arr = np.array(rs_gross)
    net_arr = np.array(rs_net)
    return {
        "win_rate": float((net_arr > 0).mean()),
        "win_rate_gross": float((gross_arr > 0).mean()),
        "win_rate_net": float((net_arr > 0).mean()),
        "n_trades": int(len(net_arr)),
        "total_pnl_r": float(net_arr.sum()),
        "max_drawdown": max_drawdown_r(net_arr.tolist()),
        "with_cost": with_cost,
        "symbol": symbol.upper(),
    }
