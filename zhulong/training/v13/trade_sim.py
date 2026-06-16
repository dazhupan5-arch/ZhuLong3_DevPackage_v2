"""V13 交易模拟：移动止损 + 质量指标（R倍数、最大不利波动）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

META_QUALITY_MIN_R = 1.5
META_QUALITY_MAX_MAE_PCT = 0.05  # 相对回撤 < 5%


@dataclass
class TradeSimResult:
    r_multiple: float
    mae_pct: float  # 持仓期间最大不利波动（占入场价比例）
    mfe_r: float
    exit_reason: str


def simulate_trade_trailing(
    direction: int,
    entry: float,
    atr: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    max_bars: int = 12,
    sl_mult: float = 1.0,
    tp_mult: float = 2.0,
    trailing: bool = True,
    trailing_breakeven_atr: float = 1.0,
    trailing_tighten_atr: float = 1.5,
    trailing_step_atr: float = 0.5,
) -> TradeSimResult:
    """
    动态止损：
    - 初始 SL = sl_mult × ATR，TP = tp_mult × ATR
    - 浮盈 ≥ 1.0×ATR → 止损移至开仓价（保本）
    - 浮盈 ≥ 1.5×ATR → 止损上移至 best ± trailing_step_atr × ATR
    """
    risk = sl_mult * atr
    if risk <= 0 or entry <= 0:
        return TradeSimResult(0.0, 0.0, 0.0, "invalid")

    bars = min(len(highs), max_bars)
    if bars <= 0:
        return TradeSimResult(0.0, 0.0, 0.0, "no_bars")

    win_r = tp_mult / sl_mult
    best_price = entry
    max_adverse = 0.0
    max_favorable_r = 0.0

    if direction > 0:
        sl = entry - risk
        tp = entry + tp_mult * atr
        for i in range(bars):
            h, l, c = float(highs[i]), float(lows[i]), float(closes[i])
            best_price = max(best_price, h)
            adverse = max(0.0, entry - l)
            max_adverse = max(max_adverse, adverse)
            max_favorable_r = max(max_favorable_r, (best_price - entry) / risk)

            if trailing:
                fav_atr = (best_price - entry) / atr if atr > 0 else 0.0
                if fav_atr >= trailing_breakeven_atr:
                    sl = max(sl, entry)
                if fav_atr >= trailing_tighten_atr:
                    sl = max(sl, best_price - trailing_step_atr * atr)

            hit_sl, hit_tp = l <= sl, h >= tp
            if hit_sl and hit_tp:
                return TradeSimResult(-1.0, max_adverse / entry, max_favorable_r, "sl_tp_same")
            if hit_sl:
                pnl_r = (sl - entry) / risk
                return TradeSimResult(float(pnl_r), max_adverse / entry, max_favorable_r, "stop_loss")
            if hit_tp:
                return TradeSimResult(win_r, max_adverse / entry, max_favorable_r, "take_profit")

        pnl_r = (float(closes[bars - 1]) - entry) / risk
        return TradeSimResult(float(pnl_r), max_adverse / entry, max_favorable_r, "time_stop")

    sl = entry + risk
    tp = entry - tp_mult * atr
    for i in range(bars):
        h, l, c = float(highs[i]), float(lows[i]), float(closes[i])
        best_price = min(best_price, l)
        adverse = max(0.0, h - entry)
        max_adverse = max(max_adverse, adverse)
        max_favorable_r = max(max_favorable_r, (entry - best_price) / risk)

        if trailing:
            fav_atr = (entry - best_price) / atr if atr > 0 else 0.0
            if fav_atr >= trailing_breakeven_atr:
                sl = min(sl, entry)
            if fav_atr >= trailing_tighten_atr:
                sl = min(sl, best_price + trailing_step_atr * atr)

        hit_sl, hit_tp = h >= sl, l <= tp
        if hit_sl and hit_tp:
            return TradeSimResult(-1.0, max_adverse / entry, max_favorable_r, "sl_tp_same")
        if hit_sl:
            pnl_r = (entry - sl) / risk
            return TradeSimResult(float(pnl_r), max_adverse / entry, max_favorable_r, "stop_loss")
        if hit_tp:
            return TradeSimResult(win_r, max_adverse / entry, max_favorable_r, "take_profit")

    pnl_r = (entry - float(closes[bars - 1])) / risk
    return TradeSimResult(float(pnl_r), max_adverse / entry, max_favorable_r, "time_stop")


def is_quality_positive(
    r_multiple: float,
    mae_pct: float,
    *,
    min_r: float = META_QUALITY_MIN_R,
    max_mae_pct: float = META_QUALITY_MAX_MAE_PCT,
) -> bool:
    return r_multiple >= min_r and mae_pct < max_mae_pct
