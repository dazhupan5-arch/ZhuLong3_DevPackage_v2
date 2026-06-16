"""市场状态识别与策略调度。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from zhulong.strategies.base import StrategyContext
from zhulong.strategies.indicators import adx_series, atr_expanding, ema


class MarketState(str, Enum):
    TREND = "TREND"
    VOLATILE = "VOLATILE"
    RANGE = "RANGE"


class StrategyStateMachine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        sm = self.config.get("state_machine") or {}
        self.primary_symbol = sm.get("primary_symbol", "XAUUSD")
        self.trend_strategy = sm.get("trend_strategy", "ai_model")
        self.volatile_strategy = sm.get("volatile_strategy", "spread_hedge")
        self.range_strategy = sm.get("range_strategy", "grid_system")
        self.atr_volatile_ratio = float(sm.get("atr_volatile_ratio", 1.5))
        self.adx_trend_min = float(sm.get("adx_trend_min", 25))
        self.adx_range_max = float(sm.get("adx_range_max", 20))
        self.atr_range_percentile = int(sm.get("atr_range_percentile", 20))

    def get_current_state(self, context: StrategyContext, symbol: str | None = None) -> MarketState:
        sym = symbol or self.primary_symbol
        if context.is_macro_silence():
            return MarketState.VOLATILE

        m5 = context.get_m5(sym)
        if m5 is None or len(m5) < 100:
            return MarketState.RANGE

        if context.get_atr_ratio(sym) >= self.atr_volatile_ratio:
            return MarketState.VOLATILE

        close = m5["close"]
        ema20 = float(ema(close, 20).iloc[-1])
        ema50 = float(ema(close, 50).iloc[-1])
        adx_val = context.get_adx(sym)

        if ema20 > ema50 and adx_val > self.adx_trend_min and atr_expanding(m5):
            return MarketState.TREND

        from zhulong.strategies.indicators import atr_series

        atr = atr_series(m5).dropna()
        if len(atr) >= 50:
            pct = float(atr.iloc[-1]) / float(atr.quantile(self.atr_range_percentile / 100.0).clip(min=1e-9))
            if pct <= 1.0 and adx_val < self.adx_range_max:
                return MarketState.RANGE

        if ema20 > ema50 and adx_val > self.adx_trend_min:
            return MarketState.TREND

        return MarketState.RANGE

    def select_strategy(self, state: MarketState) -> str:
        if state == MarketState.TREND:
            return self.trend_strategy
        if state == MarketState.VOLATILE:
            return self.volatile_strategy
        return self.range_strategy

    def describe(self, context: StrategyContext, symbol: str | None = None) -> dict[str, Any]:
        sym = symbol or self.primary_symbol
        state = self.get_current_state(context, sym)
        return {
            "state": state.value,
            "strategy": self.select_strategy(state),
            "symbol": sym,
            "atr_ratio": round(context.get_atr_ratio(sym), 3),
            "adx": round(context.get_adx(sym), 2),
            "macro_silence": context.is_macro_silence(),
        }
