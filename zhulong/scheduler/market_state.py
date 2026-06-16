"""扩展状态机：含 MODEL_DEGRADED。"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from zhulong.strategies.base import StrategyContext


class SchedulerMarketState(str, Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"
    MODEL_DEGRADED = "MODEL_DEGRADED"
    UNKNOWN = "UNKNOWN"


class SchedulerContextProto(Protocol):
    def is_macro_silence(self) -> bool: ...
    def get_atr_ratio(self, symbol: str, lookback: int = 100) -> float: ...
    def get_adx(self, symbol: str, period: int = 14) -> float: ...
    def get_recent_winrate(self, symbol: str, window: int = 20) -> float: ...


class SchedulerStateMachine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        sm = cfg.get("state_machine") if "state_machine" in cfg else cfg
        self.primary_symbol = sm.get("primary_symbol", "XAUUSD")
        self.monitor_symbols = list(sm.get("monitor_symbols") or ["XAUUSD", "USOIL"])
        self.adx_threshold = float(sm.get("adx_threshold", sm.get("adx_trend_min", 25)))
        self.atr_ratio_threshold = float(
            sm.get("atr_ratio_threshold", sm.get("atr_volatile_ratio", 1.5))
        )
        self.model_degradation_winrate = float(sm.get("model_degradation_winrate", 0.45))
        self.degradation_window = int(sm.get("degradation_window", 20))
        self.trend_strategy = sm.get("trend_strategy", "ai_model")
        self.volatile_strategy = sm.get("volatile_strategy", "spread_hedge")
        self.range_strategy = sm.get("range_strategy", "grid_system")
        self.degraded_strategy = sm.get("degraded_strategy", "trend_system")
        self.state = SchedulerMarketState.UNKNOWN

    def update(self, context: SchedulerContextProto | StrategyContext, symbol: str | None = None) -> SchedulerMarketState:
        sym = symbol or self.primary_symbol

        for monitored in self.monitor_symbols:
            recent = context.get_recent_winrate(monitored, self.degradation_window)
            if recent < self.model_degradation_winrate:
                self.state = SchedulerMarketState.MODEL_DEGRADED
                return self.state

        if context.is_macro_silence():
            self.state = SchedulerMarketState.VOLATILE
            return self.state

        if context.get_atr_ratio(sym) >= self.atr_ratio_threshold:
            self.state = SchedulerMarketState.VOLATILE
            return self.state

        adx = context.get_adx(sym)
        if adx > self.adx_threshold:
            self.state = SchedulerMarketState.TREND
        else:
            self.state = SchedulerMarketState.RANGE
        return self.state

    def get_active_strategy(self) -> str:
        if self.state == SchedulerMarketState.TREND:
            return self.trend_strategy
        if self.state == SchedulerMarketState.VOLATILE:
            return self.volatile_strategy
        if self.state == SchedulerMarketState.MODEL_DEGRADED:
            return self.degraded_strategy
        if self.state == SchedulerMarketState.RANGE:
            return self.range_strategy
        return self.trend_strategy

    def describe(self, context: SchedulerContextProto | StrategyContext, symbol: str | None = None) -> dict[str, Any]:
        sym = symbol or self.primary_symbol
        self.update(context, sym)
        return {
            "state": self.state.value,
            "strategy": self.get_active_strategy(),
            "symbol": sym,
            "atr_ratio": round(context.get_atr_ratio(sym), 3),
            "adx": round(context.get_adx(sym), 2),
            "macro_silence": context.is_macro_silence(),
        }
