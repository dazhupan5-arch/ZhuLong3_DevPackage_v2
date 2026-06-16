"""调度上下文：桥接 StrategyContext 与权重/胜率查询。"""

from __future__ import annotations

from typing import Any

from zhulong.scheduler.risk_manager import SchedulerRiskManager
from zhulong.scheduler.weight_allocator import WeightAllocator
from zhulong.strategies.base import StrategyContext


class SchedulerContext:
    def __init__(
        self,
        strategy_context: StrategyContext,
        weight_allocator: WeightAllocator,
        risk_manager: SchedulerRiskManager,
    ) -> None:
        self._ctx = strategy_context
        self.weights = weight_allocator
        self.risk = risk_manager

    def is_macro_silence(self) -> bool:
        return self._ctx.is_macro_silence()

    def get_atr_ratio(self, symbol: str, lookback: int = 100) -> float:
        return self._ctx.get_atr_ratio(symbol, lookback)

    def get_adx(self, symbol: str, period: int = 14) -> float:
        return self._ctx.get_adx(symbol, period)

    def get_recent_winrate(self, symbol: str, window: int = 20) -> float:
        return self.weights.get_recent_winrate(symbol, window)

    @property
    def strategy_context(self) -> StrategyContext:
        return self._ctx

    def extra(self) -> dict[str, Any]:
        return {
            "weights": {s: round(self.weights.get_current_winrate(s), 3) for s in self.weights.base_weights},
            "risk": self.risk.status(),
        }
