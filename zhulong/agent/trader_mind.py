"""L3 交易员层：不改 forecast 方向，只定执行计划。"""

from __future__ import annotations

from typing import Any

from zhulong.agent.tick_brief import ExecutionPlan, HorizonForecast, StructureSnapshot


class TraderMind:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        tm = cfg.get("trader_mind") or {}
        self.max_consecutive_losses = int(tm.get("max_consecutive_losses", 6))
        self.sl_atr_mult = float(tm.get("sl_atr_mult", 1.2))
        self.tp_atr_mult = float(tm.get("tp_atr_mult", 2.0))
        self.min_confidence = float(tm.get("min_confidence", 0.42))

    def plan(
        self,
        forecast: HorizonForecast,
        snapshot: StructureSnapshot,
        *,
        close: float,
        atr: float,
        consecutive_losses: int = 0,
    ) -> ExecutionPlan:
        direction = forecast.direction
        plan = ExecutionPlan(direction=direction, action="hold", should_trade=False)

        if direction == "flat":
            plan.block_reason = "forecast_flat"
            return plan
        if forecast.confidence < self.min_confidence:
            plan.block_reason = "low_forecast_confidence"
            return plan
        if consecutive_losses >= self.max_consecutive_losses:
            plan.block_reason = "consecutive_losses"
            return plan

        sl, tp, sl_reason = self._sl_tp(direction, snapshot, close, atr)
        plan.action = "enter"
        plan.should_trade = True
        plan.size_mult = 1.0
        plan.sl_price = sl
        plan.tp_price = tp
        plan.sl_reason = sl_reason
        return plan

    def _sl_tp(
        self,
        direction: str,
        snap: StructureSnapshot,
        close: float,
        atr: float,
    ) -> tuple[float, float, str]:
        if atr <= 0:
            atr = close * 0.001
        sup = close - snap.support_dist_atr * atr
        res = close + snap.resistance_dist_atr * atr
        if direction == "long":
            sl = min(sup, close - self.sl_atr_mult * atr) if sup > 0 else close - self.sl_atr_mult * atr
            tp = max(res, close + self.tp_atr_mult * atr) if res > close else close + self.tp_atr_mult * atr
            return sl, tp, "long_struct_atr"
        sl = max(res, close + self.sl_atr_mult * atr) if res > 0 else close + self.sl_atr_mult * atr
        tp = min(sup, close - self.tp_atr_mult * atr) if sup < close else close - self.tp_atr_mult * atr
        return sl, tp, "short_struct_atr"
