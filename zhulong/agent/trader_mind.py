"""L3 交易员层：不改 forecast 方向，只定执行计划。"""

from __future__ import annotations

from typing import Any

from zhulong.agent.tick_brief import ExecutionPlan, HorizonForecast, StructureSnapshot


class TraderMind:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        tm = cfg.get("trader_mind") or {}
        hp = (cfg.get("architecture") or {}).get("horizon_predictor") or {}
        self.max_consecutive_losses = int(tm.get("max_consecutive_losses", 6))
        self.sl_atr_mult = float(tm.get("sl_atr_mult", 1.2))
        self.tp_atr_mult = float(tm.get("tp_atr_mult", 2.0))
        self.ranging_sl_atr_mult = float(tm.get("ranging_sl_atr_mult", 1.8))
        self.choppy_sl_atr_mult = float(tm.get("choppy_sl_atr_mult", 2.0))
        self.min_confidence = float(
            tm.get("min_confidence", hp.get("min_direction_confidence", 0.48))
        )

    def plan(
        self,
        forecast: HorizonForecast,
        snapshot: StructureSnapshot,
        *,
        close: float,
        atr: float,
        consecutive_losses: int = 0,
        regime: str = "",
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

        sl, tp, sl_reason = self._sl_tp(direction, snapshot, close, atr, regime=regime)
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
        regime: str = "",
    ) -> tuple[float, float, str]:
        if atr <= 0:
            atr = close * 0.001
        sl_mult = self.sl_atr_mult
        reg = (regime or snap.zigzag_phase or "").lower()
        if reg in ("ranging", "range"):
            sl_mult = max(sl_mult, self.ranging_sl_atr_mult)
        elif reg == "choppy":
            sl_mult = max(sl_mult, self.choppy_sl_atr_mult)
        sup = close - snap.support_dist_atr * atr
        res = close + snap.resistance_dist_atr * atr
        if direction == "long":
            sl = min(sup, close - sl_mult * atr) if sup > 0 else close - sl_mult * atr
            tp = max(res, close + self.tp_atr_mult * atr) if res > close else close + self.tp_atr_mult * atr
            return sl, tp, f"long_struct_atr_sl{sl_mult:.1f}"
        sl = max(res, close + sl_mult * atr) if res > 0 else close + sl_mult * atr
        tp = min(sup, close - self.tp_atr_mult * atr) if sup < close else close - self.tp_atr_mult * atr
        return sl, tp, f"short_struct_atr_sl{sl_mult:.1f}"
