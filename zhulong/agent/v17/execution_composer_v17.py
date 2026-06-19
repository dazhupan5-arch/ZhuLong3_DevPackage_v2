"""V17 执行合成：DirectionScorer + LocationGate → ExecutionPlan。"""

from __future__ import annotations

from typing import Any

from zhulong.agent.execution_composer import (
    ENTRY_DEFER,
    ENTRY_IMMEDIATE,
    ENTRY_LIMIT,
    ExecutionComposer,
    decide_entry_mode,
    structure_entry_target,
)
from zhulong.agent.tick_brief import ExecutionPlan, StructureSnapshot


class ExecutionComposerV17(ExecutionComposer):
    """V17：signal_quality = direction_strength × location_quality × mtf_bonus × causal_boost。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        v17 = (config or {}).get("execution_composer_v17") or {}
        self.min_abs_direction = float(v17.get("min_abs_direction", 0.35))
        self.min_location_quality = float(v17.get("min_location_quality", 0.60))
        self.causal_boost_weight = float(v17.get("causal_boost_weight", 0.10))
        self.location_score_mode = "v17"

    def compose_v17(
        self,
        *,
        direction_score: float,
        location_quality: float,
        snapshot: StructureSnapshot,
        close: float,
        atr: float,
        pos_in_range: float = 0.5,
        consecutive_losses: int = 0,
        regime: str = "",
        causal_score: float = 0.0,
        location_gate_required: bool = True,
    ) -> ExecutionPlan:
        score = float(direction_score)
        loc_q = float(location_quality)
        plan = ExecutionPlan(
            direction="flat",
            action="hold",
            should_trade=False,
            pos_in_range=float(pos_in_range),
            valid_bars=self.valid_bars,
            source="composer_v17",
            metadata={
                "direction_score": round(score, 4),
                "location_quality": round(loc_q, 4),
            },
        )

        if abs(score) < self.min_abs_direction:
            plan.block_reason = "direction_score_below_threshold"
            return plan

        direction = "long" if score > 0 else "short"
        plan.direction = direction

        if location_gate_required and loc_q < self.min_location_quality:
            plan.block_reason = "location_quality_below_threshold"
            return plan

        if consecutive_losses >= self.max_consecutive_losses:
            plan.block_reason = "max_consecutive_losses"
            return plan

        direction_strength = abs(score)
        mtf = float(getattr(snapshot, "mtf_align", 0.0) or 0.0)
        mtf_bonus = 1.0 + 0.15 * abs(mtf)
        causal_boost = 1.0 + self.causal_boost_weight * causal_score if causal_score > 0 else 1.0
        signal_quality = min(
            1.0,
            direction_strength * loc_q * mtf_bonus * causal_boost,
        )
        plan.entry_quality = signal_quality

        loc_score = loc_q
        entry_target = structure_entry_target(
            direction, snapshot, close, atr, loc_score=loc_score
        )
        plan.entry_target = entry_target
        plan.entry_mode = decide_entry_mode(
            direction,
            close,
            entry_target,
            loc_score,
            signal_quality,
            immediate_quality_min=self.immediate_quality_min,
            limit_quality_min=self.limit_quality_min,
        )

        if plan.entry_mode == ENTRY_DEFER:
            plan.block_reason = "entry_quality_defer"
            return plan

        sl_mult = self.sl_atr_mult
        if regime in ("ranging", "range"):
            sl_mult = self.ranging_sl_atr_mult
        elif regime in ("choppy",):
            sl_mult = self.choppy_sl_atr_mult

        if direction == "long":
            plan.sl_price = close - sl_mult * atr
            plan.tp_price = close + self.tp_atr_mult * atr
            plan.action = "buy"
        else:
            plan.sl_price = close + sl_mult * atr
            plan.tp_price = close - self.tp_atr_mult * atr
            plan.action = "sell"

        plan.should_trade = plan.entry_mode in (ENTRY_IMMEDIATE, ENTRY_LIMIT)
        plan.metadata["signal_quality"] = round(signal_quality, 4)
        plan.metadata["mtf_bonus"] = round(mtf_bonus, 4)
        return plan
