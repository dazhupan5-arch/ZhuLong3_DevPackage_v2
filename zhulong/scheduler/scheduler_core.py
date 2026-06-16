"""调度器核心：权重 + 状态机 + 风控 → 最终信号。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from zhulong.scheduler.context import SchedulerContext
from zhulong.scheduler.market_state import SchedulerStateMachine
from zhulong.scheduler.risk_manager import SchedulerRiskManager
from zhulong.scheduler.types import ModelPrediction, SchedulerOutput
from zhulong.scheduler.weight_allocator import WeightAllocator


class SchedulerCore:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        wa_cfg = cfg.get("weight_allocator") or cfg
        sm_cfg = cfg.get("state_machine") or cfg
        rm_cfg = cfg.get("risk_manager") or cfg

        self.weight_allocator = WeightAllocator(wa_cfg)
        self.state_machine = SchedulerStateMachine({"state_machine": sm_cfg})
        self.risk_manager = SchedulerRiskManager(rm_cfg)
        self.vote_margin = float(cfg.get("vote_margin", 0.1))
        self.min_emit_weight = float(cfg.get("min_emit_weight", 0.15))
        self.primary_symbol = self.state_machine.primary_symbol
        self.weights: dict[str, float] = dict(self.weight_allocator.base_weights)

    def process_model_outputs(
        self,
        predictions: dict[str, ModelPrediction],
        context: SchedulerContext,
        *,
        primary_symbol: str | None = None,
    ) -> list[SchedulerOutput]:
        primary = primary_symbol or self.primary_symbol
        state = self.state_machine.state.value or "TREND"

        if not self.risk_manager.can_open_position():
            return [
                SchedulerOutput(
                    symbol=primary,
                    direction="flat",
                    confidence=0.0,
                    entry=0.0,
                    sl=0.0,
                    tp=0.0,
                    signal_id="",
                    strategy="scheduler_ai",
                    market_state=state,
                    reject_reason=self.risk_manager.block_reason or "risk_block",
                    metadata=context.extra(),
                )
            ]

        raw_weights: dict[str, float] = {}
        for sym, pred in predictions.items():
            if pred.direction == 0:
                continue
            raw_weights[sym] = self.weight_allocator.compute_weight(sym, pred.confidence)

        if not raw_weights:
            parts: list[str] = []
            for sym, pred in predictions.items():
                if pred.reject_reason:
                    parts.append(f"{sym}:{pred.reject_reason}")
                elif pred.direction == 0:
                    parts.append(f"{sym}:hold")
            reject = "; ".join(parts) if parts else "all_models_hold"
            return [
                SchedulerOutput(
                    symbol=primary,
                    direction="flat",
                    confidence=max((p.confidence for p in predictions.values()), default=0.0),
                    entry=0.0,
                    sl=0.0,
                    tp=0.0,
                    signal_id="",
                    strategy="scheduler_ai",
                    market_state=state,
                    reject_reason=reject,
                    metadata=context.extra(),
                )
            ]

        self.weights = self.weight_allocator.compute_normalized(raw_weights)

        long_weight = 0.0
        short_weight = 0.0
        for sym, pred in predictions.items():
            w = self.weights.get(sym, 0.0)
            if pred.direction == 1:
                long_weight += w
            elif pred.direction == -1:
                short_weight += w

        final_dir = 0
        confidence = 0.0
        if long_weight > short_weight + self.vote_margin:
            final_dir = 1
            confidence = long_weight
        elif short_weight > long_weight + self.vote_margin:
            final_dir = -1
            confidence = short_weight
        else:
            return [
                SchedulerOutput(
                    symbol=primary,
                    direction="flat",
                    confidence=max(long_weight, short_weight),
                    entry=0.0,
                    sl=0.0,
                    tp=0.0,
                    signal_id="",
                    strategy="scheduler_ai",
                    market_state=state,
                    weights=dict(self.weights),
                    reject_reason="direction_conflict",
                    metadata=context.extra(),
                )
            ]

        direction_str = "buy" if final_dir == 1 else "sell"
        outputs: list[SchedulerOutput] = []
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

        for sym, pred in predictions.items():
            if pred.direction != final_dir:
                continue
            w = self.weights.get(sym, 0.0)
            if w < self.min_emit_weight:
                continue
            sig_id = pred.signal_id or f"sched_{ts}_{sym}_{direction_str}"
            outputs.append(
                SchedulerOutput(
                    symbol=sym,
                    direction=direction_str,
                    confidence=confidence,
                    entry=pred.entry,
                    sl=pred.sl,
                    tp=pred.tp,
                    signal_id=sig_id,
                    strategy="scheduler_ai",
                    broker_symbol=pred.broker_symbol or sym,
                    risk_weight=w,
                    market_state=state,
                    weights=dict(self.weights),
                    metadata={**context.extra(), "model_confidence": pred.confidence},
                )
            )

        if not outputs and predictions.get(primary):
            pred = predictions[primary]
            if pred.direction == final_dir:
                w = self.weights.get(primary, confidence)
                outputs.append(
                    SchedulerOutput(
                        symbol=primary,
                        direction=direction_str,
                        confidence=confidence,
                        entry=pred.entry,
                        sl=pred.sl,
                        tp=pred.tp,
                        signal_id=pred.signal_id or f"sched_{ts}_{primary}_{direction_str}",
                        strategy="scheduler_ai",
                        broker_symbol=pred.broker_symbol or primary,
                        risk_weight=w,
                        market_state=state,
                        weights=dict(self.weights),
                        metadata=context.extra(),
                    )
                )

        return outputs

    def record_trade_result(self, symbol: str, pnl_r: float, is_win: bool, ts: datetime | None = None) -> None:
        self.weight_allocator.update(symbol, is_win)
        self.risk_manager.update(pnl_r, ts)

    def persist_blob(self) -> dict[str, Any]:
        return {
            "weight_allocator": self.weight_allocator.to_dict(),
            "risk_manager": self.risk_manager.to_dict(),
        }

    def load_blob(self, data: dict[str, Any] | None) -> None:
        if not data:
            return
        self.weight_allocator.load_dict(data.get("weight_allocator"))
        self.risk_manager.load_dict(data.get("risk_manager"))
