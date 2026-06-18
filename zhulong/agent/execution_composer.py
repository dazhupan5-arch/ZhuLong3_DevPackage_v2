"""V16 执行合成层：Horizon + KN2 + Structure → 统一 ExecutionPlan（long/short 对称）。"""

from __future__ import annotations

import logging
from typing import Any

from zhulong.agent.kn2_location_labels import LocationLabelConfig
from zhulong.agent.tick_brief import ExecutionPlan, HorizonForecast, StructureSnapshot

logger = logging.getLogger(__name__)

ENTRY_IMMEDIATE = "immediate"
ENTRY_LIMIT = "limit"
ENTRY_DEFER = "defer"


def location_score(direction: str, pos_in_range: float, cfg: LocationLabelConfig | None = None) -> float:
    """0~1：当前区间位置与训练标签一致程度（long 偏下、short 偏上）。"""
    cfg = cfg or LocationLabelConfig()
    pos = float(pos_in_range)
    if direction == "long":
        if pos <= cfg.pos_range_long_max:
            return 1.0
        span = max(1.0 - cfg.pos_range_long_max, 1e-9)
        return max(0.0, 1.0 - (pos - cfg.pos_range_long_max) / span)
    if direction == "short":
        if pos >= cfg.pos_range_short_min:
            return 1.0
        span = max(cfg.pos_range_short_min, 1e-9)
        return max(0.0, pos / span)
    return 0.0


def structure_entry_target(
    direction: str,
    snap: StructureSnapshot,
    close: float,
    atr: float,
    *,
    loc_score: float,
) -> float:
    """结构锚定入场目标价：long 靠近 support，short 靠近 resistance。"""
    if close <= 0 or atr <= 0:
        return close
    sup = close - float(snap.support_dist_atr) * atr
    res = close + float(snap.resistance_dist_atr) * atr
    pull = 0.15 * atr * max(0.35, 1.0 - loc_score)

    if direction == "long":
        if sup > 0 and sup < close:
            return max(sup, close - pull)
        return close - pull

    if direction == "short":
        if res > close:
            return min(res, close + pull)
        return close + pull

    return close


def decide_entry_mode(
    direction: str,
    close: float,
    entry_target: float,
    loc_score: float,
    entry_quality: float,
    *,
    immediate_quality_min: float = 0.72,
    limit_quality_min: float = 0.38,
) -> str:
    """根据位置与综合质量决定 immediate / limit / defer。"""
    if direction not in ("long", "short"):
        return ENTRY_DEFER
    if loc_score >= 0.88 and entry_quality >= immediate_quality_min:
        return ENTRY_IMMEDIATE
    if entry_quality >= limit_quality_min or loc_score >= 0.45:
        if direction == "long" and entry_target < close - 1e-9:
            return ENTRY_LIMIT
        if direction == "short" and entry_target > close + 1e-9:
            return ENTRY_LIMIT
        if loc_score >= 0.75:
            return ENTRY_IMMEDIATE
        return ENTRY_LIMIT
    return ENTRY_DEFER


def evaluate_entry_against_plan(
    plan: ExecutionPlan,
    *,
    direction: str,
    tick_bid: float,
    tick_ask: float,
    bar_close: float,
    atr: float,
) -> dict[str, Any]:
    """按 ExecutionPlan 评估 tick 入场；limit 模式未触价时保留信号（should_wait 但不否决方向）。"""
    result: dict[str, Any] = {
        "entry_price": bar_close,
        "should_wait": False,
        "emit_working_intent": False,
        "reason": "",
        "entry_mode": plan.entry_mode,
    }
    if direction not in ("buy", "sell") or bar_close <= 0:
        return result

    trade_dir = "long" if direction == "buy" else "short"
    if plan.direction not in ("long", "short") or plan.direction != trade_dir:
        result["entry_price"] = bar_close
        return result

    target = float(plan.entry_target or 0.0)
    if target <= 0:
        target = bar_close

    mode = plan.entry_mode or ENTRY_IMMEDIATE
    has_tick = tick_bid > 0 and tick_ask > 0

    if mode == ENTRY_IMMEDIATE:
        if not has_tick:
            result["entry_price"] = bar_close
            return result
        mid = (tick_bid + tick_ask) / 2.0
        chase_limit = atr * 0.25 if atr > 0 else bar_close * 0.0003
        if direction == "buy":
            ideal = min(bar_close, mid, target if target > 0 else bar_close)
            chase = tick_ask - ideal
            if chase > chase_limit:
                result["should_wait"] = True
                result["emit_working_intent"] = True
                result["entry_price"] = round(min(target, tick_ask), 5)
                result["reason"] = f"Ask={tick_ask:.2f} 高于理想{ideal:.2f}"
                return result
            result["entry_price"] = round(min(ideal, tick_ask), 5)
        else:
            ideal = max(bar_close, mid, target if target > 0 else bar_close)
            chase = ideal - tick_bid
            if chase > chase_limit:
                result["should_wait"] = True
                result["emit_working_intent"] = True
                result["entry_price"] = round(max(target, tick_bid), 5)
                result["reason"] = f"Bid={tick_bid:.2f} 低于理想{ideal:.2f}"
                return result
            result["entry_price"] = round(max(ideal, tick_bid), 5)
        return result

    if mode == ENTRY_LIMIT:
        result["emit_working_intent"] = True
        result["entry_price"] = round(target, 5)
        if not has_tick:
            result["should_wait"] = True
            result["reason"] = "limit 等待 tick"
            return result
        if direction == "buy":
            if tick_ask <= target:
                result["entry_price"] = round(min(target, tick_ask), 5)
                result["should_wait"] = False
            else:
                result["should_wait"] = True
                result["reason"] = f"限价买入≤{target:.2f} Ask={tick_ask:.2f}"
        else:
            if tick_bid >= target:
                result["entry_price"] = round(max(target, tick_bid), 5)
                result["should_wait"] = False
            else:
                result["should_wait"] = True
                result["reason"] = f"限价卖出≥{target:.2f} Bid={tick_bid:.2f}"
        return result

    # defer：本 bar 不追价，保留限价意图供下 tick / 下 bar 延续
    result["emit_working_intent"] = True
    result["should_wait"] = True
    result["entry_price"] = round(target, 5)
    result["reason"] = "defer 等待结构价"
    return result


def limit_fill_on_bar(
    direction: str,
    target: float,
    high: float,
    low: float,
    close: float,
) -> float | None:
    """单根 K 线限价撮合（与 C# IntentFillMatcher M1 穿价同构）。"""
    if target <= 0:
        return None
    if direction == "long" and low <= target:
        return float(min(target, close))
    if direction == "short" and high >= target:
        return float(max(target, close))
    return None


class ExecutionComposer:
    """融合 Horizon + KN2 + Structure，产出与训练标签同构的执行计划。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        tm = cfg.get("trader_mind") or {}
        ec = cfg.get("execution_composer") or {}
        hp = (cfg.get("architecture") or {}).get("horizon_predictor") or {}
        kn2 = cfg.get("kn2") or {}

        self.max_consecutive_losses = int(tm.get("max_consecutive_losses", 6))
        self.sl_atr_mult = float(tm.get("sl_atr_mult", 1.2))
        self.tp_atr_mult = float(tm.get("tp_atr_mult", 2.0))
        self.ranging_sl_atr_mult = float(tm.get("ranging_sl_atr_mult", 1.8))
        self.choppy_sl_atr_mult = float(tm.get("choppy_sl_atr_mult", 2.0))
        self.min_confidence = float(
            tm.get("min_confidence", hp.get("min_direction_confidence", 0.48))
        )
        self.kn2_min_confidence = float(kn2.get("min_confidence", 0.48))
        self.kn2_enabled = bool(kn2.get("enabled", False))
        self.valid_bars = int(ec.get("valid_bars", tm.get("valid_bars", 48)))
        self.immediate_quality_min = float(ec.get("immediate_quality_min", 0.72))
        self.limit_quality_min = float(ec.get("limit_quality_min", 0.38))
        self.loc_cfg = LocationLabelConfig()

    def compose(
        self,
        forecast: HorizonForecast,
        snapshot: StructureSnapshot,
        *,
        close: float,
        atr: float,
        pos_in_range: float = 0.5,
        kn2_dec: dict[str, Any] | None = None,
        consecutive_losses: int = 0,
        regime: str = "",
        horizon_flat: bool = False,
    ) -> ExecutionPlan:
        direction = str(forecast.direction or "flat").lower()
        plan = ExecutionPlan(
            direction=direction,
            action="hold",
            should_trade=False,
            pos_in_range=float(pos_in_range),
            valid_bars=self.valid_bars,
            source="composer",
        )

        if direction == "flat":
            plan.block_reason = "forecast_flat"
            return plan
        if forecast.confidence < self.min_confidence:
            plan.block_reason = "low_forecast_confidence"
            return plan
        if consecutive_losses >= self.max_consecutive_losses:
            plan.block_reason = "consecutive_losses"
            return plan

        loc = location_score(direction, pos_in_range, self.loc_cfg)
        kn2_conf = float(kn2_dec.get("confidence", 0)) if kn2_dec else 0.0
        kn2_trade = bool(kn2_dec.get("should_trade")) if kn2_dec else True

        if kn2_dec and self.kn2_enabled and not horizon_flat:
            if not kn2_trade:
                plan.block_reason = "kn2_veto"
                plan.metadata["kn2_should_trade"] = False
                return plan
            if kn2_conf < self.kn2_min_confidence:
                plan.block_reason = "kn2_confidence_low"
                return plan

        kn2_factor = kn2_conf if kn2_dec and self.kn2_enabled and not horizon_flat else 1.0
        if kn2_factor <= 0:
            kn2_factor = 1.0
        entry_quality = float(forecast.confidence) * kn2_factor * (0.55 + 0.45 * loc)

        entry_target = structure_entry_target(
            direction, snapshot, close, atr, loc_score=loc
        )
        entry_mode = decide_entry_mode(
            direction,
            close,
            entry_target,
            loc,
            entry_quality,
            immediate_quality_min=self.immediate_quality_min,
            limit_quality_min=self.limit_quality_min,
        )

        if entry_mode == ENTRY_DEFER and entry_quality < self.limit_quality_min * 0.85:
            plan.block_reason = "entry_quality_low"
            plan.entry_quality = round(entry_quality, 4)
            return plan

        sl, tp, sl_reason = self._sl_tp(direction, snapshot, close, atr, regime=regime, kn2_dec=kn2_dec)
        plan.action = "enter"
        plan.should_trade = True
        plan.entry_mode = entry_mode
        plan.entry_target = round(entry_target, 5)
        plan.entry_quality = round(entry_quality, 4)
        plan.size_mult = 1.0
        plan.sl_price = sl
        plan.tp_price = tp
        plan.sl_reason = sl_reason
        plan.metadata = {
            "loc_score": round(loc, 4),
            "kn2_conf": round(kn2_conf, 4),
            "kn2_should_trade": kn2_trade,
        }

        logger.info(
            "[ExecutionComposer] dir=%s mode=%s target=%.2f quality=%.3f loc=%.3f pos=%.3f",
            direction,
            entry_mode,
            entry_target,
            entry_quality,
            loc,
            pos_in_range,
        )
        return plan

    def _sl_tp(
        self,
        direction: str,
        snap: StructureSnapshot,
        close: float,
        atr: float,
        regime: str = "",
        kn2_dec: dict[str, Any] | None = None,
    ) -> tuple[float, float, str]:
        if atr <= 0:
            atr = close * 0.001
        sl_mult = self.sl_atr_mult
        tp_mult = self.tp_atr_mult
        reg = (regime or snap.zigzag_phase or "").lower()
        if reg in ("ranging", "range"):
            sl_mult = max(sl_mult, self.ranging_sl_atr_mult)
        elif reg == "choppy":
            sl_mult = max(sl_mult, self.choppy_sl_atr_mult)

        if kn2_dec:
            kn2_sl = float(kn2_dec.get("sl_atr_mult", 0) or 0)
            kn2_tp = float(kn2_dec.get("tp_atr_mult", 0) or 0)
            if kn2_sl > 0:
                sl_mult = kn2_sl
            if kn2_tp > 0:
                tp_mult = kn2_tp

        sup = close - snap.support_dist_atr * atr
        res = close + snap.resistance_dist_atr * atr
        if direction == "long":
            sl = min(sup, close - sl_mult * atr) if sup > 0 else close - sl_mult * atr
            tp = max(res, close + tp_mult * atr) if res > close else close + tp_mult * atr
            return sl, tp, f"long_struct_atr_sl{sl_mult:.1f}"
        sl = max(res, close + sl_mult * atr) if res > 0 else close + sl_mult * atr
        tp = min(sup, close - tp_mult * atr) if sup < close else close - tp_mult * atr
        return sl, tp, f"short_struct_atr_sl{sl_mult:.1f}"
