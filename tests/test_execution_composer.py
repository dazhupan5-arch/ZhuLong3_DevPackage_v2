"""ExecutionComposer 与执行契约单元测试。"""

from __future__ import annotations

import pytest

from zhulong.agent.execution_composer import (
    ExecutionComposer,
    decide_entry_mode,
    evaluate_entry_against_plan,
    limit_fill_on_bar,
    location_score,
    structure_entry_target,
)
from zhulong.agent.tick_brief import ExecutionPlan, HorizonForecast, StructureSnapshot


def _snap(sup: float = 0.4, res: float = 0.6) -> StructureSnapshot:
    return StructureSnapshot(
        vector=[0.0] * 30,
        support_dist_atr=sup,
        resistance_dist_atr=res,
    )


def test_location_score_symmetric():
    assert location_score("long", 0.2, None) == 1.0
    assert location_score("short", 0.8, None) == 1.0
    assert location_score("long", 0.9, None) < 0.5
    assert location_score("short", 0.1, None) < 0.5


def test_structure_entry_target_long_below_close():
    snap = _snap(sup=0.5, res=1.2)
    target = structure_entry_target("long", snap, close=2000.0, atr=10.0, loc_score=0.3)
    assert target < 2000.0


def test_structure_entry_target_short_above_close():
    snap = _snap(sup=0.5, res=0.5)
    target = structure_entry_target("short", snap, close=2000.0, atr=10.0, loc_score=0.3)
    assert target > 2000.0


def test_limit_fill_on_bar():
    assert limit_fill_on_bar("long", 1995.0, 2010.0, 1990.0, 2000.0) == 1995.0
    assert limit_fill_on_bar("short", 2005.0, 2010.0, 1990.0, 2000.0) == 2005.0
    assert limit_fill_on_bar("short", 2015.0, 2010.0, 1990.0, 2000.0) is None


def test_evaluate_entry_limit_emits_working_intent():
    plan = ExecutionPlan(
        direction="short",
        entry_mode="limit",
        entry_target=2005.0,
        should_trade=True,
    )
    out = evaluate_entry_against_plan(
        plan,
        direction="sell",
        tick_bid=2000.0,
        tick_ask=2000.5,
        bar_close=2000.0,
        atr=10.0,
    )
    assert out["emit_working_intent"] is True
    assert out["should_wait"] is True
    assert out["entry_price"] == 2005.0


def test_composer_kn2_veto():
    composer = ExecutionComposer({"kn2": {"enabled": True, "min_confidence": 0.5}})
    forecast = HorizonForecast(direction="short", confidence=0.6, prob_down=0.6, prob_flat=0.2, prob_up=0.2)
    plan = composer.compose(
        forecast,
        _snap(),
        close=2000.0,
        atr=10.0,
        pos_in_range=0.8,
        kn2_dec={"should_trade": False, "confidence": 0.7},
        horizon_flat=False,
    )
    assert plan.should_trade is False
    assert plan.block_reason == "kn2_veto"


def test_composer_limit_mode_low_location():
    composer = ExecutionComposer({"kn2": {"enabled": True, "min_confidence": 0.4}})
    forecast = HorizonForecast(direction="short", confidence=0.55, prob_down=0.55, prob_flat=0.25, prob_up=0.2)
    plan = composer.compose(
        forecast,
        _snap(res=0.4),
        close=2000.0,
        atr=10.0,
        pos_in_range=0.35,
        kn2_dec={"should_trade": True, "confidence": 0.6, "sl_atr_mult": 1.2, "tp_atr_mult": 2.0},
        horizon_flat=False,
    )
    assert plan.should_trade is True
    assert plan.entry_mode in ("limit", "defer", "immediate")
    assert plan.entry_target >= 2000.0


def test_decide_entry_mode_immediate_when_high_quality():
    mode = decide_entry_mode("long", 2000.0, 1998.0, loc_score=0.92, entry_quality=0.85)
    assert mode == "immediate"
