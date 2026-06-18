#!/usr/bin/env python3
"""契约对齐审计：ExecutionComposer 栈接线与关键路径探测。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    print("=== V16 Execution Parity Contract Audit ===\n")

    cfg_path = _ROOT / "config" / "config_agent.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    chk("config arch=v16", cfg.get("architecture", {}).get("version") == "v16")
    chk("execution_composer config", "execution_composer" in cfg)
    ep = (cfg.get("trading_env") or {}).get("execution_parity") or {}
    chk("execution_parity enabled", bool(ep.get("enabled")))

    from zhulong.agent.execution_composer import (
        ExecutionComposer,
        evaluate_entry_against_plan,
        limit_fill_on_bar,
        location_score_v2,
    )
    from zhulong.agent.tick_brief import ExecutionPlan, HorizonForecast, StructureSnapshot
    from zhulong.agent.trader_mind import TraderMind
    from zhulong.agent.trading_agent import TradingAgent

    chk("ExecutionComposer import", True)
    chk("TraderMind extends Composer", issubclass(TraderMind, ExecutionComposer))
    chk("_rl_sizing_action", hasattr(TradingAgent, "_rl_sizing_action"))

    snap = StructureSnapshot(vector=[0.0] * 30, support_dist_atr=0.4, resistance_dist_atr=0.5)
    fc = HorizonForecast(direction="short", confidence=0.68, prob_down=0.68, prob_flat=0.22, prob_up=0.1)
    composer = ExecutionComposer(cfg)
    plan = composer.compose(
        fc,
        snap,
        close=2000.0,
        atr=10.0,
        pos_in_range=0.75,
        kn2_dec={"should_trade": True, "confidence": 0.55, "sl_atr_mult": 1.2, "tp_atr_mult": 2.0},
        horizon_flat=False,
    )
    chk("compose short high pos → entry (v16_strict_3)", plan.should_trade,
        f"mode={plan.entry_mode} target={plan.entry_target} block={plan.block_reason}")
    if plan.should_trade:
        chk("compose entry_quality > 0", plan.entry_quality > 0, f"eq={plan.entry_quality}")
        chk("compose metadata loc_score", "loc_score" in (plan.metadata or {}),
            str(plan.metadata))
    else:
        chk("compose blocked with KN2 disabled (old model)", plan.block_reason is not None,
            str(plan.block_reason))

    # v16_strict_3: 验证 location_score_v2 多维评分
    loc_v2 = location_score_v2("long", 0.25, snap)
    chk("location_score_v2 long low pos > 0.5", loc_v2 > 0.5, f"loc_v2={loc_v2:.4f}")
    loc_v2_short = location_score_v2("short", 0.75, snap)
    chk("location_score_v2 short high pos > 0.5", loc_v2_short > 0.5, f"loc_v2={loc_v2_short:.4f}")
    loc_v2_bad = location_score_v2("long", 0.9, snap)
    chk("location_score_v2 long high pos < 0.6", loc_v2_bad < 0.6, f"loc_v2_bad={loc_v2_bad:.4f}")

    # v16_strict_3: 验证默认门禁收紧
    chk("composer immediate_quality_min=0.78", composer.immediate_quality_min >= 0.78,
        str(composer.immediate_quality_min))
    chk("composer limit_quality_min=0.45", composer.limit_quality_min >= 0.45,
        str(composer.limit_quality_min))
    chk("composer entry_quality_position_weight=0.70", composer.entry_quality_position_weight >= 0.65,
        str(composer.entry_quality_position_weight))

    # v16_strict_3: 验证 pull 系数增大后 entry_target 离 close 更远
    snap2 = StructureSnapshot(vector=[0.0] * 30, support_dist_atr=0.8, resistance_dist_atr=0.5)
    plan2 = composer.compose(
        HorizonForecast(direction="long", confidence=0.65, prob_up=0.65, prob_flat=0.2, prob_down=0.15),
        snap2,
        close=2000.0, atr=10.0, pos_in_range=0.25,
        kn2_dec={"should_trade": True, "confidence": 0.58},
    )
    if plan2.should_trade:
        gap = 2000.0 - plan2.entry_target
        chk("structure_entry_target pull >= 1.5 (0.175 ATR)", gap >= 1.5,
            f"entry_target={plan2.entry_target} gap={gap:.2f}")
    else:
        chk("structure_entry_target pull (plan blocked)", plan2.block_reason is not None,
            f"blocked: {plan2.block_reason}")

    limit_plan = ExecutionPlan(direction="short", entry_mode="limit", entry_target=2005.0, should_trade=True)
    ev = evaluate_entry_against_plan(
        limit_plan, direction="sell", tick_bid=2000.0, tick_ask=2000.5, bar_close=2000.0, atr=10.0
    )
    chk("limit emit_working_intent", ev.get("emit_working_intent") is True and ev.get("should_wait") is True)

    fill = limit_fill_on_bar("short", 2005.0, 2010.0, 1990.0, 2000.0)
    chk("limit_fill_on_bar short", fill == 2005.0, str(fill))

    ta_src = (_ROOT / "zhulong" / "agent" / "trading_agent.py").read_text(encoding="utf-8")
    chk("trading_agent uses evaluate_entry_against_plan", "evaluate_entry_against_plan" in ta_src)
    chk("trading_agent preserve_working_intent", "preserve_working_intent" in ta_src)
    chk("trading_agent KN2 before compose", "kn2_dec_early" in ta_src and "trader_mind.plan" in ta_src)
    chk("trading_agent RL sizing", "_rl_sizing_action" in ta_src)

    env_src = (_ROOT / "zhulong" / "agent" / "trading_env.py").read_text(encoding="utf-8")
    chk("trading_env execution_parity", "self.execution_parity" in env_src and "pending_direction" in env_src)

    remote = _ROOT / "scripts" / "train_rl_v16_remote.ps1"
    chk("remote RL script", remote.is_file())

    fails = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n=== Summary: {len(CHECKS) - len(fails)}/{len(CHECKS)} PASS ===")
    if fails:
        print("FAILED:", ", ".join(fails))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
