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
    )
    from zhulong.agent.tick_brief import ExecutionPlan, HorizonForecast, StructureSnapshot
    from zhulong.agent.trader_mind import TraderMind
    from zhulong.agent.trading_agent import TradingAgent

    chk("ExecutionComposer import", True)
    chk("TraderMind extends Composer", issubclass(TraderMind, ExecutionComposer))
    chk("_rl_sizing_action", hasattr(TradingAgent, "_rl_sizing_action"))

    snap = StructureSnapshot(vector=[0.0] * 30, support_dist_atr=0.4, resistance_dist_atr=0.5)
    fc = HorizonForecast(direction="short", confidence=0.58, prob_down=0.58, prob_flat=0.22, prob_up=0.2)
    composer = ExecutionComposer(cfg)
    plan = composer.compose(
        fc,
        snap,
        close=2000.0,
        atr=10.0,
        pos_in_range=0.35,
        kn2_dec={"should_trade": True, "confidence": 0.55, "sl_atr_mult": 1.2, "tp_atr_mult": 2.0},
        horizon_flat=False,
    )
    chk("compose short low pos → limit/defer/immediate", plan.should_trade and plan.entry_target >= 2000.0,
        f"mode={plan.entry_mode} target={plan.entry_target}")

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
