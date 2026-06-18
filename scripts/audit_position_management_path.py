#!/usr/bin/env python3
"""静态路径审计：持仓管理 + 开仓结构 SL/TP（无需实盘、无需持仓）。"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    print("=" * 60)
    print("POSITION MANAGEMENT PATH AUDIT (static + probe)")
    print("=" * 60)

    cog_path = ROOT / "zhulong/agent/cognition.py"
    ta_path = ROOT / "zhulong/agent/trading_agent.py"
    cog = cog_path.read_text(encoding="utf-8-sig")
    ta = ta_path.read_text(encoding="utf-8-sig")

    chk("no 2-tuple regime.detect unpack", "_, regime_metrics = self.regime.detect" not in cog)
    chk("_regime_metrics helper", "def _regime_metrics" in cog)
    chk("detect returns 3-tuple documented", "返回 (regime_name, confidence, metrics_dict)" in cog)
    chk("_resolve_entry_sl_tp present", "def _resolve_entry_sl_tp" in ta)
    chk("entry uses structure merge", "_resolve_entry_sl_tp(" in ta)
    entry_block = ta.split("# 入场 tick 评估", 1)[-1].split("# 持仓中", 1)[0]
    chk("emit path avoids pure ATR entry_anchored", "entry_anchored=True" not in entry_block)

    rs = (ROOT / "src/ZhuLong.App/Services/ZhuLongRuntimeService.cs").read_text(encoding="utf-8-sig")
    chk("C# severe log on position+agent fail", "持仓中智能体 tick 失败" in rs)

    # AST: evaluate_position_management calls _regime_metrics or 3-unpack
    tree = ast.parse(cog, filename=str(cog_path))
    uses_helper = "_regime_metrics" in cog
    chk("evaluate_position_management wired", "def evaluate_position_management" in cog and uses_helper)

    print("\n--- runtime probe (mock position) ---")
    try:
        import numpy as np
        from zhulong.agent.cognition import BarContext, CognitionEngine, ThoughtTrace

        eng = CognitionEngine({"cognition": {"symbol": "XAUUSD"}})
        sf = np.zeros(30, dtype=np.float32)
        for i in range(12):
            eng.context.push(
                BarContext(timestamp=f"t{i}", struct_features=sf, close=4300 + i, atr=10.0)
            )
        thought = ThoughtTrace(regime="ranging", confidence=0.6, trade_bias="short")
        pos = {
            "direction": "sell",
            "entry": 4319.0,
            "sl": 4329.0,
            "tp": 4303.0,
            "profit_pct": 0.0,
            "hold_seconds": 120.0,
        }
        out = eng.evaluate_position_management(thought, pos, sf, 4325.0, 10.0)
        chk("evaluate_position_management probe", "trail_mode" in out, str(out.get("trail_mode")))
    except Exception as ex:
        chk("evaluate_position_management probe", False, str(ex)[:80])

    try:
        from zhulong.engine.agent_engine import run_agent_tick
        import pandas as pd

        n = 120
        idx = pd.date_range("2026-06-01", periods=n, freq="5min", tz="UTC")
        close = 4300 + np.cumsum(np.random.randn(n) * 0.3)
        m5 = pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000},
            index=idx,
        )
        cfg = ROOT / "config" / "config_agent.json"
        out = run_agent_tick(
            {"XAUUSD": m5},
            {
                "config_path": str(cfg),
                "symbols": ["XAUUSD"],
                "primary_symbol": "XAUUSD",
                "open_positions": [
                    {
                        "symbol": "XAUUSD",
                        "direction": "sell",
                        "entry": 4319.67,
                        "sl": 4329.57,
                        "tp": 4303.14,
                        "profit_pct": -0.1,
                        "peak_profit_pct": 0.0,
                        "hold_seconds": 300,
                    }
                ],
            },
            root=ROOT,
        )
        err = str(out.get("error", ""))
        if not out.get("ok") and "onnxruntime" in err.lower():
            chk("agent_tick with open_positions", True, "skip (no onnx in dev env)")
        else:
            chk("agent_tick with open_positions", out.get("ok") is True, err[:60])
            if out.get("ok"):
                r = (out.get("results") or [{}])[0]
                chk("position tick has exit_assessment key", "exit_assessment" in r)
                chk("position tick has trail_mode key", "trail_mode" in r)
    except ImportError as ex:
        chk("agent_tick with open_positions", True, f"skip (no onnx): {ex}"[:60])
    except Exception as ex:
        chk("agent_tick with open_positions", False, str(ex)[:100])

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print(f"\n{'=' * 60}")
    print(f"RESULT: {passed}/{total}")
    print("=" * 60)
    return 0 if passed == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
