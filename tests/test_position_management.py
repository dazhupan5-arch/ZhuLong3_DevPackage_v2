"""持仓管理与开仓 SL/TP 路径测试（无需实盘）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhulong.agent.cognition import BarContext, CognitionEngine, ThoughtTrace
from zhulong.agent.trading_agent import TradingAgent


def _seed_context(engine: CognitionEngine, n: int = 12, close: float = 4300.0) -> np.ndarray:
    struct = np.zeros(30, dtype=np.float32)
    struct[3] = 2.0  # support_dist
    struct[4] = 2.5  # resistance_dist
    for i in range(n):
        engine.context.push(
            BarContext(
                timestamp=f"2026-06-18T10:{i:02d}:00Z",
                struct_features=struct.copy(),
                close=close + i * 0.1,
                atr=10.0,
                volume=1000.0,
                regime="ranging",
            )
        )
    return struct


def test_regime_metrics_unpack_triple():
    engine = CognitionEngine({"cognition": {"enabled": True, "symbol": "XAUUSD"}})
    sf = _seed_context(engine)
    metrics = engine._regime_metrics(sf)
    assert isinstance(metrics, dict)
    assert "pos_in_range" in metrics


def test_evaluate_position_management_does_not_crash():
    engine = CognitionEngine({"cognition": {"enabled": True, "symbol": "XAUUSD"}})
    sf = _seed_context(engine)
    thought = ThoughtTrace(
        regime="ranging",
        confidence=0.6,
        calibrated_probs=[0.2, 0.2, 0.6],
        trade_bias="short",
        ai_sl_price=4320.0,
        ai_tp_price=4280.0,
    )
    pos = {
        "direction": "sell",
        "entry": 4319.67,
        "sl": 4329.57,
        "tp": 4303.14,
        "profit_pct": -0.05,
        "peak_profit_pct": 0.02,
        "hold_seconds": 300.0,
    }
    out = engine.evaluate_position_management(
        thought,
        pos,
        sf,
        close=4325.0,
        atr=10.0,
        rl_action=4,
        kn2_dec={"should_trade": True, "confidence": 0.55, "sl_atr_mult": 1.2, "tp_atr_mult": 2.0},
    )
    assert "exit_score" in out
    assert "trail_mode" in out
    assert out["pos_in_range"] is not None


def test_evaluate_exit_for_position_regime_path():
    engine = CognitionEngine({"cognition": {"enabled": True, "symbol": "XAUUSD"}})
    sf = _seed_context(engine)
    thought = ThoughtTrace(regime="ranging", confidence=0.55, calibrated_probs=[0.3, 0.3, 0.4])
    pos = {"direction": "sell", "entry": 4319.0, "sl": 4330.0, "tp": 4300.0, "profit_pct": 0.0}
    out = engine.evaluate_exit_for_position(
        thought, pos, rl_action=0, close=4320.0, atr=10.0, struct_features=sf
    )
    assert "exit_score" in out


def test_resolve_entry_sl_tp_sell_merges_kn2_and_structure():
    class _AgentStub:
        cognition = CognitionEngine({"cognition": {"symbol": "XAUUSD"}})

    struct = np.zeros(30, dtype=np.float32)
    struct[3] = 1.5
    struct[4] = 2.0
    thought = ThoughtTrace(
        trade_bias="short",
        confidence=0.6,
        ai_sl_price=4335.0,
        ai_tp_price=4295.0,
    )
    kn2 = {"sl_atr_mult": 1.5, "tp_atr_mult": 2.5}

    class _Plan:
        sl_price = 4332.0
        tp_price = 4290.0

    ep = 4320.0
    atr = 10.0
    sl, tp = TradingAgent._resolve_entry_sl_tp(
        _AgentStub(), "sell", ep, atr, struct, thought, kn2, _Plan()
    )
    assert sl > ep, f"sell SL must be above entry, got sl={sl}"
    assert tp < ep, f"sell TP must be below entry, got tp={tp}"
    assert sl <= 4335.0
    assert tp <= 4295.0


def test_no_stale_regime_detect_unpack_in_cognition_source():
    src = (pytest.importorskip("pathlib").Path(__file__).resolve().parent.parent / "zhulong/agent/cognition.py")
    text = src.read_text(encoding="utf-8")
    assert "_, regime_metrics = self.regime.detect" not in text


def test_has_open_position_helper():
    acct = {
        "_positions": [{"symbol": "XAUUSD", "direction": "sell", "entry": 4300.0, "is_filled": True}],
    }
    assert TradingAgent._has_open_position("XAUUSD", acct) is True
    assert TradingAgent._has_open_position("USOIL", acct) is False


def test_m5_index_loc_handles_ndarray():
    idx = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    loc = np.array([1, 2])
    assert TradingAgent._m5_index_loc(idx, loc) == 2
