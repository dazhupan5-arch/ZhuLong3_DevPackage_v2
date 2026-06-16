"""多策略与模型能力自动化评估。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zhulong.strategies.base import StrategyContext  # noqa: E402
from zhulong.strategies.grid_system import GridSystem  # noqa: E402
from zhulong.strategies.indicators import atr_series, ema_cross_up  # noqa: E402
from zhulong.strategies.state_machine import MarketState, StrategyStateMachine  # noqa: E402
from zhulong.strategies.trend_system import TrendSystem  # noqa: E402


def _synthetic_m5(n: int = 300, trend: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = 2000 + np.cumsum(np.random.default_rng(42).normal(trend, 0.5, n))
    high = close + 1.5
    low = close - 1.5
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 100.0},
        index=idx,
    )


class TestIndicators(unittest.TestCase):
    def test_atr_positive(self) -> None:
        m5 = _synthetic_m5(100)
        atr = atr_series(m5)
        self.assertFalse(np.isnan(atr.iloc[-1]))
        self.assertGreater(float(atr.iloc[-1]), 0)


class TestStateMachine(unittest.TestCase):
    def test_select_strategy_mapping(self) -> None:
        sm = StrategyStateMachine(
            {"state_machine": {"trend_strategy": "ai_model", "range_strategy": "grid_system"}}
        )
        self.assertEqual(sm.select_strategy(MarketState.TREND), "ai_model")
        self.assertEqual(sm.select_strategy(MarketState.RANGE), "grid_system")


class TestTrendSystem(unittest.TestCase):
    def test_flat_without_cross(self) -> None:
        m5 = _synthetic_m5(120, trend=0.0)
        ctx = StrategyContext(m5_by_symbol={"XAUUSD": m5}, config={})
        sig = TrendSystem({"min_atr_pct": 0.001}).on_bar("XAUUSD", ctx)
        self.assertIsNotNone(sig)
        assert sig is not None
        self.assertEqual(sig.direction, "flat")


class TestModelAcceptanceChecklist(unittest.TestCase):
    """样本外指标门槛（来自 manifest / 验收报告）。"""

    MANIFEST_METRICS = {
        "test1_win_rate": 0.5567,
        "val_precision": 0.5716,
    }
    THRESHOLDS = {
        "test1_win_rate": 0.52,
        "val_precision": 0.50,
    }

    def test_manifest_meets_minimum(self) -> None:
        manifest_path = ROOT / "models" / "XAUUSD" / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest("无 XAUUSD manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = manifest.get("metrics") or {}
        for key, min_val in self.THRESHOLDS.items():
            if key in metrics:
                self.assertGreaterEqual(metrics[key], min_val, msg=key)


class TestGridSystem(unittest.TestCase):
    def test_low_vol_may_emit_or_flat(self) -> None:
        m5 = _synthetic_m5(200, trend=0.0)
        ctx = StrategyContext(m5_by_symbol={"XAUUSD": m5}, config={})
        sig = GridSystem({"low_volatility_atr_percentile": 80}).on_bar("XAUUSD", ctx)
        self.assertIsNotNone(sig)


if __name__ == "__main__":
    unittest.main()
