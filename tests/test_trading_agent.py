"""TradingAgent tick 测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zhulong.agent.trading_agent import TradingAgent


def _m5(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-06-01", periods=n, freq="5min", tz="UTC")
    close = 2400 + np.cumsum(np.random.randn(n) * 0.2)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np.ones(n) * 50,
        },
        index=idx,
    )


def test_agent_on_bar_no_crash(tmp_path):
    cfg = {
        "enabled": True,
        "primary_symbol": "XAUUSD",
        "structure_analyzer": {"lookback": 100},
        "knowledge_net": {"model_path": str(tmp_path / "missing.pth")},
        "rl": {"model_path": str(tmp_path / "missing.zip")},
        "trading_env": {"initial_balance": 10000},
        "state_file": str(tmp_path / "state.json"),
    }
    agent = TradingAgent(cfg, root=tmp_path)
    m5 = _m5()
    results = agent.on_bar("XAUUSD", {"XAUUSD": m5})
    assert len(results) == 1
    assert results[0]["strategy"] == "rl_agent"
