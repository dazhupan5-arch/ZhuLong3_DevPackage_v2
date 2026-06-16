"""TradingEnv 反事实奖励测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhulong.agent.trading_env import TradingEnv


def _env_data(n: int = 300):
    close = 2000 + np.cumsum(np.random.randn(n) * 0.3)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(n) * 100,
        }
    )
    struct = np.random.randn(n, 30).astype(np.float32) * 0.1
    emb = np.random.randn(n, 32).astype(np.float32) * 0.1
    shocks = np.random.randn(n).astype(np.float32) * 0.05
    return df, struct, emb, shocks


def test_env_counterfactual_flag_does_not_crash():
    pytest.importorskip("gymnasium")
    df, struct, emb, shocks = _env_data()
    env = TradingEnv(
        df,
        struct,
        emb,
        {
            "initial_balance": 10000,
            "hold_penalty": 0,
            "counterfactual": {"enabled": True},
            "causal": {"coef_path": "models/missing_causal.pkl"},
        },
        exogenous_shocks=shocks,
    )
    obs, _ = env.reset()
    for _ in range(50):
        obs, reward, done, trunc, _ = env.step(1)
        assert np.isfinite(reward)
        if done:
            break
    assert len(env.trades) >= 0
