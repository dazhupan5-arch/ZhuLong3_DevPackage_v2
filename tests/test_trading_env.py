"""TradingEnv 单元测试。"""

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
    return df, struct, emb


def test_env_random_steps():
    pytest.importorskip("gymnasium")
    df, struct, emb = _env_data()
    env = TradingEnv(
        df,
        struct,
        emb,
        {"initial_balance": 10000, "hold_penalty": 0, "action_space": "simple3"},
    )
    assert env.action_space.n == 3
    obs, _ = env.reset()
    assert obs.shape == (74,)
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, done, trunc, _ = env.step(action)
        assert obs.shape == (74,)
        assert np.isfinite(reward)
        if done:
            break
    assert env.balance > 0
