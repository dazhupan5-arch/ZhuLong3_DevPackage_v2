"""Regime 条件元学习测试。"""

from __future__ import annotations

import numpy as np

from zhulong.agent.meta_learner import MetaLearner


def test_regime_conditional_bias(tmp_path):
    ml = MetaLearner(
        {
            "enabled": True,
            "meta_batch_size": 2,
            "meta_learning_rate": 0.05,
            "regime_conditional": True,
            "regime_learning_rate": 0.05,
        },
        state_dir=tmp_path,
    )
    long_traj = [{"state": np.zeros(4), "action": 1, "reward": 0.0, "regime": "ranging"}]
    short_traj = [{"state": np.zeros(4), "action": 2, "reward": 0.0, "regime": "trend"}]
    ml.add_trajectory(long_traj, regime="ranging", pnl_r=0.6)
    ml.add_trajectory(long_traj, regime="ranging", pnl_r=0.4)
    ml.add_trajectory(short_traj, regime="trend", pnl_r=-0.4)
    ml.add_trajectory(short_traj, regime="trend", pnl_r=-0.2)
    result = ml.meta_update(batch_size=4)
    assert not result.get("skipped")
    bias_ranging = ml.action_bias("ranging")
    bias_trend = ml.action_bias("trend")
    assert bias_ranging.shape == (6,)
    assert not np.allclose(bias_ranging, bias_trend)
