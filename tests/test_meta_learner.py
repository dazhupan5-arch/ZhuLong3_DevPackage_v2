"""元学习与适应触发测试。"""

from __future__ import annotations

import numpy as np

from zhulong.agent.adaptation_trigger import AdaptationTrigger
from zhulong.agent.meta_learner import MetaLearner


def test_adaptation_trigger_fires_on_low_winrate():
    t = AdaptationTrigger(window=10, threshold=0.45)
    for _ in range(10):
        t.add_result(False)
    assert t.should_adapt()


def test_adaptation_trigger_holds_on_good_winrate():
    t = AdaptationTrigger(window=10, threshold=0.45)
    for _ in range(10):
        t.add_result(True)
    assert not t.should_adapt()


def test_meta_learner_update_preserves_small_bias(tmp_path):
    ml = MetaLearner(
        {
            "enabled": True,
            "meta_batch_size": 2,
            "meta_learning_rate": 0.01,
            "max_param_delta_pct": 0.05,
        },
        state_dir=tmp_path,
    )
    traj = [
        {"state": np.zeros(74), "action": 1, "reward": 0.5},
        {"state": np.ones(74), "action": 1, "reward": 0.3},
    ]
    ml.add_trajectory(traj)
    ml.add_trajectory(traj)
    result = ml.meta_update(batch_size=2)
    assert not result.get("skipped")
    assert float(np.linalg.norm(ml.action_bias())) < 1.0


def test_meta_learner_skips_when_insufficient_data(tmp_path):
    ml = MetaLearner({"enabled": True, "meta_batch_size": 10}, state_dir=tmp_path)
    result = ml.meta_update()
    assert result.get("skipped")
