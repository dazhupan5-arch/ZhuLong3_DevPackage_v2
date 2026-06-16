"""V12 结构过滤器单元测试。"""

from __future__ import annotations

import numpy as np

from zhulong.agent.structure_analyzer import FEATURE_DIM, FEATURE_NAMES
from zhulong.strategies.v12_structure_filter import V12WithStructureFilter


def _feat(**overrides: float) -> np.ndarray:
    row = np.zeros(FEATURE_DIM, dtype=np.float32)
    for k, v in overrides.items():
        row[FEATURE_NAMES.index(k)] = v
    return row


def test_long_at_support():
    f = V12WithStructureFilter(long_prob_threshold=0.6, max_support_dist=0.5, mtf_align_min=0.01)
    feat = _feat(
        m5_trend=0.5,
        m5_support_dist=0.3,
        m5_support_strength=0.6,
        mtf_trend_align=0.2,
    )
    assert f.should_open_long(feat, 0.7, 1.0, 2000.0) is True


def test_short_blocked_without_resistance():
    f = V12WithStructureFilter(short_prob_threshold=0.6)
    feat = _feat(m5_trend=-0.5, m5_resistance_dist=2.0, m5_resistance_strength=0.1)
    assert f.should_open_short(feat, 0.8, 1.0, 2000.0) is False


def test_get_signal_priority_long():
    f = V12WithStructureFilter(long_prob_threshold=0.5, short_prob_threshold=0.5, mtf_align_min=0.01)
    feat = _feat(
        m5_trend=0.4,
        m5_support_dist=0.2,
        m5_support_strength=0.5,
        mtf_trend_align=0.3,
    )
    sig = f.get_signal(feat, [0.1, 0.7, 0.7], 1.0, 2000.0)
    assert sig == 1
