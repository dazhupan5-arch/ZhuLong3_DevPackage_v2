"""V13 增强特征单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zhulong.analysis.feature_engineering import FEATURES_ENHANCED, add_enhanced_features
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
from zhulong.training.v13.train_pipeline import FEATURE_COLUMNS_V13_ENHANCED, compute_features_v13


def _sample_m5(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(42)
    close = 2000 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 0.3, n),
            "high": close + rng.uniform(0, 0.5, n),
            "low": close - rng.uniform(0, 0.5, n),
            "close": close,
            "volume": rng.integers(100, 1000, n),
        },
        index=idx,
    )


def test_enhanced_column_count():
    assert len(FEATURES_ENHANCED) >= 18
    assert len(FEATURE_COLUMNS_V13_ENHANCED) == len(FEATURE_COLUMNS_LGB_V13) + len(FEATURES_ENHANCED)


def test_compute_features_v13_enhanced_no_inf():
    m5 = _sample_m5(250)
    feats = compute_features_v13(m5, include_enhanced=True)
    assert len(feats) > 0
    assert list(feats.columns) == FEATURE_COLUMNS_V13_ENHANCED
    assert not np.isinf(feats.to_numpy()).any()
    assert feats.isna().sum().sum() == 0


def test_add_enhanced_features_keys():
    m5 = _sample_m5(120)
    base = compute_features(m5, include_reversal=True)
    work = m5.loc[base.index].copy()
    for c in FEATURE_COLUMNS_LGB_V13:
        work[c] = base[c]
    out = add_enhanced_features(work)
    for c in FEATURES_ENHANCED:
        assert c in out.columns
