"""结构分析器单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zhulong.agent.structure_analyzer import FEATURE_DIM, StructureAnalyzer


def _synthetic_m5(n: int = 250) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 2000 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    vol = np.random.randint(100, 500, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_structure_output_shape():
    m5 = _synthetic_m5()
    sa = StructureAnalyzer({"lookback": 250})
    out = sa.compute({"M5": m5})
    assert out.shape[1] == FEATURE_DIM
    assert out.shape[0] == len(m5)
    assert not np.isnan(out).any()


def test_structure_latest():
    m5 = _synthetic_m5(120)
    sa = StructureAnalyzer()
    row = sa.compute_latest({"M5": m5})
    assert row.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(row))


def test_structure_mtf_dims_populated():
    m5 = _synthetic_m5(500)
    sa = StructureAnalyzer({"lookback": 250})
    out = sa.compute_all(m5, progress_every=0)
    mtf_block = out[:, 21:30]
    assert np.any(mtf_block != 0.0)
