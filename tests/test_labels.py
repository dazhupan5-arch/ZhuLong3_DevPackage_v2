"""标签与过滤链单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from zhulong.training.lgb.labels import generate_labels, LabelConfig


def test_generate_labels_shape():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    m5 = pd.DataFrame({
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100.5, 110.5, n),
        "volume": np.ones(n) * 100,
    }, index=idx)
    labels = generate_labels(m5, config=LabelConfig())
    assert len(labels) == n
    assert "label" in labels.columns
