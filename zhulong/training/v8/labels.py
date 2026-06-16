"""v8 标签：回归（未来收益率）+ 二分类（做多）。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON = 12
GAIN_THRESHOLD = 0.002  # 0.20%


def generate_v8_labels(
    m5: pd.DataFrame,
    horizon: int = HORIZON,
    gain_threshold: float = GAIN_THRESHOLD,
) -> pd.DataFrame:
    close = m5["close"]
    fut_ret = (close.shift(-horizon) - close) / close.replace(0, np.nan)
    label_cls = (fut_ret > gain_threshold).astype(np.int8)
    out = pd.DataFrame(
        {
            "label_cls": label_cls,
            "label_reg": fut_ret.astype(np.float32),
        },
        index=m5.index,
    )
    valid = ~fut_ret.isna()
    pos = int((label_cls[valid] == 1).sum())
    logger.info(
        "v8 labels h=%s gain=%.2f%% long=%.2f%% n=%s",
        horizon,
        gain_threshold * 100,
        100.0 * pos / max(valid.sum(), 1),
        valid.sum(),
    )
    return out
