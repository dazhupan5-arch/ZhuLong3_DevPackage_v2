"""v11 三分类标签：0=观望, 1=做多, 2=做空。"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HORIZON = 12
DEFAULT_GAIN = 0.002


def generate_triple_labels(
    m5: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    gain: float = DEFAULT_GAIN,
) -> pd.DataFrame:
    close = m5["close"]
    ret = (close.shift(-horizon) - close) / close.replace(0, np.nan)
    labels = np.zeros(len(m5), dtype=np.int8)
    labels[ret > gain] = 1
    labels[ret < -gain] = 2
    out = pd.DataFrame({"label": labels}, index=m5.index)
    n = max(int((~ret.isna()).sum()), 1)
    logger.info(
        "triple h=%s gain=%.2f%% long=%.1f%% short=%.1f%% flat=%.1f%%",
        horizon,
        gain * 100,
        100 * (labels == 1).sum() / n,
        100 * (labels == 2).sum() / n,
        100 * (labels == 0).sum() / n,
    )
    return out
