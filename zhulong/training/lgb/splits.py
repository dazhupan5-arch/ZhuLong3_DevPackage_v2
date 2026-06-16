"""按 DeepSeek 方案固定时间划分。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

TRAIN_END = pd.Timestamp("2024-06-30 23:59:59")
VAL_START = pd.Timestamp("2024-07-02 00:00:00")  # 留 1 天间隙
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
TEST1_START = pd.Timestamp("2025-01-01 00:00:00")
TEST1_END = pd.Timestamp("2025-06-30 23:59:59")
STRESS_START = pd.Timestamp("2020-03-01 00:00:00")
STRESS_END = pd.Timestamp("2020-04-30 23:59:59")


@dataclass
class DataSplits:
    train: pd.Index
    val: pd.Index
    test1: pd.Index
    stress: pd.Index


def split_indices(index: pd.DatetimeIndex) -> DataSplits:
    train = index[index <= TRAIN_END]
    val = index[(index >= VAL_START) & (index <= VAL_END)]
    test1 = index[(index >= TEST1_START) & (index <= TEST1_END)]
    stress = index[(index >= STRESS_START) & (index <= STRESS_END)]
    return DataSplits(train=train, val=val, test1=test1, stress=stress)


def split_report(splits: DataSplits) -> dict[str, int]:
    return {
        "train": len(splits.train),
        "val": len(splits.val),
        "test1": len(splits.test1),
        "stress": len(splits.stress),
    }
