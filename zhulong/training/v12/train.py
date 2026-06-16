"""v12：v11 三分类 + 做空样本 2× 过采样重训。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zhulong.training.v11.train import V11TrainResult, run_v11_training, save_v11_model

logger = logging.getLogger(__name__)


def boost_short_samples(train_df: pd.DataFrame, short_mult: int = 2, seed: int = 42) -> pd.DataFrame:
    """复制做空样本 (label=2)，使训练集更重视空头模式。"""
    if short_mult <= 1:
        return train_df
    rng = np.random.default_rng(seed)
    y = train_df["label"].values.astype(int)
    ix_short = np.where(y == 2)[0]
    if len(ix_short) == 0:
        return train_df
    extra_n = len(ix_short) * (short_mult - 1)
    extra_ix = rng.choice(ix_short, extra_n, replace=True)
    boosted = pd.concat([train_df, train_df.iloc[extra_ix]], ignore_index=True)
    return boosted.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run_v12_training(
    feats: pd.DataFrame,
    labels: pd.Series,
    feature_columns: list[str],
    m5: pd.DataFrame,
    train_balanced: pd.DataFrame,
    short_mult: int = 2,
    quick: bool = False,
) -> V11TrainResult:
    boosted = boost_short_samples(train_balanced, short_mult=short_mult)
    n_short = int((boosted["label"] == 2).sum())
    n_long = int((boosted["label"] == 1).sum())
    n_flat = int((boosted["label"] == 0).sum())
    logger.info("v12 boosted train: n=%d long=%d short=%d flat=%d", len(boosted), n_long, n_short, n_flat)
    return run_v11_training(feats, labels, feature_columns, m5, boosted, quick=quick)


def save_v12_model(result: V11TrainResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_model(str(out_dir / "xgb_triple.json"))
    import joblib
    import json

    joblib.dump(
        {
            "feature_columns": result.feature_columns,
            "long_threshold": result.thresholds.long_thr,
            "short_threshold": result.thresholds.short_thr,
            "max_hold_bars": 12,
            "cooldown_bars": 18,
            "short_boost": True,
        },
        out_dir / "v12_meta.pkl",
    )
    (out_dir / "config_v12.json").write_text(
        json.dumps(
            {
                "long_threshold": result.thresholds.long_thr,
                "short_threshold": result.thresholds.short_thr,
                "passed": result.report.passed,
                "short_boost": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
