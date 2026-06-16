"""V13 训练管线共享：数据加载、评估、保存。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from zhulong.analysis.feature_engineering import FEATURES_ENHANCED, FEATURES_ENHANCED_WITH_LEVELS
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
from zhulong.training.lgb.splits import split_indices
from zhulong.training.v13.triple import TEST_END, TEST_START

logger = logging.getLogger(__name__)

FEATURE_COLUMNS_V13_ENHANCED = list(FEATURE_COLUMNS_LGB_V13) + list(FEATURES_ENHANCED)
FEATURE_COLUMNS_V13_QUALITY = list(FEATURE_COLUMNS_LGB_V13) + list(FEATURES_ENHANCED_WITH_LEVELS)


def _undersample_flat_1_1_2(
    X: pd.DataFrame,
    y: np.ndarray,
    flat_mult: float = 2.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    ix_long = np.where(y == 1)[0]
    ix_short = np.where(y == 2)[0]
    ix_flat = np.where(y == 0)[0]
    n_base = min(len(ix_long), len(ix_short))
    if n_base == 0:
        return X.iloc[:0], y[:0]
    pick_long = rng.choice(ix_long, n_base, replace=False)
    pick_short = rng.choice(ix_short, n_base, replace=False)
    n_flat = min(int(flat_mult * n_base), len(ix_flat))
    pick_flat = rng.choice(ix_flat, n_flat, replace=False)
    pick = np.concatenate([pick_long, pick_short, pick_flat])
    rng.shuffle(pick)
    return X.iloc[pick], y[pick]


def compute_features_v13(
    m5: pd.DataFrame,
    *,
    include_reversal: bool = True,
    include_enhanced: bool = False,
    include_key_levels: bool = False,
    root: Path | None = None,
) -> pd.DataFrame:
    from zhulong.analysis.feature_engineering import add_enhanced_features

    if include_enhanced or include_key_levels:
        df = m5.copy()
        base = compute_features(m5, include_reversal=include_reversal)
        work = df.loc[base.index].copy()
        for c in FEATURE_COLUMNS_LGB_V13:
            if c in base.columns:
                work[c] = base[c]
        work = add_enhanced_features(work, root=root, include_key_levels=include_key_levels)
        cols = FEATURE_COLUMNS_V13_QUALITY if include_key_levels else FEATURE_COLUMNS_V13_ENHANCED
        feats = work[cols].replace([np.inf, -np.inf], np.nan).dropna()
        logger.info("v13 features: %s x %s", len(feats), len(cols))
        return feats
    return compute_features(m5, include_reversal=include_reversal)


def load_training_bundle(
    root: Path,
    symbol: str = "XAUUSD",
    labels_path: str = "data/training/XAUUSD_triple_v3.csv",
    train_balanced_path: str = "data/training/train_balanced_v3.csv",
    include_enhanced: bool = False,
    include_key_levels: bool = False,
    label_type: str = "triple",
    quality_labels_path: str = "data/labels/XAUUSD_quality_labels.parquet",
    refresh_features: bool = False,
) -> dict[str, Any]:
    m5 = load_vendor_csv(root / "data" / "training" / "lgb" / symbol / f"{symbol}_M5.csv")
    if include_key_levels or label_type == "quality":
        sub = "v13_quality"
        use_enhanced = True
        use_key = True
    elif include_enhanced:
        sub = "v13_enhanced"
        use_enhanced = True
        use_key = False
    else:
        sub = "v13"
        use_enhanced = False
        use_key = False

    feat_cache = root / "data" / "training" / sub / symbol / "features.parquet"
    if feat_cache.is_file() and not refresh_features:
        feats = pd.read_parquet(feat_cache)
    else:
        feats = compute_features_v13(
            m5, include_enhanced=use_enhanced, include_key_levels=use_key, root=root,
        )
        feat_cache.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(feat_cache)

    if label_type == "quality":
        lab = pd.read_parquet(root / quality_labels_path)
        if "label" not in lab.columns:
            lab = lab.rename(columns={lab.columns[0]: "label"})
    else:
        lab = pd.read_csv(root / labels_path, index_col=0, parse_dates=True)
    aligned = feats.join(lab[["label"]], how="inner").dropna(subset=["label"])
    aligned["label"] = aligned["label"].astype(int)
    if label_type == "quality":
        cols = list(FEATURE_COLUMNS_V13_QUALITY)
    else:
        cols = list(FEATURE_COLUMNS_V13_ENHANCED if include_enhanced else FEATURE_COLUMNS_LGB_V13)

    splits = split_indices(aligned.index)
    va_ix = splits.val.intersection(aligned.index)
    te_ix = aligned.index[(aligned.index >= TEST_START) & (aligned.index <= TEST_END)]
    tr_ix = splits.train.intersection(aligned.index)
    if label_type == "quality":
        X_tr = aligned.loc[tr_ix, cols]
        y_tr = aligned.loc[tr_ix, "label"].values.astype(int)
        X_bal, y_bal = _undersample_flat_1_1_2(X_tr, y_tr, flat_mult=2.0)
        train_bal = X_bal.copy()
        train_bal["label"] = y_bal
    else:
        train_bal_path = root / train_balanced_path
        train_bal_raw = (
            pd.read_csv(train_bal_path, index_col=0, parse_dates=True)
            if train_bal_path.is_file() else pd.DataFrame()
        )
        need_resample = (
            include_enhanced or include_key_levels
            or train_bal_raw.empty
            or not all(c in train_bal_raw.columns for c in cols)
        )
        if need_resample:
            X_tr = aligned.loc[tr_ix, cols]
            y_tr = aligned.loc[tr_ix, "label"].values.astype(int)
            X_bal, y_bal = _undersample_flat_1_1_2(X_tr, y_tr, flat_mult=2.0)
            train_bal = X_bal.copy()
            train_bal["label"] = y_bal
        else:
            bal_ix = train_bal_raw.index.intersection(aligned.index)
            train_bal = aligned.loc[bal_ix, cols + ["label"]]

    return {
        "m5": m5,
        "feats": feats,
        "aligned": aligned,
        "cols": cols,
        "va_ix": va_ix,
        "te_ix": te_ix,
        "train_bal": train_bal,
        "splits": splits,
    }


@dataclass
class TrainArtifacts:
    model: Any
    model_type: str
    feature_columns: list[str]
    long_threshold: float
    short_threshold: float
    metrics: dict[str, Any]


def save_artifacts(
    out_dir: Path,
    artifacts: TrainArtifacts,
    params: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if artifacts.model_type == "xgb":
        artifacts.model.save_model(str(out_dir / "xgb_triple_v3.json"))
    elif artifacts.model_type == "lgb":
        artifacts.model.booster_.save_model(str(out_dir / "lgb_triple_v3.txt"))
        joblib.dump(artifacts.model, out_dir / "lgb_triple_v3.pkl")
    elif artifacts.model_type == "meta":
        joblib.dump(artifacts.model, out_dir / "meta_label.pkl")

    meta = {
        "model_type": artifacts.model_type,
        "params": params,
        "feature_columns": artifacts.feature_columns,
        "long_threshold": artifacts.long_threshold,
        "short_threshold": artifacts.short_threshold,
        "metrics": artifacts.metrics,
        **(extra or {}),
    }
    joblib.dump(meta, out_dir / "params_v13.pkl")
    (out_dir / "feature_columns.json").write_text(
        json.dumps(artifacts.feature_columns, indent=2), encoding="utf-8"
    )
    (out_dir / "config_v13.json").write_text(
        json.dumps(
            {
                "long_threshold": artifacts.long_threshold,
                "short_threshold": artifacts.short_threshold,
                "model_type": artifacts.model_type,
                "metrics": artifacts.metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
