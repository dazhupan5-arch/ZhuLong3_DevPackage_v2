"""元标签信号过滤：主模型方向 + 元模型置信度二次筛选。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from zhulong.training.lgb.backtest import _atr_series

META_EXTRA = ["proba_flat", "proba_long", "proba_short", "signal_conf", "atr_pct", "adx_norm"]


def load_meta_config(root: Path) -> dict[str, Any]:
    cfg_path = root / "config" / "config_meta_label.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    pkl_path = root / "models" / "XAUUSD" / "meta_label" / "meta_config.pkl"
    if pkl_path.is_file():
        pkl_cfg = joblib.load(pkl_path)
        cfg.setdefault("feature_columns", pkl_cfg.get("feature_columns"))
        cfg.setdefault("meta_threshold", pkl_cfg.get("meta_threshold", 0.6))
    cfg["threshold"] = cfg.get("threshold", cfg.get("meta_threshold", 0.6))
    cfg["model_path"] = cfg.get("model_path", cfg.get("meta_model_path", "models/XAUUSD/meta_label/meta_label.pkl"))
    cfg["enabled"] = cfg.get("enabled", True)
    return cfg


def load_meta_model(root: Path, cfg: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    cfg = cfg or load_meta_config(root)
    model_path = root / cfg["model_path"]
    if not model_path.is_file():
        raise FileNotFoundError(f"元模型不存在: {model_path}")
    model = joblib.load(model_path)
    meta_cols = cfg.get("feature_columns")
    if not meta_cols:
        cols_path = root / "models" / "XAUUSD" / "meta_label" / "meta_feature_columns.json"
        if cols_path.is_file():
            meta_cols = json.loads(cols_path.read_text(encoding="utf-8"))
    if not meta_cols:
        raise ValueError("元模型特征列未配置")
    cfg = {**cfg, "feature_columns": meta_cols}
    return model, cfg


def build_meta_frame(
    feats: pd.DataFrame,
    times: pd.DatetimeIndex,
    proba: np.ndarray,
    m5: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """构建元模型输入特征（主模型特征 + 概率/市场状态）。"""
    sub = feats.loc[times, feature_cols].copy()
    sub["proba_flat"] = proba[:, 0]
    sub["proba_long"] = proba[:, 1]
    sub["proba_short"] = proba[:, 2]
    sub["signal_conf"] = np.max(proba[:, 1:3], axis=1)
    atr = _atr_series(m5)
    close = m5["close"].reindex(times)
    sub["atr_pct"] = (atr.reindex(times) / close.replace(0, np.nan)).fillna(0.0).values
    for c in META_EXTRA:
        if c not in sub.columns:
            sub[c] = 0.0
    return sub


def apply_meta_filter(
    directions: np.ndarray,
    meta_model: Any,
    meta_cols: list[str],
    meta_frame: pd.DataFrame,
    threshold: float = 0.6,
) -> tuple[np.ndarray, dict[str, float]]:
    """对非零方向信号应用元标签过滤，返回过滤后方向及统计。"""
    out = directions.copy()
    sig_ix = np.where(out != 0)[0]
    if len(sig_ix) == 0:
        return out, {"n_before": 0, "n_after": 0, "filter_rate": 0.0}

    X_meta = meta_frame.iloc[sig_ix][meta_cols]
    meta_prob = meta_model.predict_proba(X_meta)[:, 1]
    keep = meta_prob >= threshold
    filtered_count = int((~keep).sum())
    for j, i in enumerate(sig_ix):
        if not keep[j]:
            out[i] = 0

    n_before = len(sig_ix)
    n_after = int((out != 0).sum())
    return out, {
        "n_before": n_before,
        "n_after": n_after,
        "filter_rate": filtered_count / max(n_before, 1),
        "meta_threshold": threshold,
    }
