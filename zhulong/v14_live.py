#!/usr/bin/env python3
"""V14 实机推理：XGBoost V14 方向预测 + 正确列映射。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from zhulong.training.lgb.backtest import _atr_series
from zhulong.utils.paths import install_dir

logger = logging.getLogger(__name__)

MIN_ATR_PCT = 0.001  # 0.1%，原 v10.backtest 常量

DEFAULT_LONG_THR = 0.70
DEFAULT_SHORT_THR = 0.70


def _read_json_file(path: Path) -> dict:
    """读取 JSON 配置，兼容 UTF-8 / GBK 编码。"""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"无法解析 JSON: {path}")
DEFAULT_SL_ATR = 1.2
DEFAULT_TP_ATR = 2.0


def validate_v14_artifacts(symbol: str = "XAUUSD", root: Path | None = None) -> bool:
    """检查 V14 模型四件套（优先 models/{symbol}/v14/，兼容根目录布局）。"""
    r = resolve_model_root(root)
    for sub in ("v14", ""):
        d = r / "models" / symbol / sub if sub else r / "models" / symbol
        names = ("xgb_v14.json", "feature_columns.json")
        meta_names = ("v14_meta.pkl", "v12_meta.pkl")
        if not all((d / n).is_file() for n in names):
            continue
        if any((d / m).is_file() for m in meta_names):
            return True
    logger.error("V14 缺少 %s 模型文件", symbol)
    return False


def resolve_model_root(root: Path | None = None) -> Path:
    if root is None:
        return install_dir()
    if isinstance(root, str):
        return Path(root)
    return root


def load_v15_bundle(
    symbol: str = "XAUUSD",
    root: Path | None = None,
) -> dict:
    """加载 V15 模型包（xgb_v15.json + v15_meta.pkl + feature_columns.json）。"""
    r = resolve_model_root(root)
    d = r / "models" / symbol / "v15"
    model_file = d / "xgb_v15.json"
    if not model_file.is_file():
        raise FileNotFoundError(f"V15 模型不存在: {model_file}")

    model = xgb.XGBClassifier()
    model.load_model(str(model_file))
    cols = json.loads((d / "feature_columns.json").read_text(encoding="utf-8"))

    meta = {}
    meta_path = d / "v15_meta.pkl"
    if meta_path.is_file():
        import joblib
        meta = joblib.load(meta_path)

    config = {}
    config_path = d / "config_v15.json"
    if config_path.is_file():
        config = _read_json_file(config_path)

    long_thr = float(config.get("long_threshold", meta.get("long_threshold", 0.52)))
    short_thr = float(config.get("short_threshold", meta.get("short_threshold", 0.48)))

    logger.info(
        "V15 loaded: %s, %d features, long_thr=%.2f short_thr=%.2f",
        model_file, len(cols), long_thr, short_thr,
    )
    return {
        "model": model,
        "columns": cols,
        "long_thr": long_thr,
        "short_thr": short_thr,
        "sl_atr": float(meta.get("params", {}).get("sl_atr", DEFAULT_SL_ATR)),
        "tp_atr": float(meta.get("params", {}).get("tp_atr", DEFAULT_TP_ATR)),
        "model_version": "v15",
    }


def load_v14_bundle(
    symbol: str = "XAUUSD",
    model_subdir: str = "v14",
    root: Path | None = None,
) -> dict:
    """加载 V14 模型包（xgb_v14.json + v14_meta.pkl + feature_columns.json）。"""
    r = resolve_model_root(root)
    d = r / "models" / symbol / model_subdir
    if not (d / "xgb_v14.json").is_file() and model_subdir:
        flat = r / "models" / symbol
        if (flat / "xgb_v14.json").is_file():
            d = flat
    if not (d / "xgb_v14.json").is_file():
        raise FileNotFoundError(f"V14 模型不存在: {d / 'xgb_v14.json'}")

    model = xgb.XGBClassifier()
    model.load_model(str(d / "xgb_v14.json"))
    cols = json.loads((d / "feature_columns.json").read_text(encoding="utf-8"))

    meta = {}
    meta_path = d / "v14_meta.pkl"
    if not meta_path.is_file():
        meta_path = d / "v12_meta.pkl"
    if meta_path.is_file():
        import joblib
        meta = joblib.load(meta_path)

    config_path = d / "config_v14.json"
    config = {}
    if config_path.is_file():
        config = _read_json_file(config_path)

    long_thr = float(config.get("long_threshold", meta.get("long_threshold", DEFAULT_LONG_THR)))
    short_thr = float(config.get("short_threshold", meta.get("short_threshold", DEFAULT_SHORT_THR)))

    logger.info(
        "V14 loaded: %s, %d features, long_thr=%.2f short_thr=%.2f",
        d / "xgb_v14.json", len(cols), long_thr, short_thr,
    )
    return {
        "model": model,
        "columns": cols,
        "long_thr": long_thr,
        "short_thr": short_thr,
        "sl_atr": float(meta.get("params", {}).get("sl_atr", DEFAULT_SL_ATR)),
        "tp_atr": float(meta.get("params", {}).get("tp_atr", DEFAULT_TP_ATR)),
    }


def _proba_to_direction_v14(
    proba: np.ndarray,
    long_thr: float,
    short_thr: float,
) -> int:
    """
    V14 列映射：0=flat, 1=long, 2=short。
    与训练脚本 train_v14.py 的 to_multiclass_v14 / proba_to_directions_v14 一致。
    """
    p_flat, p_long, p_short = float(proba[0]), float(proba[1]), float(proba[2])
    if p_long >= long_thr and p_long >= p_short and p_long > p_flat:
        return 1
    if p_short >= short_thr and p_short >= p_long and p_short > p_flat:
        return -1
    return 0


class V14LiveSignal:
    __slots__ = ("direction", "confidence", "entry", "sl", "tp",
                 "signal_id", "symbol", "probabilities", "reject_reason")

    def __init__(
        self,
        direction: str,
        confidence: float,
        entry: float,
        sl: float,
        tp: float,
        signal_id: str,
        symbol: str,
        probabilities: list[float],
        reject_reason: str = "",
    ) -> None:
        self.direction = direction
        self.confidence = confidence
        self.entry = entry
        self.sl = sl
        self.tp = tp
        self.signal_id = signal_id
        self.symbol = symbol
        self.probabilities = probabilities
        self.reject_reason = reject_reason

    def to_draw_payload(self, expiry_minutes: int = 240) -> dict:
        if self.direction == "flat":
            return {}
        return {
            "action": "draw_signal",
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "confidence": round(self.confidence, 4),
            "expiry_minutes": expiry_minutes,
        }


def build_live_v14_features(
    symbol: str = "XAUUSD",
    m5: pd.DataFrame | None = None,
    m5_bars: int = 2000,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    构建 V14 实机特征（与训练完全一致的 68 维）。
    返回 (feature_row, feature_columns, m5_dataframe)。
    """
    if m5 is None:
        from zhulong.live_v8_features import m5_from_mt5
        m5_raw = m5_from_mt5(symbol, bars=m5_bars)
    else:
        m5_raw = m5

    from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
    feats = compute_features(m5_raw, include_mtf=True, include_reversal=True)
    cols = list(FEATURE_COLUMNS_LGB_V13)

    if feats.empty:
        raise RuntimeError("V14 特征计算为空（数据不足）")

    row = feats.iloc[-1][cols].to_numpy(dtype=np.float32)
    feats_row = feats.iloc[[-1]][cols]
    logger.debug("V14 features built: %d dims, time=%s", len(cols), feats.index[-1])
    return row, cols, m5_raw, feats_row


def predict_v14(
    bundle: dict,
    feature_row: np.ndarray,
    m5: pd.DataFrame,
    bar_time: pd.Timestamp | None = None,
    feats_df: pd.DataFrame | None = None,
) -> V14LiveSignal:
    """单次 V14 推理，返回 V14LiveSignal。"""
    model: xgb.XGBClassifier = bundle["model"]
    feat_cols: list[str] = bundle["columns"]
    long_thr: float = bundle["long_thr"]
    short_thr: float = bundle["short_thr"]

    if bar_time is None:
        bar_time = m5.index[-1]

    close = float(m5.loc[bar_time, "close"])

    # ATR 检查
    atr_s = _atr_series(m5)
    idx = m5.index.get_loc(bar_time)
    if isinstance(idx, slice):
        idx = -1
    atr = float(atr_s.iloc[idx])
    if atr <= 0 or (atr / close) < MIN_ATR_PCT:
        return V14LiveSignal("flat", 0.0, close, 0.0, 0.0, "", "XAUUSD", [], "atr_too_low")

    # 推理
    x = np.asarray(feature_row[:len(feat_cols)], dtype=np.float32).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    direction_int = _proba_to_direction_v14(proba, long_thr, short_thr)

    if direction_int == 0:
        p_flat, p_long, p_short = float(proba[0]), float(proba[1]), float(proba[2])
        conf = max(p_flat, p_long, p_short)
        return V14LiveSignal("flat", conf, close, 0.0, 0.0, "", "XAUUSD",
                             proba.tolist(), "below_threshold")

    side = "buy" if direction_int > 0 else "sell"
    sl_mult = bundle.get("sl_atr", DEFAULT_SL_ATR)
    tp_mult = bundle.get("tp_atr", DEFAULT_TP_ATR)

    if side == "buy":
        sl = close - atr * sl_mult
        tp = close + atr * tp_mult
        conf = float(proba[1])
    else:
        sl = close + atr * sl_mult
        tp = close - atr * tp_mult
        conf = float(proba[2])

    now_utc = datetime.now(timezone.utc)
    sig_id = f"{now_utc.strftime('%Y%m%d_%H%M')}_XAUUSD_{side}"

    return V14LiveSignal(side, conf, close, sl, tp, sig_id, "XAUUSD", proba.tolist())
