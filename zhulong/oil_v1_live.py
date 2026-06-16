"""USOIL v1 实机推理（供 inference_engine 调用）。"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from zhulong.inference.oil_v1 import OilV1Inference, load_oil_v1_config
from zhulong.live_oil_features import build_live_oil_row
from zhulong.utils.paths import model_dir_for_symbol

logger = logging.getLogger(__name__)


def validate_oil_v1_artifacts(symbol: str) -> bool:
    d = model_dir_for_symbol(symbol)
    required = [
        d / "manifest.json",
        d / "v1" / "xgb_triple_oil.json",
        d / "v1" / "oil_v1_meta.pkl",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        logger.error("品种 %s oil_v1 缺少: %s", symbol, [p.name for p in missing])
        return False
    cols_ok = (d / "v1" / "feature_columns.json").is_file() or (d / "feature_columns.json").is_file()
    if not cols_ok:
        logger.error("品种 %s oil_v1 缺少 feature_columns.json", symbol)
        return False
    return True


def load_oil_v1_bundle(symbol: str) -> OilV1Inference:
    cfg = load_oil_v1_config()
    cfg.symbol = symbol
    cfg.training_symbol = symbol
    cfg.model_path = f"models/{symbol}/v1/xgb_triple_oil.json"
    cfg.meta_path = f"models/{symbol}/v1/oil_v1_meta.pkl"
    cfg.feature_columns = f"models/{symbol}/v1/feature_columns.json"
    eng = OilV1Inference(cfg)
    eng.load()
    return eng


def predict_oil_v1(symbol: str, engine: OilV1Inference, m5: pd.DataFrame | None = None) -> dict:
    row, _cols, m5_df, _ = build_live_oil_row(
        symbol,
        m5=m5,
        broker_symbol=engine.cfg.broker_symbol,
        imf_cache=model_dir_for_symbol(symbol) / "imf_vmd.parquet",
    )
    sig = engine.build_signal(m5_df, row)
    proba = sig.probabilities or [0.0, 0.0, 0.0]
    while len(proba) < 3:
        proba.append(0.0)

    if sig.direction == "buy":
        direction, confidence = 1, float(sig.confidence)
    elif sig.direction == "sell":
        direction, confidence = -1, float(sig.confidence)
    else:
        direction = 0
        confidence = float(sig.confidence)

    expected_return = confidence * 0.25 if direction != 0 else 0.0
    return {
        "direction": direction,
        "confidence": confidence,
        "entry_offset": 0.0,
        "expected_return": expected_return,
        "probabilities": proba[:3],
        "reason": sig.reject_reason or "",
    }
