#!/usr/bin/env python3
"""XGBoost + LightGBM 软投票集成预测。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.v13.train_pipeline import load_training_bundle
from zhulong.training.v11.train import proba_to_directions


def load_proba(model_path: Path, model_type: str, X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if model_type == "lgb":
        m = joblib.load(model_path)
        return m.predict_proba(X[cols])
    m = xgb.XGBClassifier()
    m.load_model(str(model_path))
    return m.predict_proba(X[cols])


def ensemble_proba(probas: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    w = np.array(weights or [1.0 / len(probas)] * len(probas))
    w = w / w.sum()
    out = np.zeros_like(probas[0])
    for p, wi in zip(probas, w):
        out += p * wi
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--xgb", default="models/XAUUSD/triple_barrier/xgb_triple_v3.json")
    parser.add_argument("--lgb", default="models/XAUUSD/lgb/lgb_triple_v3.pkl")
    parser.add_argument("--long-thr", type=float, default=0.5)
    parser.add_argument("--short-thr", type=float, default=0.5)
    parser.add_argument("--enhanced", action="store_true")
    args = parser.parse_args()

    bundle = load_training_bundle(_ROOT, args.symbol, include_enhanced=args.enhanced)
    va_ix, cols = bundle["va_ix"], bundle["cols"]
    X_va = bundle["aligned"].loc[va_ix, cols]

    probas = []
    xgb_p = _ROOT / args.xgb
    lgb_p = _ROOT / args.lgb
    if xgb_p.is_file():
        probas.append(load_proba(xgb_p, "xgb", X_va, cols))
    if lgb_p.is_file():
        probas.append(load_proba(lgb_p, "lgb", X_va, cols))
    if not probas:
        print("未找到模型文件")
        return 1

    proba = ensemble_proba(probas)
    dirs = proba_to_directions(proba, args.long_thr, args.short_thr)
    y_va = bundle["aligned"].loc[va_ix, "label"].values.astype(int)
    pred = np.zeros(len(dirs), dtype=int)
    pred[dirs == 1] = 1
    pred[dirs == -1] = 2
    mask = pred > 0
    prec = float((y_va[mask] == pred[mask]).mean()) if mask.any() else 0.0
    print(f"ensemble val signals={int(mask.sum())} weighted_prec={prec:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
