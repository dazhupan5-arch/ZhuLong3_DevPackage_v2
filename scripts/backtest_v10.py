#!/usr/bin/env python3
"""v10 双向回测：对称做空 + 阈值调优（复用 v9 模型，无需重训）。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.splits import split_indices
from zhulong.training.v9.ensemble import ensemble_proba
from zhulong.training.v10.backtest import (
    backtest_both,
    build_directions,
    short_ground_truth,
    tune_short_threshold,
)

logger = logging.getLogger(__name__)


def load_proba(
    root: Path,
    symbol: str,
    ix: pd.DatetimeIndex,
    feature_columns: list[str],
) -> np.ndarray:
    feats = pd.read_parquet(root / "data" / "training" / "v8" / symbol / "features.parquet")
    sub = feats.loc[ix, feature_columns]
    xgb_m = xgb.XGBClassifier()
    xgb_m.load_model(str(root / "models" / symbol / "v9" / "xgb_classifier.json"))
    lgb_m = lgb.Booster(model_file=str(root / "models" / symbol / "v8" / "lgb_classifier.txt"))
    meta = joblib.load(root / "models" / symbol / "v9" / "v9_meta.pkl")
    w = meta.get("xgb_weight", 0.5)
    xgb_p = xgb_m.predict_proba(sub)[:, 1]
    lgb_p = lgb_m.predict(sub)
    return ensemble_proba(xgb_p, lgb_p, w)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--split", choices=["val", "test1"], default="test1")
    parser.add_argument("--long-threshold", type=float, default=-1.0)
    parser.add_argument("--short-threshold", type=float, default=-1.0)
    parser.add_argument("--mode", choices=["long", "short", "both"], default="both")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = _ROOT
    meta = joblib.load(root / "models" / args.symbol / "v9" / "v9_meta.pkl")
    cfg = json.loads((root / "models" / args.symbol / "v9" / "config_v9.json").read_text(encoding="utf-8"))
    long_thr = args.long_threshold if args.long_threshold >= 0 else meta.get("threshold", cfg.get("threshold", 0.80))
    cols = meta["feature_columns"]

    m5 = load_vendor_csv(root / "data" / "training" / "lgb" / args.symbol / f"{args.symbol}_M5.csv")
    feats = pd.read_parquet(root / "data" / "training" / "v8" / args.symbol / "features.parquet")
    splits = split_indices(feats.index)
    va_ix = splits.val.intersection(feats.index)
    te_ix = splits.test1.intersection(feats.index)

    proba_va = load_proba(root, args.symbol, va_ix, cols)
    y_short_va = short_ground_truth(m5, va_ix)
    short_best, short_rows = tune_short_threshold(proba_va, va_ix, m5, y_short_va)

    short_thr = args.short_threshold if args.short_threshold >= 0 else short_best.threshold
    logger.info(
        "thresholds: long=%.2f short=%.2f (val short prec=%.3f n=%s)",
        long_thr, short_thr, short_best.precision, short_best.n_signals,
    )

    ix = va_ix if args.split == "val" else te_ix
    proba = load_proba(root, args.symbol, ix, cols)
    dirs = build_directions(m5, ix, proba, long_thr, short_thr, mode=args.mode)
    bt = backtest_both(m5, ix, dirs)

    report = {
        "version": "v10",
        "split": args.split,
        "mode": args.mode,
        "long_threshold": long_thr,
        "short_threshold": short_thr,
        "short_tune_val": short_best.__dict__,
        "short_sweep": [r.__dict__ for r in short_rows],
        "backtest": bt,
    }
    out_dir = root / "data" / "training" / "reports" / "v10" / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_{args.split}_v10.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== v10 {args.mode} {args.split} long>={long_thr:.2f} short<={short_thr:.2f} ===")
    for k, v in bt.items():
        print(f"  {k}: {v}")
    print(f"report -> {out_path}")

    (out_dir / "config_v10.json").write_text(
        json.dumps({"long_threshold": long_thr, "short_threshold": short_thr, "mode": args.mode}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
