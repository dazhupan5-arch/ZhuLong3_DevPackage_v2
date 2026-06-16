#!/usr/bin/env python3
"""v4 验证集阈值扫描（做多/做空类别概率）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from zhulong.training.lgb.splits import split_indices
from zhulong.training.lgb.train import (
    CLASS_LONG,
    CLASS_SHORT,
    format_threshold_table,
    threshold_sweep,
    tune_threshold,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--model", default="", help="lgb_multiclass.txt 路径")
    parser.add_argument("--meta", default="", help="lgb_meta.pkl 路径")
    parser.add_argument("--side", choices=["long", "short", "both"], default="both")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--gain", type=float, default=0.0025)
    parser.add_argument("--target-precision", type=float, default=0.50)
    parser.add_argument("--max-signals-per-day", type=float, default=8.0)
    args = parser.parse_args()

    root = _ROOT
    model_dir = root / "models" / args.symbol / "lgb"
    data_dir = root / "data" / "training" / "lgb" / args.symbol
    model_path = Path(args.model) if args.model else model_dir / "lgb_multiclass.txt"
    meta_path = Path(args.meta) if args.meta else model_dir / "lgb_meta.pkl"

    meta = joblib.load(meta_path)
    cols = meta["feature_columns"]
    feats = pd.read_parquet(data_dir / f"{args.symbol}_features.parquet")
    labels_csv = root / "data" / "training" / f"{args.symbol}_labeled_v4_2.csv"
    labels = pd.read_csv(labels_csv, index_col=0, parse_dates=True) if labels_csv.is_file() else pd.read_parquet(data_dir / f"{args.symbol}_labels.parquet")
    aligned = feats.join(labels, how="inner")
    va_ix = split_indices(aligned.index).val.intersection(aligned.index)
    val = aligned.loc[va_ix]
    y = val["label"].values

    booster = lgb.Booster(model_file=str(model_path))
    proba = booster.predict(val[cols])
    if proba.ndim == 1:
        proba = proba.reshape(-1, 3)

    out = {"long": [], "short": []}
    report_dir = root / "data" / "training" / "reports" / "lgb" / args.symbol
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.side in ("long", "both"):
        y_long = (y == 1).astype(int)
        p_long = proba[:, CLASS_LONG]
        print("=== LONG proba ===")
        print(f"min={p_long.min():.4f} max={p_long.max():.4f} mean={p_long.mean():.4f}")
        rows = threshold_sweep(y_long, p_long, va_ix)
        best, _ = tune_threshold(
            y_long, p_long, va_ix,
            target_precision=args.target_precision,
            max_signals_per_day=args.max_signals_per_day / 2,
        )
        print(format_threshold_table(rows))
        print(f"selected: thr={best.threshold:.2f} prec={best.precision:.3f} rec={best.recall:.3f} sig/day={best.signals_per_day:.1f}")
        out["long"] = [r.__dict__ for r in rows]
        out["long_best"] = best.__dict__

    if args.side in ("short", "both"):
        y_short = (y == -1).astype(int)
        p_short = proba[:, CLASS_SHORT]
        print("\n=== SHORT proba ===")
        print(f"min={p_short.min():.4f} max={p_short.max():.4f} mean={p_short.mean():.4f}")
        rows = threshold_sweep(y_short, p_short, va_ix)
        best, _ = tune_threshold(
            y_short, p_short, va_ix,
            target_precision=args.target_precision,
            max_signals_per_day=args.max_signals_per_day / 2,
        )
        print(format_threshold_table(rows))
        print(f"selected: thr={best.threshold:.2f} prec={best.precision:.3f} rec={best.recall:.3f} sig/day={best.signals_per_day:.1f}")
        out["short"] = [r.__dict__ for r in rows]
        out["short_best"] = best.__dict__

    out_path = report_dir / "threshold_tune_v4_2.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
