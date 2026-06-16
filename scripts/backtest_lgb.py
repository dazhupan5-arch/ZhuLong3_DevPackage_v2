#!/usr/bin/env python3
"""v4.2 样本外回测（max_hold_bars=48）。"""

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
import pandas as pd

from zhulong.training.lgb.backtest import MAX_HOLD_BARS, backtest_signals
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.splits import split_indices
from zhulong.training.lgb.train import predict_directions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--model", default="")
    parser.add_argument("--meta", default="")
    parser.add_argument("--thr-long", type=float, default=-1.0)
    parser.add_argument("--thr-short", type=float, default=-1.0)
    parser.add_argument("--threshold", type=float, default=-1.0, help="同时用于 long/short")
    parser.add_argument("--split", choices=["test1", "val", "stress"], default="test1")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--gain", type=float, default=0.0025)
    parser.add_argument("--max-hold-bars", type=int, default=MAX_HOLD_BARS)
    args = parser.parse_args()

    root = _ROOT
    model_dir = root / "models" / args.symbol / "lgb"
    data_dir = root / "data" / "training" / "lgb" / args.symbol
    model_path = Path(args.model) if args.model else model_dir / "lgb_multiclass.txt"
    meta_path = Path(args.meta) if args.meta else model_dir / "lgb_meta.pkl"

    meta = joblib.load(meta_path)
    if args.threshold >= 0:
        thr_long = thr_short = args.threshold
    else:
        thr_long = args.thr_long if args.thr_long >= 0 else meta["thr_long"]
        thr_short = args.thr_short if args.thr_short >= 0 else meta["thr_short"]
    cols = meta["feature_columns"]
    max_hold = args.max_hold_bars

    m5 = load_vendor_csv(data_dir / f"{args.symbol}_M5.csv")
    feats = pd.read_parquet(data_dir / f"{args.symbol}_features.parquet")
    labels_csv = root / "data" / "training" / f"{args.symbol}_labeled_v4_2.csv"
    if labels_csv.is_file():
        labels = pd.read_csv(labels_csv, index_col=0, parse_dates=True)
    else:
        labels = pd.read_parquet(data_dir / f"{args.symbol}_labels.parquet")
    aligned = feats.join(labels[["label"] if "label" in labels.columns else labels], how="inner")
    splits = split_indices(aligned.index)
    ix = getattr(splits, args.split).intersection(aligned.index)

    booster = lgb.Booster(model_file=str(model_path))
    X = aligned.loc[ix, cols]
    proba = booster.predict(X)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 3)
    dirs = predict_directions(proba, thr_long, thr_short)
    bt = backtest_signals(m5, ix, dirs, max_hold=max_hold)

    report = {
        "version": "v4.2",
        "split": args.split,
        "horizon": args.horizon,
        "gain": args.gain,
        "max_hold_bars": max_hold,
        "thr_long": thr_long,
        "thr_short": thr_short,
        "backtest": bt,
    }
    out_dir = root / "data" / "training" / "reports" / "lgb" / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_{args.split}_v4_2.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== Backtest v4.2 {args.split} (hold={max_hold} bars) ===")
    for k, v in bt.items():
        print(f"  {k}: {v}")
    print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
