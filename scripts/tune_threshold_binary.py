#!/usr/bin/env python3
"""v5 二分类阈值调优（0.30–0.90）。"""

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

from zhulong.training.lgb.splits import split_indices
from zhulong.training.lgb.train_binary import (
    format_threshold_table,
    threshold_sweep_binary,
    to_binary_long,
    tune_threshold_binary,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-file", default="", help="标签 CSV，默认 profit 或 v5_1")
    parser.add_argument("--profit-labels", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--meta", default="")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--gain", type=float, default=0.0020)
    parser.add_argument("--target-precision", type=float, default=0.45)
    parser.add_argument("--target-recall", type=float, default=0.10)
    parser.add_argument("--max-signals-per-day", type=float, default=8.0)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--acceptance-stage", default="", help="v61 时使用 v6.1 阈值与扫描范围")
    args = parser.parse_args()

    root = _ROOT
    is_v61 = args.acceptance_stage == "v61" or "profit_24" in args.labels_file
    if args.profit_labels or "profit" in args.labels_file:
        lab_path = root / (args.labels_file or "data/training/XAUUSD_labeled_profit.csv")
        out_name = "threshold_tune_v61.json" if is_v61 else "threshold_tune_v6.json"
        tp, tr, sweep_hi = (0.50, 0.15, 0.70) if is_v61 else (0.55, 0.20, 0.90)
    else:
        lab_path = root / (args.labels_file or f"data/training/{args.symbol}_labeled_v5_1_binary.csv")
        out_name = "threshold_tune_v5_1.json"
        tp, tr, sweep_hi = args.target_precision, args.target_recall, 0.90
    if args.target_precision != 0.45:
        tp = args.target_precision
    if args.target_recall != 0.10:
        tr = args.target_recall
    model_dir = root / "models" / args.symbol / "lgb"
    data_dir = root / "data" / "training" / "lgb" / args.symbol
    model_path = Path(args.model) if args.model else (
        model_dir / ("lgb_profit_24.txt" if is_v61 else "lgb_profit.txt")
        if args.profit_labels or "profit" in args.labels_file
        else model_dir / "lgb_binary_v5_1.txt"
    )
    meta_path = Path(args.meta) if args.meta else (
        model_dir / ("lgb_profit_24_meta.pkl" if is_v61 else "lgb_profit_meta.pkl")
        if args.profit_labels or "profit" in args.labels_file
        else model_dir / "lgb_binary_v5_1_meta.pkl"
    )

    meta = joblib.load(meta_path)
    cols = meta["feature_columns"]
    feats = pd.read_parquet(data_dir / f"{args.symbol}_features.parquet")
    lab = pd.read_csv(lab_path, index_col=0, parse_dates=True)
    aligned = feats.join(lab[["label"]], how="inner")
    va_ix = split_indices(aligned.index).val.intersection(aligned.index)
    val = aligned.loc[va_ix]
    y = to_binary_long(val["label"].values)

    booster = lgb.Booster(model_file=str(model_path))
    proba = booster.predict(val[cols])
    print(f"proba: min={proba.min():.4f} max={proba.max():.4f} mean={proba.mean():.4f}")

    rows = threshold_sweep_binary(y, proba, va_ix, hi=sweep_hi)
    best, _ = tune_threshold_binary(
        y, proba, va_ix,
        target_precision=tp,
        target_recall=tr,
        max_signals_per_day=args.max_signals_per_day,
        sweep_hi=sweep_hi,
    )
    print(format_threshold_table(rows))
    print(f"\nselected: thr={best.threshold:.2f} prec={best.precision:.3f} rec={best.recall:.3f} sig/day={best.signals_per_day:.1f}")

    out = {"horizon": args.horizon, "gain": args.gain, "best": best.__dict__, "sweep": [r.__dict__ for r in rows]}
    report_dir = root / "data" / "training" / "reports" / "lgb" / args.symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / out_name
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
