#!/usr/bin/env python3
"""LSTM 验证集阈值调优。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from zhulong.training.lgb.train_binary import format_threshold_table, tune_threshold_binary
from zhulong.training.lgb.train_binary import to_binary_long


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/training/lstm/XAUUSD")
    parser.add_argument("--model", default="models/XAUUSD/lstm/lstm_model.keras")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--target-precision", type=float, default=0.50)
    parser.add_argument("--target-recall", type=float, default=0.20)
    parser.add_argument("--max-signals-per-day", type=float, default=8.0)
    args = parser.parse_args()

    root = _ROOT
    val = np.load(root / args.data_dir / "val.npz")
    X_va, y_va = val["X"], val["y"]
    val_times = pd.to_datetime(val["times"], unit="s")
    model = load_model(root / args.model)
    proba = model.predict(X_va, batch_size=512, verbose=0).ravel()
    print(f"proba: min={proba.min():.4f} max={proba.max():.4f} mean={proba.mean():.4f}")

    rows = __import__(
        "zhulong.training.lgb.train_binary", fromlist=["threshold_sweep_binary"]
    ).threshold_sweep_binary(to_binary_long(y_va.astype(int)), proba, val_times, hi=0.70)
    best, _ = tune_threshold_binary(
        to_binary_long(y_va.astype(int)),
        proba,
        val_times,
        target_precision=args.target_precision,
        target_recall=args.target_recall,
        max_signals_per_day=args.max_signals_per_day,
        sweep_hi=0.70,
    )
    print(format_threshold_table(rows))
    print(
        f"\nselected: thr={best.threshold:.2f} prec={best.precision:.3f} "
        f"rec={best.recall:.3f} sig/day={best.signals_per_day:.1f}"
    )

    out = {"best": best.__dict__, "sweep": [r.__dict__ for r in rows]}
    report_dir = root / "data" / "training" / "reports" / "lstm" / args.symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "threshold_tune_v7.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    meta_path = root / "models" / args.symbol / "lstm" / "config_v7.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"threshold": best.threshold, "model": str(args.model)}, indent=2),
        encoding="utf-8",
    )
    print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
