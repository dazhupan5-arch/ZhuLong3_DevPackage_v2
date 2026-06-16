#!/usr/bin/env python3
"""LSTM 样本外回测（SL/TP 与 v6.1 一致）。"""

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

from zhulong.training.lgb.backtest import DEFAULT_COOLDOWN_BARS, backtest_signals, simulate_trade, _atr_series
from zhulong.training.lgb.data_io import load_vendor_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data/training/lstm/XAUUSD")
    parser.add_argument("--model", default="models/XAUUSD/lstm/lstm_model.keras")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--max-hold-bars", type=int, default=24)
    parser.add_argument("--cooldown-bars", type=int, default=DEFAULT_COOLDOWN_BARS)
    args = parser.parse_args()

    root = _ROOT
    cfg_path = root / "models" / args.symbol / "lstm" / "config_v7.json"
    thr = args.threshold
    if thr < 0 and cfg_path.is_file():
        thr = json.loads(cfg_path.read_text(encoding="utf-8"))["threshold"]

    data = np.load(root / args.data_dir / f"{args.split}.npz")
    times = pd.to_datetime(data["times"], unit="s")
    model = load_model(root / args.model)
    proba = model.predict(data["X"], batch_size=512, verbose=0).ravel()
    dirs = np.where(proba >= thr, 1, 0)

    m5 = load_vendor_csv(root / "data" / "training" / "lgb" / args.symbol / f"{args.symbol}_M5.csv")
    bt = backtest_signals(m5, times, dirs, max_hold=args.max_hold_bars, cooldown_bars=args.cooldown_bars)

    atr_s = _atr_series(m5)
    close = m5["close"]
    rs: list[float] = []
    last_idx = -10**9
    for t, d in zip(times, dirs):
        if d == 0 or t not in m5.index:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice) or (idx - last_idx) < args.cooldown_bars:
            continue
        if idx + 1 >= len(m5):
            continue
        a = float(atr_s.iloc[idx])
        if a <= 0:
            continue
        entry = float(close.iloc[idx])
        end = min(idx + 1 + args.max_hold_bars, len(m5))
        rs.append(
            simulate_trade(
                1,
                entry,
                a,
                m5["high"].iloc[idx + 1 : end].to_numpy(),
                m5["low"].iloc[idx + 1 : end].to_numpy(),
                m5["close"].iloc[idx + 1 : end].to_numpy(),
                args.max_hold_bars,
            )
        )
        last_idx = idx
    equity = np.cumsum(rs).tolist() if rs else []

    report = {
        "version": "v7",
        "split": args.split,
        "threshold": thr,
        "max_hold_bars": args.max_hold_bars,
        "cooldown_bars": args.cooldown_bars,
        "backtest": bt,
        "equity_curve_r": equity[-100:] if len(equity) > 100 else equity,
        "equity_final_r": float(equity[-1]) if equity else 0.0,
    }
    out_dir = root / "data" / "training" / "reports" / "lstm" / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_{args.split}_v7.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== v7 LSTM Backtest {args.split} thr={thr:.2f} ===")
    for k, v in bt.items():
        print(f"  {k}: {v}")
    print(f"  equity_final_r: {report['equity_final_r']}")
    print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
