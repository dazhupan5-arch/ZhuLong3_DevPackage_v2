#!/usr/bin/env python3
"""v5/v6 二分类样本外回测（做多 + 可选冷却）。"""

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

from zhulong.training.lgb.backtest import DEFAULT_COOLDOWN_BARS, backtest_signals, simulate_trade
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.labels_profit import DEFAULT_MAX_HOLD_BARS
from zhulong.training.lgb.splits import split_indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--model", default="models/XAUUSD/lgb/lgb_profit.txt")
    parser.add_argument("--meta", default="models/XAUUSD/lgb/lgb_profit_meta.pkl")
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--split", choices=["test1", "val", "stress"], default="test1")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--max-hold-bars", type=int, default=-1)
    parser.add_argument("--cooldown-bars", type=int, default=DEFAULT_COOLDOWN_BARS)
    parser.add_argument("--profit-labels", action="store_true", default=True)
    args = parser.parse_args()

    root = _ROOT
    data_dir = root / "data" / "training" / "lgb" / args.symbol
    model_path = root / args.model
    meta_path = root / args.meta

    meta = joblib.load(meta_path)
    thr = args.threshold if args.threshold >= 0 else meta["threshold"]
    cols = meta["feature_columns"]
    max_hold = args.max_hold_bars if args.max_hold_bars > 0 else meta.get("max_hold_bars", DEFAULT_MAX_HOLD_BARS)
    cooldown = args.cooldown_bars

    m5 = load_vendor_csv(data_dir / f"{args.symbol}_M5.csv")
    feats = __import__("pandas").read_parquet(data_dir / f"{args.symbol}_features.parquet")
    splits = split_indices(feats.index)
    ix = getattr(splits, args.split).intersection(feats.index)

    booster = lgb.Booster(model_file=str(model_path))
    proba = booster.predict(feats.loc[ix, cols])
    dirs = np.where(proba >= thr, 1, 0)
    bt = backtest_signals(m5, ix, dirs, max_hold=max_hold, cooldown_bars=cooldown)

    atr_s = __import__("zhulong.training.lgb.backtest", fromlist=["_atr_series"])._atr_series(m5)
    close = m5["close"]
    rs: list[float] = []
    last_idx = -10**9
    for t, d in zip(ix, dirs):
        if d == 0 or t not in m5.index:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice) or (idx - last_idx) < cooldown:
            continue
        if idx + 1 >= len(m5):
            continue
        a = float(atr_s.iloc[idx])
        if a <= 0:
            continue
        entry = float(close.iloc[idx])
        end = min(idx + 1 + max_hold, len(m5))
        rs.append(simulate_trade(1, entry, a, m5["high"].iloc[idx + 1 : end].to_numpy(),
                                 m5["low"].iloc[idx + 1 : end].to_numpy(),
                                 m5["close"].iloc[idx + 1 : end].to_numpy(), max_hold))
        last_idx = idx
    equity = np.cumsum(rs).tolist() if rs else []

    is_v61 = max_hold >= 24 and "profit_24" in str(model_path)
    ver = "v6.1" if is_v61 else ("v6" if args.profit_labels else "v5.1")
    report = {
        "version": ver,
        "split": args.split,
        "threshold": thr,
        "max_hold_bars": max_hold,
        "cooldown_bars": cooldown,
        "backtest": bt,
        "equity_curve_r": equity[-100:] if len(equity) > 100 else equity,
        "equity_final_r": float(equity[-1]) if equity else 0.0,
    }
    out_dir = root / "data" / "training" / "reports" / "lgb" / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "v61" if ver == "v6.1" else ("v6" if ver == "v6" else "v5_1")
    out_path = out_dir / f"backtest_{args.split}_{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== {ver} Backtest {args.split} thr={thr:.2f} hold={max_hold} cooldown={cooldown} ===")
    for k, v in bt.items():
        print(f"  {k}: {v}")
    print(f"  equity_final_r: {report['equity_final_r']}")
    print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
