#!/usr/bin/env python3
"""
生成优质开仓点标签（先触 TP 且期间最大不利波动 < max_dd_atr × ATR）。

标签：1=优质做多，2=优质做空，0=其他（训练用 0/1/2 三分类）

用法:
  py -3 scripts/generate_v13_quality_labels.py --symbol XAUUSD
"""

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

from zhulong.training.lgb.backtest import _atr_series
from zhulong.training.lgb.data_io import load_vendor_csv


def _long_quality(
    i: int,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    *,
    sl_atr: float,
    tp_atr: float,
    max_bars: int,
    max_dd_atr: float,
) -> bool:
    entry, a = close[i], atr[i]
    if a <= 0 or np.isnan(a):
        return False
    sl, tp = entry - sl_atr * a, entry + tp_atr * a
    end = min(i + max_bars + 1, len(close))
    max_adverse = 0.0
    for j in range(i + 1, end):
        if low[j] <= sl:
            return False
        max_adverse = max(max_adverse, entry - low[j])
        if high[j] >= tp:
            return (max_adverse / a) < max_dd_atr
    return False


def _short_quality(
    i: int,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    *,
    sl_atr: float,
    tp_atr: float,
    max_bars: int,
    max_dd_atr: float,
) -> bool:
    entry, a = close[i], atr[i]
    if a <= 0 or np.isnan(a):
        return False
    sl, tp = entry + sl_atr * a, entry - tp_atr * a
    end = min(i + max_bars + 1, len(close))
    max_adverse = 0.0
    for j in range(i + 1, end):
        if high[j] >= sl:
            return False
        max_adverse = max(max_adverse, high[j] - entry)
        if low[j] <= tp:
            return (max_adverse / a) < max_dd_atr
    return False


def generate_quality_labels(
    m5: pd.DataFrame,
    *,
    sl_atr: float = 1.0,
    tp_atr: float = 2.0,
    max_bars: int = 12,
    max_dd_atr: float = 0.5,
) -> np.ndarray:
    """返回 int8 标签数组，与 m5 行对齐：0=flat, 1=long, 2=short。"""
    atr = _atr_series(m5).to_numpy()
    high = m5["high"].to_numpy(dtype=np.float64)
    low = m5["low"].to_numpy(dtype=np.float64)
    close = m5["close"].to_numpy(dtype=np.float64)
    n = len(m5)
    labels = np.zeros(n, dtype=np.int8)
    kw = dict(sl_atr=sl_atr, tp_atr=tp_atr, max_bars=max_bars, max_dd_atr=max_dd_atr)

    for i in range(n - max_bars - 1):
        if i > 0 and i % 100_000 == 0:
            print(f"  labels {i}/{n} long={int((labels == 1).sum())} short={int((labels == 2).sum())}")
        long_ok = _long_quality(i, high, low, close, atr, **kw)
        short_ok = _short_quality(i, high, low, close, atr, **kw)
        if long_ok and not short_ok:
            labels[i] = 1
        elif short_ok and not long_ok:
            labels[i] = 2
        elif long_ok and short_ok:
            labels[i] = 1  # 极少同时满足，优先多
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--tp-atr", type=float, default=2.0)
    parser.add_argument("--max-bars", type=int, default=12)
    parser.add_argument("--max-dd-atr", type=float, default=0.5, help="最大不利波动上限（ATR倍数）")
    parser.add_argument("--output", default="data/labels/XAUUSD_quality_labels.npy")
    args = parser.parse_args()

    m5_path = _ROOT / "data" / "training" / "lgb" / args.symbol / f"{args.symbol}_M5.csv"
    m5 = load_vendor_csv(m5_path)
    print(f"Loaded {len(m5)} bars, generating quality labels...")
    labels = generate_quality_labels(
        m5,
        sl_atr=args.sl_atr,
        tp_atr=args.tp_atr,
        max_bars=args.max_bars,
        max_dd_atr=args.max_dd_atr,
    )

    out_npy = _ROOT / args.output
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, labels)

    out_parquet = out_npy.with_suffix(".parquet")
    pd.DataFrame({"label": labels.astype(int)}, index=m5.index).to_parquet(out_parquet)
    meta = {
        "symbol": args.symbol,
        "n_bars": len(labels),
        "sl_atr": args.sl_atr,
        "tp_atr": args.tp_atr,
        "max_bars": args.max_bars,
        "max_dd_atr": args.max_dd_atr,
        "n_long": int((labels == 1).sum()),
        "n_short": int((labels == 2).sum()),
        "n_flat": int((labels == 0).sum()),
        "index_start": str(m5.index[0]),
        "index_end": str(m5.index[-1]),
    }
    (out_npy.parent / f"{args.symbol}_quality_labels_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"Saved -> {out_npy}")
    print(f"Saved -> {out_parquet}")
    print(
        f"Distribution: long(1)={meta['n_long']} ({100*meta['n_long']/len(labels):.2f}%) "
        f"short(2)={meta['n_short']} ({100*meta['n_short']/len(labels):.2f}%) "
        f"flat(0)={meta['n_flat']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
