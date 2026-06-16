#!/usr/bin/env python3
"""为已有 horizon_v16 NPZ 补齐 RL 所需的 OHLCV/ATR（无需重算结构特征）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.training_utils import load_m5_csv, load_npz, load_training_config, resolve_symbol_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--npz", default="data/training_horizon_v16.npz")
    args = parser.parse_args()

    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}")
        return 1
    data = load_npz(npz_path)
    if "close" in data and len(data["close"]) == len(data["struct"]):
        print(f"NPZ 已含 OHLCV rows={len(data['struct'])}，跳过")
        return 0

    cfg = load_training_config(_ROOT / "config_training.yaml")
    paths = resolve_symbol_paths(args.symbol, cfg)
    times = pd.to_datetime(data["time"])
    df = load_m5_csv(paths["csv"], str(times.min())[:10], str(times.max())[:10])
    m5 = df.set_index("time")
    common = m5.index.intersection(times)
    if len(common) < len(times) * 0.99:
        print(f"WARN: CSV 对齐率 {len(common)/len(times):.1%}")

    idx_map = {t: i for i, t in enumerate(m5.index)}
    n = len(data["struct"])
    open_ = np.zeros(n, dtype=np.float64)
    high = np.zeros(n, dtype=np.float64)
    low = np.zeros(n, dtype=np.float64)
    close = np.zeros(n, dtype=np.float64)
    volume = np.zeros(n, dtype=np.float64)
    for j, ts in enumerate(times[:n]):
        i = idx_map.get(pd.Timestamp(ts))
        if i is None:
            continue
        open_[j] = m5["open"].iloc[i]
        high[j] = m5["high"].iloc[i]
        low[j] = m5["low"].iloc[i]
        close[j] = m5["close"].iloc[i]
        volume[j] = m5["volume"].iloc[i]

    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = np.zeros(n, dtype=np.float64)
    for i in range(14, n):
        atr[i] = tr[i - 13 : i + 1].mean()
    if n > 14:
        atr[:14] = atr[14]

    out = dict(data)
    out.update(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr": atr,
        }
    )
    np.savez(npz_path, **out)
    print(f"enriched {npz_path} rows={n} (open/high/low/close/volume/atr)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
