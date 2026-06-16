#!/usr/bin/env python3
"""Prepare KN 2.0 training data: V14(68) + struct(30) = 98-dim, aligned OHLCV."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(r"D:\trae_projects\ZhuLong3_v2\data")


def main() -> int:
    # ----- Load OHLCV + V14(68) -----
    xau_path = SRC / "xau_training_data.npz"
    raw = np.load(xau_path, allow_pickle=True)
    n_raw = len(raw["time"])
    print(f"xau_training_data: {n_raw:,} bars")
    print(f"  keys: {list(raw.keys())}")
    print(f"  v14 struct shape: {raw['struct'].shape}")

    out = {}
    for key in ["open", "high", "low", "close", "volume", "atr", "time", "labels"]:
        if key in raw:
            out[key] = raw[key]
    v14 = raw["struct"].astype(np.float32)  # (673186, 68) — these ARE V14 features

    # ----- Load 30d struct features -----
    struct_path = SRC / "training" / "struct" / "XAUUSD" / "struct_features.npz"
    sdata = np.load(struct_path, allow_pickle=True)
    struct30 = sdata["struct"].astype(np.float32)
    n_struct = len(struct30)
    print(f"struct_features: {n_struct:,} rows x {struct30.shape[1]} dims")

    # ----- Align -----
    n = min(n_raw, n_struct)
    if n_raw != n_struct:
        print(f"WARN: length mismatch, align to {n}")
    for k in out:
        out[k] = out[k][:n]
    v14 = v14[:n, :68]
    struct30 = struct30[:n, :30]

    # Combine: [V14(68) | struct(30)] = 98-dim market features
    out["market_feat"] = np.concatenate([v14, struct30], axis=1).astype(np.float32)

    # Fix NaN/Inf
    np.nan_to_num(out["market_feat"], nan=0.0, posinf=10.0, neginf=-10.0, copy=False)
    for k in ["open", "high", "low", "close", "atr"]:
        v = out.get(k)
        if v is not None:
            np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=1e-4, copy=False)

    # Ensure float32 for OHLCV
    for k in ["open", "high", "low", "close", "volume", "atr"]:
        if k in out:
            out[k] = out[k].astype(np.float32)

    out_path = ROOT / "data" / "kn2_training_data.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")
    print(f"  Keys: {list(out.keys())}")
    print(f"  Bars: {n:,}")
    print(f"  market_feat shape: {out['market_feat'].shape}")

    # Quick validation
    print(f"\nValidation:")
    print(f"  open range:  [{out['open'].min():.2f}, {out['open'].max():.2f}]")
    print(f"  close range: [{out['close'].min():.2f}, {out['close'].max():.2f}]")
    print(f"  market_feat mean: {out['market_feat'].mean():.4f}  std:{out['market_feat'].std():.4f}")
    print(f"  market_feat[0][:10]: {out['market_feat'][0,:10]}")
    print(f"  market_feat[1000][:10]: {out['market_feat'][1000,:10]}")
    print(f"  market_feat[1000][68:78]: {out['market_feat'][1000,68:78]}")
    time0 = str(out["time"][0]) if len(out["time"]) > 0 else "N/A"
    timeN = str(out["time"][-1]) if len(out["time"]) > 1 else "N/A"
    print(f"  Time range: {time0} -> {timeN}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
