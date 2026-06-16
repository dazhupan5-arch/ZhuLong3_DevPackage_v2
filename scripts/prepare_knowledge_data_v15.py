#!/usr/bin/env python3
"""KnowledgeNet 训练数据：V15 76 维特征 + Regime Triple Barrier 标签。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.training_utils import load_m5_csv, load_training_config, resolve_symbol_paths
from zhulong.training.lgb.features_v15 import FEATURE_COLUMNS_V15, compute_features_v15
from zhulong.training.lgb.labels_v15 import generate_labels_v15


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default="config_training.yaml")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    sym = args.symbol.upper()

    cfg = load_training_config(_ROOT / args.config)
    paths = resolve_symbol_paths(sym, cfg)
    data_cfg = cfg.get("data") or {}

    csv_path = paths["csv"]
    if not csv_path.is_file():
        print(f"CSV 不存在: {csv_path}")
        return 1

    start = data_cfg.get("default_start", "2016-01-01")
    end = "2025-12-31"
    df = load_m5_csv(csv_path, start, end)
    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.iloc[-args.max_rows :].reset_index(drop=True)
    print(f"Loading {csv_path} rows={len(df)}")

    m5 = df.set_index("time")
    feat_cache = _ROOT / "data" / "training" / "v15" / sym / "features_kn.parquet"
    if feat_cache.is_file():
        feats = pd.read_parquet(feat_cache)
        common = m5.index.intersection(feats.index)
        m5 = m5.loc[common]
        feats = feats.loc[common]
        print(f"V15 KN cache: {feats.shape}")
    else:
        print("计算 V15 76 维特征...", flush=True)
        feats = compute_features_v15(m5)
        feat_cache.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(feat_cache)
        print(f"V15 features cached: {feat_cache}", flush=True)

    labels, _ = generate_labels_v15(m5.loc[feats.index])
    counts = {int(v): int((labels == v).sum()) for v in (-1, 0, 1)}
    print(f"V15 labels: short={counts[-1]} flat={counts[0]} long={counts[1]}")

    close = m5["close"].values.astype(np.float64)
    high = m5["high"].values
    low = m5["low"].values
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = np.zeros(len(close), dtype=np.float64)
    for i in range(14, len(close)):
        atr[i] = tr[i - 13 : i + 1].mean()
    atr[:14] = atr[14]

    cols = [c for c in FEATURE_COLUMNS_V15 if c in feats.columns]
    struct = feats[cols].values.astype(np.float32)
    out = Path(args.out) if args.out else _ROOT / "data" / "training_data_v15.npz"
    n = min(len(struct), len(labels))
    np.savez(
        out,
        symbol=np.array([sym]),
        time=m5.index[:n].astype(str).values,
        open=m5["open"].values[:n],
        high=m5["high"].values[:n],
        low=m5["low"].values[:n],
        close=m5["close"].values[:n],
        volume=m5["volume"].values[:n],
        atr=atr[:n],
        struct=struct[:n],
        labels=labels[:n],
    )
    print(f"saved {out} rows={n} struct_dim={struct.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
