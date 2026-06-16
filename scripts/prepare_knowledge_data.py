#!/usr/bin/env python3
"""KnowledgeNet 训练数据：V14 68 维特征 + 方向标签。"""

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
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features


def _default_out(symbol: str) -> Path:
    sym = symbol.upper()
    if sym == "USOIL":
        return _ROOT / "data" / "oil_training_data.npz"
    return _ROOT / "data" / "training_data.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD", "USOIL"])
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default="config_training.yaml")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    sym = args.symbol.upper()

    cfg = load_training_config(_ROOT / args.config)
    paths = resolve_symbol_paths(sym, cfg)
    data_cfg = cfg.get("data") or {}
    kn_cfg = cfg.get("knowledge_net") or {}
    oil_cfg = kn_cfg.get("oil") or {}

    horizon = args.horizon or int(
        (data_cfg.get("label_horizon", 12) if sym == "XAUUSD" else oil_cfg.get("label_horizon", data_cfg.get("label_horizon", 12)))
    )
    gain = args.gain or float(oil_cfg.get("label_gain", data_cfg.get("label_threshold", 0.002)))
    if sym == "XAUUSD" and args.horizon == 0 and args.gain == 0.0:
        gain = float(data_cfg.get("label_threshold", 0.002))

    csv_path = paths["csv"]
    if not csv_path.is_file():
        print(f"CSV 不存在: {csv_path}")
        return 1

    start = data_cfg.get("default_start", "2016-01-01")
    end = data_cfg.get("default_end", "2025-12-31")
    df = load_m5_csv(csv_path, start, end)
    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.iloc[-args.max_rows :].reset_index(drop=True)
    print(f"Loading {csv_path} rows={len(df)}")

    m5 = df.set_index("time")
    feat_cache = _ROOT / "data" / "training" / "v14" / sym / "features.parquet"
    struct: np.ndarray | None = None
    if feat_cache.is_file():
        feats = pd.read_parquet(feat_cache)
        if feats.index.tz is not None and m5.index.tz is None:
            m5.index = m5.index.tz_localize("UTC")
        elif feats.index.tz is None and m5.index.tz is not None:
            feats.index = feats.index.tz_localize("UTC")
        common = m5.index.intersection(feats.index)
        if len(common) >= 500:
            m5 = m5.loc[common]
            feats = feats.loc[common]
            cols = [c for c in FEATURE_COLUMNS_LGB_V13 if c in feats.columns]
            struct = feats[cols].values.astype(np.float32)
            print(f"V14 cache: {struct.shape}")
        else:
            print(f"WARN: V14 cache 索引不匹配 (common={len(common)}), 重新计算特征...")

    if struct is None:
        print("计算 V14 68 维特征...", flush=True)
        feats = compute_features(m5, include_mtf=True, include_reversal=True)
        cols = list(FEATURE_COLUMNS_LGB_V13)
        struct = feats[cols].values.astype(np.float32)
        print(f"V14 computed: {struct.shape}", flush=True)
        feat_cache.parent.mkdir(parents=True, exist_ok=True)
        feats[cols].to_parquet(feat_cache)
        print(f"V14 cache saved: {feat_cache}", flush=True)

    if struct is None or len(m5) < 500:
        print(f"ERROR: 有效样本不足 (rows={len(m5)})")
        return 1

    close = m5["close"].values.astype(np.float64)
    labels = np.zeros(len(close), dtype=np.int8)
    for i in range(len(close) - horizon):
        ret = (close[i + horizon] - close[i]) / max(close[i], 1e-9)
        if ret > gain:
            labels[i] = 1
        elif ret < -gain:
            labels[i] = -1
    counts = {v: int((labels == v).sum()) for v in (-1, 0, 1)}
    print(f"labels horizon={horizon} gain={gain}: short={counts[-1]} flat={counts[0]} long={counts[1]}")

    high = m5["high"].values
    low = m5["low"].values
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = np.zeros(len(close), dtype=np.float64)
    for i in range(14, len(close)):
        atr[i] = tr[i - 13 : i + 1].mean()
    atr[:14] = atr[14]

    out = Path(args.out) if args.out else _default_out(sym)
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
        struct=struct[:n].astype(np.float32),
        labels=labels[:n],
    )
    print(f"saved {out} rows={n} struct_dim={struct.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
