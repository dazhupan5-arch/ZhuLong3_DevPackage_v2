#!/usr/bin/env python3
"""离线准备 RL 特征：结构 + 知识嵌入 → npz。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.knowledge_net import KnowledgeNetInference, build_labels_from_close, train_knowledge_net
from zhulong.agent.structure_analyzer import StructureAnalyzer
from zhulong.strategies.indicators import atr_series


def load_m5_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    df = df.sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/XAUUSD_M5.csv")
    parser.add_argument("--out", default="data/rl_features.npz")
    parser.add_argument("--config", default="config/config_agent.json")
    parser.add_argument("--train-knowledge", action="store_true")
    args = parser.parse_args()

    import json

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    df = load_m5_csv(_ROOT / args.csv)
    analyzer = StructureAnalyzer(cfg.get("structure_analyzer") or {})
    struct = analyzer.compute({"M5": df})
    n = min(len(df), len(struct))
    df = df.iloc[-n:].reset_index(drop=True)
    struct = struct[-n:]
    df["time"] = pd.date_range("2020-01-01", periods=n, freq="5min", tz="UTC")
    atr = atr_series(df.rename(columns=str.lower)).bfill().fillna(df["close"] * 0.001)
    df["atr"] = atr.values

    kn_path = _ROOT / (cfg.get("knowledge_net") or {}).get("model_path", "models/knowledge_net.pth")
    if args.train_knowledge or not kn_path.is_file():
        labels = build_labels_from_close(df["close"].values)
        train_knowledge_net(struct, labels, out_path=kn_path, shuffle_train=True)

    kn = KnowledgeNetInference(kn_path)
    _, emb = kn.predict(struct)

    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        struct=struct,
        emb=emb,
        close=df["close"].values,
        high=df["high"].values,
        low=df["low"].values,
        atr=df["atr"].values,
        time=df["time"].astype(str).values,
    )
    print(f"saved {out} rows={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
