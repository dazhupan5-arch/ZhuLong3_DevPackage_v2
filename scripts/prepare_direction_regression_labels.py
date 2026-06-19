#!/usr/bin/env python3
"""生成 V17 direction_score 回归标签 NPZ。"""

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

from zhulong.agent.training_utils import load_npz, TRAIN_END_DEFAULT
from zhulong.agent.v17.labels import make_direction_regression_labels, summarize_labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/clean/training_horizon_v16.npz")
    parser.add_argument("--out", default="data/clean/training_v17_direction.npz")
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument("--symbol", default="XAUUSD")
    args = parser.parse_args()

    src = _ROOT / args.npz
    if not src.is_file():
        print(f"缺少 {src}")
        return 1

    raw = load_npz(src)
    close = np.asarray(raw["close"], dtype=np.float64)
    atr = np.asarray(raw["atr"], dtype=np.float64)
    direction_score = make_direction_regression_labels(
        close, atr, horizon_bars=args.horizon_bars
    )
    summary = summarize_labels(direction_score, np.zeros(len(close), dtype=np.int8))

    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: raw[k] for k in raw if k != "labels"}
    payload["direction_score"] = direction_score
    payload["symbol"] = args.symbol
    np.savez_compressed(out_path, **payload)

    report = _ROOT / "data/training/reports/v17/direction_label_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"保存 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
