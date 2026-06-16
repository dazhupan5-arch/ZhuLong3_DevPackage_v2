#!/usr/bin/env python3
"""生成 LSTM 训练用 .npz（60 根 OHLCV + 24 根盈亏标签）。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lstm.dataset import prepare_lstm_splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument(
        "--input",
        default="data/training/lgb/XAUUSD/XAUUSD_M5.csv",
    )
    parser.add_argument(
        "--labels",
        default="data/training/XAUUSD_labeled_profit_24.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training/lstm/XAUUSD",
    )
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--max-hold-bars", type=int, default=24)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = _ROOT
    m5_path = root / args.input
    labels_path = root / args.labels if args.labels else None
    out_dir = root / args.output_dir

    counts = prepare_lstm_splits(
        m5_path,
        labels_path,
        out_dir,
        symbol=args.symbol,
        seq_len=args.seq_len,
        max_hold=args.max_hold_bars,
    )
    print(f"wrote {out_dir}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
