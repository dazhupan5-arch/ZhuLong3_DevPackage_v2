#!/usr/bin/env python3
"""Walk-forward 窗口划分骨架（不训练，仅切分 M5 CSV）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description="Walk-forward 窗口划分")
    p.add_argument("csv", type=Path, help="M5 CSV (time,open,high,low,close)")
    p.add_argument("--train-months", type=int, default=3)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=Path("data/training/walk_forward"))
    args = p.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["time"])
    df = df.sort_values("time").set_index("time")
    if len(df) < 100:
        print("数据过短")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_days = args.train_months * 30
    test_days = args.test_months * 30
    start = df.index.min()
    end = df.index.max()
    cursor = start
    fold = 0

    while cursor + pd.Timedelta(days=train_days + test_days) <= end:
        train_end = cursor + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        train = df.loc[cursor:train_end]
        test = df.loc[train_end:test_end]
        if len(train) > 50 and len(test) > 10:
            train.to_csv(args.out_dir / f"fold{fold}_train.csv")
            test.to_csv(args.out_dir / f"fold{fold}_test.csv")
            print(f"fold{fold}: train={len(train)} test={len(test)} "
                  f"{train.index.min()} .. {test.index.max()}")
            fold += 1
        cursor += pd.Timedelta(days=test_days)

    print(f"共 {fold} 折 -> {args.out_dir}")
    return 0 if fold else 1


if __name__ == "__main__":
    raise SystemExit(main())
