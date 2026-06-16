#!/usr/bin/env python3
"""M5 回放回测：读取 M5 CSV + 可选信号 CSV，使用 lgb/backtest 固定 SL/TP 逻辑。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zhulong.training.lgb.backtest import backtest_signals


def load_m5(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = "time" if "time" in df.columns else "Time"
    df[col] = pd.to_datetime(df[col])
    df = df.rename(columns={col: "time"})
    df = df.set_index("time").sort_index()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df


def load_signals(path: Path, m5_index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray]:
    times: list = []
    dirs: list[int] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = pd.Timestamp(row["time"])
            d = row.get("direction", row.get("dir", "0")).lower()
            if d in ("buy", "long", "1"):
                di = 1
            elif d in ("sell", "short", "-1"):
                di = -1
            else:
                di = int(float(d))
            if t in m5_index:
                times.append(t)
                dirs.append(di)
    return pd.DatetimeIndex(times), np.array(dirs, dtype=int)


def rule_signals(m5: pd.DataFrame, every: int = 60) -> tuple[pd.DatetimeIndex, np.ndarray]:
    times, dirs = [], []
    for i in range(60, len(m5), every):
        times.append(m5.index[i])
        dirs.append(1 if (i // every) % 2 == 0 else -1)
    return pd.DatetimeIndex(times), np.array(dirs, dtype=int)


def main() -> int:
    p = argparse.ArgumentParser(description="ZhuLong M5 回测")
    p.add_argument("csv", type=Path, help="M5 CSV")
    p.add_argument("--signals", type=Path, help="可选 signals.csv: time,direction")
    p.add_argument("--every", type=int, default=60, help="无 signals 时每 N 根交替多空")
    args = p.parse_args()

    m5 = load_m5(args.csv)
    if args.signals and args.signals.is_file():
        times, directions = load_signals(args.signals, m5.index)
    else:
        times, directions = rule_signals(m5, args.every)

    stats = backtest_signals(m5, times, directions)
    print(
        f"bars={len(m5)} trades={stats['n_trades']} "
        f"win_rate={stats['win_rate']:.1%} expectancy={stats['expectancy']:.3f}R "
        f"max_dd={stats['max_drawdown']:.1%} total_R={stats['total_pnl_r']:.2f}"
    )
    return 0 if stats["n_trades"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
