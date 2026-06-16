#!/usr/bin/env python3
"""从 MT5 拉取 M1 历史并保存为 parquet（IMPLEMENTATION_PLAN Phase 2）。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fetch_m1(mt5, symbol: str, months: int, max_bars: int) -> list:
    tf = mt5.TIMEFRAME_M1
    if months > 0:
        utc_to = datetime.now(timezone.utc)
        utc_from = utc_to - timedelta(days=months * 31)
        rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
    else:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, max_bars)
    if rates is None:
        return []
    return list(rates)


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 M1 K 线到 parquet")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--months", type=int, default=6, help="拉取最近 N 个月（优先于 --bars）")
    parser.add_argument("--bars", type=int, default=200_000, help="months=0 时按根数拉取")
    parser.add_argument(
        "--out",
        default="",
        help="输出 parquet 路径，默认 data/history/{symbol}_M1.parquet",
    )
    args = parser.parse_args()

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("需要 MetaTrader5: py -3 -m pip install MetaTrader5")
        return 1

    try:
        import pandas as pd
    except ImportError:
        print("需要 pandas: py -3 -m pip install pandas pyarrow")
        return 1

    if not mt5.initialize():
        print("MT5 initialize 失败:", mt5.last_error())
        return 1

    sym = args.symbol
    if not mt5.symbol_select(sym, True):
        for alt in (sym + "m", sym + ".", sym):
            if mt5.symbol_select(alt, True):
                sym = alt
                break

    rates = fetch_m1(mt5, sym, args.months, args.bars)
    mt5.shutdown()
    if not rates:
        print("未获取到 K 线")
        return 1

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    out = Path(args.out) if args.out else ROOT / "data" / "history" / f"{args.symbol}_M1.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"OK {sym} rows={len(df)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
