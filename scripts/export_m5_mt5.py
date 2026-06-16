#!/usr/bin/env python3
"""从已登录的 MT5 终端导出尽可能多的 M5/M1 K 线。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TF_MAP = {"M1": 1, "M5": 5, "M15": 15, "H1": 60}


def _tf_const(mt5, name: str):
    return {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
    }[name]


def fetch_max_bars(mt5, symbol: str, tf_name: str, request: int) -> list:
    """向 MT5 请求尽量多的 K 线（服务器上限由券商决定）。"""
    tf = _tf_const(mt5, tf_name)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, request)
    if rates is None or len(rates) == 0:
        return []
    return list(rates)


def write_csv(rates, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["time,open,high,low,close,volume"]
    for r in rates:
        ts = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        vol = int(r["real_volume"]) if int(r["real_volume"]) > 0 else int(r["tick_volume"])
        lines.append(f"{ts},{r['open']},{r['high']},{r['low']},{r['close']},{vol}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rates)


def main() -> int:
    parser = argparse.ArgumentParser(description="MT5 历史 K 线导出（尽量拉满券商可用量）")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5", choices=["M1", "M5", "M15", "H1"])
    parser.add_argument("--bars", type=int, default=0, help="0 = 自动拉取服务器最大可用量")
    parser.add_argument("--out", default="", help="输出 CSV 路径")
    args = parser.parse_args()

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("需要 MetaTrader5: py -3 -m pip install MetaTrader5")
        return 1

    if not mt5.initialize():
        print("MT5 initialize 失败:", mt5.last_error())
        print("请确保 MT5 已登录并保持运行")
        return 1

    try:
        if not mt5.symbol_select(args.symbol, True):
            print("symbol_select 失败:", args.symbol, mt5.last_error())
            return 1

        request = args.bars if args.bars > 0 else 5_000_000
        rates = fetch_max_bars(mt5, args.symbol, args.timeframe, request)
        if not rates:
            print("无数据:", mt5.last_error())
            return 1

        out = (
            Path(args.out)
            if args.out
            else ROOT / "data" / "training" / f"{args.symbol}_{args.timeframe}.csv"
        )
        n = write_csv(rates, out)
        t0 = datetime.fromtimestamp(int(rates[0]["time"]), tz=timezone.utc)
        t1 = datetime.fromtimestamp(int(rates[-1]["time"]), tz=timezone.utc)
        days = (t1 - t0).days
        print(f"OK: {out}")
        print(f"rows={n} tf={args.timeframe} range={t0.date()} .. {t1.date()} (~{days} days)")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
