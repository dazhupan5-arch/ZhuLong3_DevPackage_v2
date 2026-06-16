#!/usr/bin/env python3
"""探测当前行情下推理结果与信号过滤原因（不依赖 GUI）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ZhuLong.PythonEngine"))

import MetaTrader5 as mt5  # noqa: E402


def main() -> int:
    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error())
        return 1

    symbol = "XAUUSD"
    if not mt5.symbol_select(symbol, True):
        for alt in ("XAUUSDm", "GOLD"):
            if mt5.symbol_select(alt, True):
                symbol = alt
                break

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 1, 80)
    if rates is None or len(rates) < 60:
        print(f"M5 不足: {0 if rates is None else len(rates)}")
        mt5.shutdown()
        return 1

    close = float(rates[-1]["close"])
    print(f"symbol={symbol} close={close:.2f} m5_bars={len(rates)}")

    try:
        from inference import predict_symbol  # ZhuLong.PythonEngine

        seq = [[0.0] * 30 for _ in range(60)]
        hourly = [0.0] * 10
        macro = [0.0] * 8
        result = predict_symbol(symbol, seq, hourly, macro)
        print("inference:", json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print("inference error:", e)
        print("(演示模型需由 ZhuLong 进程内 Python.NET 加载，此处仅验证 MT5 数据)")

    mt5.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
