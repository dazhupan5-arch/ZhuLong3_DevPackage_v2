#!/usr/bin/env python3
"""命名管道 smoke：M1 bar 和/或 draw_signal（需 ZhuLong 运行中）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

try:
    import win32file
    import pywintypes
except ImportError:
    print("需要 pywin32: pip install pywin32")
    sys.exit(1)

PIPE_DATA = r"\\.\pipe\ZhuLong_Data"
PIPE_DRAW = r"\\.\pipe\ZhuLong_Drawing"


def write_pipe(pipe: str, payload: dict) -> None:
    handle = win32file.CreateFile(
        pipe,
        win32file.GENERIC_WRITE,
        0,
        None,
        win32file.OPEN_EXISTING,
        0,
        None,
    )
    line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    win32file.WriteFile(handle, line)
    win32file.CloseHandle(handle)


def main() -> int:
    ap = argparse.ArgumentParser(description="ZhuLong 管道 smoke")
    ap.add_argument("--draw", action="store_true", help="发送 draw_signal（需 MT5 已连 Drawing 管道）")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--direction", default="buy")
    args = ap.parse_args()

    if args.draw:
        sig_id = f"SMOKE_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        payload = {
            "action": "draw_signal",
            "signal_id": sig_id,
            "symbol": args.symbol,
            "direction": args.direction,
            "entry": 2350.0,
            "sl": 2340.0,
            "tp": 2370.0,
            "confidence": 0.99,
            "expiry_minutes": 240,
        }
        print(f"连接 {PIPE_DRAW} ...")
        try:
            write_pipe(PIPE_DRAW, payload)
        except pywintypes.error as e:
            print(f"draw 失败: {e}. 需 ZhuLong 运行且 MT5 指标已连接 Drawing 管道")
            return 1
        print("已发送 draw_signal:", payload)
        return 0

    bar = {
        "type": "bar",
        "symbol": args.symbol,
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "open": 2350.0,
        "high": 2351.0,
        "low": 2349.5,
        "close": 2350.5,
        "volume": 120,
    }
    print(f"连接 {PIPE_DATA} ...")
    try:
        write_pipe(PIPE_DATA, bar)
    except pywintypes.error as e:
        print(f"连接失败: {e}. 请先启动 ZhuLong.exe 并点击「开始运行」")
        return 1
    print("已发送 M1 bar:", bar)
    time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
