#!/usr/bin/env python3
"""向 SQLite 注入测试信号（无需正式模型，用于 L2-4/L2-5 匹配托管验收）。"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timezone


def dotnet_string_hash(s: str) -> int:
    """与 C# string.GetHashCode() 兼容（.NET Core 确定性哈希）。"""
    h = 5381
    for ch in s:
        h = ((h << 5) + h) ^ ord(ch)
    return h & 0xFFFFFFFF


def main() -> int:
    p = argparse.ArgumentParser(description="注入测试信号到 trading.db")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--direction", choices=["buy", "sell"], default="buy")
    p.add_argument("--entry", type=float, default=0, help="0=自动用当前价占位")
    p.add_argument("--atr-pct", type=float, default=0.5, help="用于 SL/TP 估算")
    p.add_argument("--prefix", default="ZhuLong")
    p.add_argument("--db", default="", help="默认 %APPDATA%\\ZhuLong\\trading.db")
    args = p.parse_args()

    db = args.db or os.path.join(os.environ.get("APPDATA", ""), "ZhuLong", "trading.db")
    if not os.path.isfile(db):
        print(json.dumps({"ok": False, "error": f"数据库不存在: {db}"}))
        return 1

    now = datetime.now(timezone.utc)
    signal_id = f"{now.strftime('%Y%m%d_%H%M')}_{args.symbol}_{args.direction}_TEST"
    magic = dotnet_string_hash(signal_id) & 0xFFFF or 1
    comment = f"{args.prefix}_{signal_id}"

    entry = args.entry if args.entry > 0 else (2350.0 if args.symbol == "XAUUSD" else 75.0)
    atr_abs = entry * args.atr_pct / 100.0
    if args.direction == "buy":
        sl, tp = entry - atr_abs * 1.2, entry + atr_abs * 2.0
    else:
        sl, tp = entry + atr_abs * 1.2, entry - atr_abs * 2.0

    ts = int(time.time())
    row = (
        signal_id,
        ts,
        args.symbol,
        args.direction,
        entry,
        sl,
        tp,
        0.85,
        1.0,
        magic,
        comment,
        "",
        "pending",
        json.dumps({"source": "inject_test_signal"}),
        ts,
    )

    con = sqlite3.connect(db)
    con.execute(
        """
        INSERT OR REPLACE INTO signals (
            signal_id, timestamp, symbol, direction, entry_price, stop_loss, take_profit,
            confidence, expected_return, magic_number, comment_hint, strategy, status, params_snapshot, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        row,
    )
    con.commit()
    con.close()

    print(json.dumps({
        "ok": True,
        "signal_id": signal_id,
        "comment": comment,
        "magic": magic,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "next": "在 MT5 手动下单，Comment 设为 comment 值；重启或刷新烛龙信号列表",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
