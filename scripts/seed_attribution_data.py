#!/usr/bin/env python3
"""向 trading.db 灌入归因演示数据（无需真实交易）。"""
from __future__ import annotations

import argparse
import os
import sqlite3
import time


def main() -> int:
    p = argparse.ArgumentParser(description="灌入 trades / position_events 演示数据")
    p.add_argument("--db", default="", help="默认 %APPDATA%\\ZhuLong\\trading.db")
    p.add_argument("--count", type=int, default=5)
    args = p.parse_args()

    db = args.db or os.path.join(os.environ.get("APPDATA", ""), "ZhuLong", "trading.db")
    if not os.path.isfile(db):
        print(f"数据库不存在: {db}")
        return 1

    con = sqlite3.connect(db)
    now = int(time.time())
    for i in range(args.count):
        sid = f"DEMO_{now}_{i}"
        con.execute(
            """INSERT OR IGNORE INTO signals (
                SignalId, Timestamp, Symbol, Direction, EntryPrice, StopLoss, TakeProfit,
                Confidence, ExpectedReturn, MagicNumber, CommentHint, Status, CreatedAt
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, now - 3600 * i, "XAUUSD", "buy" if i % 2 == 0 else "sell",
             2350 + i, 2340, 2370, 0.8, 0.5, 1000 + i, f"ZhuLong_{sid}", "closed", now - 3600 * i),
        )
        pnl = 0.3 if i % 3 else -0.2
        con.execute(
            """INSERT INTO trades (SignalId, OpenTime, OpenPrice, CloseTime, ClosePrice,
                PnlPoints, PnlPercent, IsWin, CloseReason)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (sid, now - 3500 * i, 2350 + i, now - 3400 * i, 2350 + i + pnl,
             pnl * 10, pnl, 1 if pnl > 0 else 0, "demo_seed"),
        )
        con.execute(
            """INSERT INTO position_events (SignalId, EventTime, EventType, Price, OldSl, NewSl)
            VALUES (?,?,?,?,?,?)""",
            (sid, now - 3450 * i, "trailing_sl", 2351 + i, 2340, 2345),
        )
    con.commit()
    con.close()
    print(f"已灌入 {args.count} 笔演示交易 -> {db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
