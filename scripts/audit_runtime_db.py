#!/usr/bin/env python3
"""Audit ZhuLong trading.db for signal/trade state."""
import sqlite3
from pathlib import Path

DB = Path.home() / "AppData" / "Roaming" / "ZhuLong" / "trading.db"

def main() -> None:
    if not DB.exists():
        print("MISSING_DB", DB)
        return
    print("DB", DB, "size", DB.stat().st_size, "mtime", DB.stat().st_mtime)
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1")
    tables = [r[0] for r in cur.fetchall()]
    print("TABLES", tables)

    if "signals" in tables:
        cur.execute("PRAGMA table_info(signals)")
        sig_cols = {r[1] for r in cur.fetchall()}
        order_col = "created_at" if "created_at" in sig_cols else "signal_id"
        cur.execute(
            f"SELECT signal_id, symbol, direction, status, {order_col} "
            f"FROM signals ORDER BY {order_col} DESC LIMIT 20"
        )
        print("\n=== RECENT SIGNALS ===")
        for r in cur.fetchall():
            print(r)

        cur.execute(
            "SELECT signal_id, symbol, direction, status FROM signals "
            "WHERE status IN ('active','awaiting_fill','pending')"
        )
        rows = cur.fetchall()
        print("\n=== ACTIVE IN DB ===", "count=", len(rows))
        for r in rows:
            print(r)

        cur.execute(
            "SELECT status, COUNT(*) FROM signals GROUP BY status ORDER BY COUNT(*) DESC"
        )
        print("\n=== STATUS COUNTS ===")
        for r in cur.fetchall():
            print(r)

    if "trades" in tables:
        cur.execute(
            "SELECT signal_id, open_time, close_time, pnl_percent, close_reason "
            "FROM trades ORDER BY close_time DESC LIMIT 15"
        )
        print("\n=== RECENT TRADES ===")
        for r in cur.fetchall():
            print(r)
    else:
        print("\nNO trades TABLE")

    if "position_events" in tables:
        cur.execute("PRAGMA table_info(position_events)")
        cols = {r[1] for r in cur.fetchall()}
        if "created_at" in cols:
            cur.execute(
                "SELECT signal_id, event_type, created_at FROM position_events "
                "ORDER BY created_at DESC LIMIT 10"
            )
        else:
            cur.execute(
                "SELECT signal_id, event_type FROM position_events LIMIT 10"
            )
        print("\n=== POSITION EVENTS ===")
        for r in cur.fetchall():
            print(r)

    con.close()

if __name__ == "__main__":
    main()
