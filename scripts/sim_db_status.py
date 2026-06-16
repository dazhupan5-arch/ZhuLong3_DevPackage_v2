#!/usr/bin/env python3
import json
import os
import sqlite3
import sys

db = os.path.join(os.environ.get("APPDATA", ""), "ZhuLong", "trading.db")
if not os.path.isfile(db):
    print(json.dumps({"ok": False, "error": "no db"}))
    sys.exit(1)

c = sqlite3.connect(db)
cur = c.cursor()
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
macro_n = cur.execute("SELECT COUNT(*) FROM macro_events").fetchone()[0] if "macro_events" in tables else -1
out = {
    "ok": True,
    "signals": cur.execute("SELECT COUNT(*) FROM signals").fetchone()[0],
    "trades": cur.execute("SELECT COUNT(*) FROM trades").fetchone()[0],
    "events": cur.execute("SELECT COUNT(*) FROM position_events").fetchone()[0],
    "macro_events": macro_n,
    "latest": cur.execute(
        "SELECT SignalId, Symbol, Direction, Confidence, Status, CommentHint "
        "FROM signals ORDER BY CreatedAt DESC LIMIT 1"
    ).fetchone(),
}
print(json.dumps(out))
