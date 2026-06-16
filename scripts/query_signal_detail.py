import sqlite3
from datetime import datetime, timezone, timedelta

CN = timezone(timedelta(hours=8))
DB = r"C:\Users\xiaomi\AppData\Roaming\ZhuLong\trading.db"
cn = sqlite3.connect(DB)
cn.row_factory = sqlite3.Row

print("=== TABLES ===")
for t in cn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
    print(t[0])

for sid in ["agent_20260615_0336_XAUUSD_buy_46cf1f", "agent_20260614_2332_XAUUSD_buy_e46471"]:
    print(f"\n=== SIGNAL {sid} ===")
    r = cn.execute("SELECT * FROM signals WHERE signal_id=?", (sid,)).fetchone()
    if r:
        d = dict(r)
        for k in ["status", "entry_price", "stop_loss", "take_profit", "comment_hint", "params_snapshot"]:
            print(f"  {k}: {d.get(k)}")
        for k in ["created_at", "timestamp"]:
            ts = d.get(k)
            if ts:
                print(f"  {k}: {datetime.fromtimestamp(ts, CN)}")

    t = cn.execute("SELECT * FROM trades WHERE signal_id=?", (sid,)).fetchone()
    print(f"  trade row: {dict(t) if t else 'NONE'}")

    try:
        ev = cn.execute("SELECT * FROM position_events WHERE signal_id=? ORDER BY id", (sid,)).fetchall()
        print(f"  position_events: {len(ev)}")
        for e in ev[-5:]:
            print(f"    {dict(e)}")
    except sqlite3.OperationalError as ex:
        print(f"  position_events: {ex}")

cn.close()
