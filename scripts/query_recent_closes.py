"""Query recent closed signals from trading.db"""
import sqlite3
from datetime import datetime, timezone, timedelta

CN = timezone(timedelta(hours=8))
DB = r"C:\Users\xiaomi\AppData\Roaming\ZhuLong\trading.db"

def fmt(ts):
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(ts, CN).strftime("%Y-%m-%d %H:%M:%S")

cn = sqlite3.connect(DB)
cn.row_factory = sqlite3.Row

print("=== RECENT CLOSED SIGNALS ===")
rows = cn.execute("""
SELECT s.signal_id, s.symbol, s.direction, s.status, s.entry_price, s.stop_loss, s.take_profit,
       s.timestamp, s.created_at, t.open_time, t.close_time, t.close_price, t.pnl_percent, t.close_reason
FROM signals s
LEFT JOIN trades t ON t.signal_id = s.signal_id
WHERE s.status NOT IN ('pending', 'active', 'awaiting_fill')
ORDER BY COALESCE(t.close_time, s.created_at) DESC
LIMIT 20
""").fetchall()

for r in rows:
    pnl = r["pnl_percent"]
    ui_display = f"{pnl * 100:+.2f}%" if pnl is not None else "—"
    print("---")
    print(f"id: {r['signal_id']}")
    print(f"  {r['direction']} | status={r['status']} | reason={r['close_reason']}")
    print(f"  entry={r['entry_price']} sl={r['stop_loss']} tp={r['take_profit']} close={r['close_price']}")
    print(f"  pnl_percent(raw)={pnl} | UI_bug_display={ui_display}")
    print(f"  signal.created_at = {fmt(r['created_at'])}  (UI shows this)")
    print(f"  trade.open_time   = {fmt(r['open_time'])}")
    print(f"  trade.close_time  = {fmt(r['close_time'])}  (actual exit)")

cn.close()
