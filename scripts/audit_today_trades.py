import sqlite3
from datetime import datetime, timezone, timedelta

DB = r"C:\Users\Administrator\AppData\Roaming\ZhuLong\trading.db"
bj = timezone(timedelta(hours=8))
cn = sqlite3.connect(DB)
cn.row_factory = sqlite3.Row
cur = cn.cursor()
cur.execute(
    """
    SELECT s.created_at, s.signal_id, s.status, s.entry_price, s.stop_loss, s.take_profit,
           s.confidence, t.open_price, t.close_price, t.pnl_percent, t.close_reason
    FROM signals s
    LEFT JOIN trades t ON t.signal_id = s.signal_id
    WHERE s.symbol = 'XAUUSD' AND s.direction = 'buy'
      AND s.created_at >= strftime('%s', '2026-06-12 00:00:00', '-8 hours')
    ORDER BY s.created_at
    """
)
print("=== Today XAUUSD buy signals ===")
for r in cur.fetchall():
    ts = datetime.fromtimestamp(r["created_at"], tz=timezone.utc).astimezone(bj)
    entry = r["open_price"] or r["entry_price"]
    sl = r["stop_loss"]
    close = r["close_price"]
    sl_dist = (entry - sl) / entry * 100 if entry and sl else 0
    hit = ""
    if close and sl:
        hit = "REAL_SL" if close <= sl + 0.05 else f"close>{sl:.2f}"
    print(
        f"{ts:%H:%M:%S} | {r['status']:14} | conf={r['confidence']:.3f} | "
        f"entry={entry:.2f} sl={sl:.2f} ({sl_dist:.2f}%) close={close or 0:.2f} | "
        f"pnl={r['pnl_percent'] if r['pnl_percent'] is not None else 'NA'} | {r['close_reason']} | {hit} | {r['signal_id']}"
    )
