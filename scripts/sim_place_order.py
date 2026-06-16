#!/usr/bin/env python3
import json
import os
import sqlite3
import sys

try:
    import MetaTrader5 as mt5
except ImportError:
    print(json.dumps({"ok": False, "error": "MetaTrader5 not installed"}))
    sys.exit(1)

db = os.path.join(os.environ.get("APPDATA", ""), "ZhuLong", "trading.db")
row = sqlite3.connect(db).execute(
    "SELECT SignalId, Symbol, Direction, EntryPrice, StopLoss, TakeProfit, MagicNumber, CommentHint "
    "FROM signals ORDER BY CreatedAt DESC LIMIT 1"
).fetchone()
if not row:
    print(json.dumps({"ok": False, "error": "no signal"}))
    sys.exit(0)

signal_id, symbol, direction, entry, sl, tp, magic, comment = row
if not mt5.initialize():
    print(json.dumps({"ok": False, "error": str(mt5.last_error())}))
    sys.exit(0)

broker = symbol
if not mt5.symbol_select(broker, True):
    for alt in (symbol + "m", symbol):
        if mt5.symbol_select(alt, True):
            broker = alt
            break

tick = mt5.symbol_info_tick(broker)
if not tick:
    print(json.dumps({"ok": False, "error": "no tick"}))
    mt5.shutdown()
    sys.exit(0)

existing = [p for p in (mt5.positions_get() or []) if comment in (p.comment or "")]
if existing:
    print(json.dumps({"ok": True, "matched": True, "ticket": existing[0].ticket, "note": "already open"}))
    mt5.shutdown()
    sys.exit(0)

lot = 0.01
order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
price = tick.ask if direction == "buy" else tick.bid
req = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": broker,
    "volume": lot,
    "type": order_type,
    "price": price,
    "sl": sl,
    "tp": tp,
    "deviation": 30,
    "magic": int(magic),
    "comment": comment[:31],
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}
r = mt5.order_send(req)
ok = r is not None and r.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)
print(json.dumps({
    "ok": ok,
    "retcode": r.retcode if r else None,
    "ticket": r.order if r else None,
    "comment": comment,
}))
mt5.shutdown()
