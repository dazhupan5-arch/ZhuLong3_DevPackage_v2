#!/usr/bin/env python3
import json
import sys

try:
    import MetaTrader5 as mt5
except ImportError:
    print(json.dumps({"ok": False, "error": "MetaTrader5 not installed"}))
    sys.exit(0)

if not mt5.initialize():
    print(json.dumps({"ok": False, "error": str(mt5.last_error())}))
    sys.exit(0)

acc = mt5.account_info()
sym = "XAUUSD"
if not mt5.symbol_select(sym, True):
    for alt in ("XAUUSDm", "GOLD", "XAUUSD."):
        if mt5.symbol_select(alt, True):
            sym = alt
            break
tick = mt5.symbol_info_tick(sym)
print(json.dumps({
    "ok": True,
    "login": acc.login if acc else None,
    "server": acc.server if acc else None,
    "trade_mode": acc.trade_mode if acc else None,
    "balance": acc.balance if acc else None,
    "symbol": sym,
    "bid": tick.bid if tick else None,
    "ask": tick.ask if tick else None,
    "positions": len(mt5.positions_get() or []),
}))
mt5.shutdown()
