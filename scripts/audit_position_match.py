#!/usr/bin/env python3
"""实机审计：pending 信号 vs MT5 持仓 vs 烛龙托管匹配条件。"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


def main() -> int:
    db = os.path.join(os.environ.get("APPDATA", ""), "ZhuLong", "trading.db")
    out: dict = {"ok": True, "checks": []}

    if not os.path.isfile(db):
        out["ok"] = False
        out["error"] = f"no db: {db}"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    cols = [r[1] for r in con.execute("pragma table_info(signals)")]
    out["signal_columns"] = cols

    pending = con.execute(
        "SELECT * FROM signals WHERE status='pending' ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    out["pending_count"] = len(pending)
    out["pending"] = [dict(r) for r in pending]

    if mt5 is None:
        out["checks"].append({"name": "mt5_module", "pass": False, "detail": "MetaTrader5 not installed"})
    else:
        if not mt5.initialize():
            out["checks"].append({"name": "mt5_init", "pass": False, "detail": str(mt5.last_error())})
        else:
            positions = mt5.positions_get() or []
            out["mt5_positions"] = [
                {
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "type": "buy" if p.type == 0 else "sell",
                    "volume": p.volume,
                    "price_open": p.price_open,
                    "comment": p.comment,
                    "time": p.time,
                }
                for p in positions
            ]
            mt5.shutdown()

            prefix = "ZhuLong"
            tol_pts = 5
            win_sec = 60
            now = int(time.time())

            for sig_row in pending:
                sig = dict(sig_row)
                sid = sig.get("signal_id") or sig.get("SignalId") or ""
                sym = sig.get("symbol") or sig.get("Symbol") or ""
                direction = sig.get("direction") or sig.get("Direction") or ""
                entry = float(sig.get("entry_price") or sig.get("EntryPrice") or 0)
                created = int(sig.get("created_at") or sig.get("CreatedAt") or 0)
                comment_hint = sig.get("comment_hint") or sig.get("CommentHint") or f"{prefix}_{sid}"

                matched = []
                for p in out["mt5_positions"]:
                    if sym not in p["symbol"] and p["symbol"] != sym:
                        continue
                    dir_ok = (direction == "buy" and p["type"] == "buy") or (
                        direction == "sell" and p["type"] == "sell"
                    )
                    if not dir_ok:
                        continue
                    comment_ok = p["comment"].startswith(prefix + "_") and sid in p["comment"]
                    price_ok = abs(p["price_open"] - entry) <= tol_pts * 0.01 if entry else False
                    time_ok = abs(p["time"] - created) <= win_sec if created else False
                    matched.append(
                        {
                            "ticket": p["ticket"],
                            "comment_ok": comment_ok,
                            "price_fallback_ok": price_ok and time_ok,
                            "comment": p["comment"],
                            "price_open": p["price_open"],
                            "time_diff_sec": abs(p["time"] - created) if created else None,
                        }
                    )

                out["checks"].append(
                    {
                        "name": f"match_{sid}",
                        "pass": any(m["comment_ok"] or m["price_fallback_ok"] for m in matched),
                        "signal_id": sid,
                        "comment_hint": comment_hint,
                        "entry": entry,
                        "matches": matched,
                    }
                )

    log_dir = os.path.join(os.environ.get("APPDATA", ""), "ZhuLong", "logs")
    if os.path.isdir(log_dir):
        logs = sorted(
            [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("log")],
            key=os.path.getmtime,
            reverse=True,
        )
        if logs:
            hits = []
            with open(logs[0], encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if any(k in line for k in ("托管匹配", "持仓扫描", "matched", "浮盈", "移动止损")):
                        hits.append(line.strip())
            out["zhulong_position_log"] = hits[-12:]

    con.close()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
