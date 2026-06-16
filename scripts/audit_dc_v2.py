"""
Full audit: KN2 with dual-channel + circuit breaker on 4 market days.
Tests the COMPLETE production code path.
"""
import torch, sys, os, json, numpy as np, pandas as pd
from pathlib import Path

INSTALL = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2")
APPDATA = Path(os.environ["APPDATA"]) / "ZhuLong"

df = pd.read_csv(INSTALL / "scripts" / "_june_multi_bars.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

sys.path.insert(0, str(INSTALL))
cfg = json.loads((APPDATA / "config_agent.json").read_text(encoding="utf-8-sig"))
from zhulong.agent.trading_agent import TradingAgent
from zhulong.strategies.indicators import atr_series

def m5(sl):
    idx = pd.DatetimeIndex(sl["datetime"], tz="UTC")
    return pd.DataFrame({"open": sl["open"].values, "high": sl["high"].values, "low": sl["low"].values, "close": sl["close"].values, "volume": sl["volume"].fillna(0).values}, index=idx)

TARGETS = [
    ("2026-06-02", "Sideways"),
    ("2026-06-10", "Downtrend"),
    ("2026-06-11", "Uptrend"),
    ("2026-06-12", "Reversal"),
]

WINDOW = 256

print("KN2 DUAL-CHANNEL AUDIT")
print(f"  dual_channel enabled:  {(cfg.get('kn2') or {}).get('dual_channel', {}).get('enabled', False)}")
print(f"  circuit_breaker enabled: {(cfg.get('kn2') or {}).get('circuit_breaker', {}).get('enabled', True)}")
print()

for target_date, label in TARGETS:
    agent = TradingAgent(config=cfg, root=str(INSTALL))
    agent._breaker_enabled = True
    agent.reset_kn2_hidden()

    day_idx = df[df["datetime"].dt.strftime("%Y-%m-%d") == target_date].index.tolist()
    if len(day_idx) == 0: continue

    pos = None
    trades = []

    for idx in day_idx:
        if idx < WINDOW: continue
        seg = df.iloc[idx-WINDOW:idx].copy()
        cp = float(seg["close"].iloc[-1])
        dt = str(seg["datetime"].iloc[-1])
        m5t = m5(seg)

        if pos:
            if pos["d"] == "buy":
                pnl = (cp - pos["ep"]) / pos["ep"]
            else:
                pnl = (pos["ep"] - cp) / pos["ep"]
            pos["pnl"] = pnl
            pos["mfe"] = max(pos.get("mfe", pnl), pnl)
            pos["mae"] = min(pos.get("mae", pnl), pnl)
            pos["bars"] += 1

            if pos["d"] == "buy" and cp <= pos["sl"]:
                trades.append({**pos, "xr": "SL_HIT", "pl": pnl*100})
                pos = None
                continue
            elif pos["d"] == "sell" and cp >= pos["sl"]:
                trades.append({**pos, "xr": "SL_HIT", "pl": pnl*100})
                pos = None
                continue

        acct = {"balance": 10000, "equity": 10000, "_positions": []}
        if pos:
            acct["_positions"] = [{"symbol": "XAUUSD", "direction": pos["d"], "open_price": pos["ep"], "sl": pos["sl"], "_bars_held": pos["bars"]}]

        try:
            rr = agent.tick_symbols({"XAUUSD": m5t}, ["XAUUSD"], acct)
            if not rr: continue
            r = rr[0]
            sig = r.get("signal") or {}
            dir_ = sig.get("direction", "flat")
            sl_price = sig.get("sl", 0)
            reject = sig.get("reject_reason", "")

            # Track direction for breakdown
            if not pos and dir_ in ("buy", "sell"):
                pos = {"d": dir_, "eb": idx, "edt": dt, "ep": cp, "bars": 0, "mfe": 0, "mae": 0, "pnl": 0, "sl": sl_price}
            if pos and dir_ != "flat" and pos["d"] != dir_:
                trades.append({**pos, "xr": f"KN2_FLIP_to_{dir_}", "pl": pos["pnl"]*100})
                pos = None
        except Exception as e:
            break

    if pos:
        lp = float(df["close"].iloc[day_idx[-1]])
        if pos["d"] == "buy":
            pnl = (lp - pos["ep"]) / pos["ep"]
        else:
            pnl = (pos["ep"] - lp) / pos["ep"]
        pos["pnl"] = pnl
        trades.append({**pos, "xr": "EOD", "pl": pnl*100})

    # Stats
    total_pl = sum(t["pl"] for t in trades)
    buys = sum(1 for t in trades if t["d"] == "buy")
    sells = sum(1 for t in trades if t["d"] == "sell")
    sls = sum(1 for t in trades if t["xr"] == "SL_HIT")
    wins = sum(1 for t in trades if t["pl"] > 0)
    longest_hold = max((t["bars"] for t in trades), default=0) * 5 / 60

    print(f"  {target_date} ({label:>10s}) | trades={len(trades):>2d} buy={buys} sell={sells} SL={sls} win={wins} | PnL={total_pl:>+6.2f}% | maxHold={longest_hold:.1f}h")
    for t in trades:
        rej = f" ({t.get('xr','')})" if t.get('xr') else ""
        xp = t.get('xp', 'EOD')
        xp_str = f"{xp:.1f}" if isinstance(xp, (int, float)) else str(xp)
        print(f"    {t['d']:>5s} ep={t['ep']:.1f} -> xp={xp_str:>6s} | {t['bars']:>3d}b | {t['pl']:>+6.2f}%{rej}")
    print()
