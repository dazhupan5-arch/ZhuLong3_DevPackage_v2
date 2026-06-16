"""
KN2 Circuit Breaker Validation (v2 - uses built-in agent breaker)
Runs KN2 backtest on 4 diverse market days.
WITH breaker vs WITHOUT breaker (via agent._breaker_enabled toggle).
"""
import torch, sys, os, json, numpy as np, pandas as pd
from pathlib import Path

INSTALL = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2")  # use dev code directly
APPDATA = Path(os.environ["APPDATA"]) / "ZhuLong"
SRC = str(INSTALL / "scripts")

df = pd.read_csv(SRC + r"\_june_multi_bars.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

def m5(sl):
    idx = pd.DatetimeIndex(sl["datetime"], tz="UTC")
    return pd.DataFrame({
        "open": sl["open"].values, "high": sl["high"].values,
        "low": sl["low"].values, "close": sl["close"].values,
        "volume": sl["volume"].fillna(0).values
    }, index=idx)

cfg = json.loads((APPDATA / "config_agent.json").read_text(encoding="utf-8-sig"))
sys.path.insert(0, str(INSTALL))
from zhulong.agent.trading_agent import TradingAgent

WINDOW = 256
TARGET_DATES = [
    ("2026-06-02", "Sideways (+0.03%, 1.74% range)"),
    ("2026-06-10", "Strong Downtrend (-4.27%, 4.47% range)"),
    ("2026-06-11", "Strong Uptrend (+3.42%, 4.82% range)"),
    ("2026-06-12", "Reversal Day (+0.08%, 1.81% range)"),
]

def run_audit(breaker_enabled: bool, label: str):
    results = {}
    for target_date, desc in TARGET_DATES:
        agent = TradingAgent(config=cfg, root=str(INSTALL))
        agent._breaker_enabled = breaker_enabled
        agent.reset_kn2_hidden()

        day_idx = df[df["datetime"].dt.strftime("%Y-%m-%d") == target_date].index.tolist()
        if len(day_idx) == 0:
            continue

        pos = None
        trades = []
        decisions = []

        for idx in day_idx:
            if idx < WINDOW:
                continue
            seg = df.iloc[idx-WINDOW:idx].copy()
            cp = float(seg["close"].iloc[-1])
            dt = str(seg["datetime"].iloc[-1])

            # Update position PnL + SL check
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
                    trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp, "xr": "SL_HIT", "pl": pnl*100})
                    pos = None
                    continue
                elif pos["d"] == "sell" and cp >= pos["sl"]:
                    trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp, "xr": "SL_HIT", "pl": pnl*100})
                    pos = None
                    continue

            m5t = m5(seg)
            acct = {"balance": 10000, "equity": 10000, "_positions": []}
            if pos:
                acct["_positions"] = [{
                    "symbol": "XAUUSD", "direction": pos["d"],
                    "open_price": pos["ep"], "sl": pos["sl"],
                    "_bars_held": pos["bars"],
                }]

            try:
                rr = agent.tick_symbols({"XAUUSD": m5t}, ["XAUUSD"], acct)
                if not rr:
                    continue
                r = rr[0]
                sig = r.get("signal") or {}
                dir_ = sig.get("direction", "flat")
                conf = sig.get("confidence", 0)
                sl_price = sig.get("sl", 0)
                reject = sig.get("reject_reason", "")

                decisions.append({"d": dir_, "c": conf, "reject": reject})

                if not pos and dir_ in ("buy", "sell"):
                    pos = {"d": dir_, "eb": idx, "edt": dt, "ep": cp, "bars": 0,
                           "mfe": 0, "mae": 0, "pnl": 0, "sl": sl_price}

                if pos and dir_ != "flat" and pos["d"] != dir_:
                    trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp,
                                   "xr": "KN2_FLIP", "pl": pos["pnl"]*100})
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
            trades.append({**pos, "xbar": "EOD", "xdt": "EOD", "xp": lp,
                           "xr": "EOD(open)", "pl": pnl*100})

        sl_trades = [t for t in trades if t["xr"] == "SL_HIT"]
        win_trades = [t for t in trades if t["pl"] > 0]
        total_pl = sum(t["pl"] for t in trades)
        longest_hold = max((t["bars"] for t in trades), default=0)
        buy_sigs = sum(1 for d in decisions if d["d"] == "buy")
        sell_sigs = sum(1 for d in decisions if d["d"] == "sell")
        breaker_rejects = sum(1 for d in decisions if "circuit_breaker" in d.get("reject", ""))

        results[target_date] = {
            "desc": desc,
            "trades": len(trades),
            "sl_hits": len(sl_trades),
            "wins": len(win_trades),
            "total_pl": total_pl,
            "longest_hold_h": longest_hold * 5 / 60,
            "buy_sigs": buy_sigs,
            "sell_sigs": sell_sigs,
            "breaker_rejects": breaker_rejects,
        }

    print(f"\n{'='*95}")
    print(f"  {label}")
    print(f"{'='*95}")
    print(f"  {'Date':>10s}  {'Market Type':<35s}  {'Trades':>6s}  {'SL':>4s}  {'Wins':>5s}  {'PnL%':>8s}  {'MaxHld':>7s}  {'Buy':>4s}  {'Sell':>4s}  {'BrkRj':>5s}")
    print(f"  {'-'*10}  {'-'*35}  {'-'*6}  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*4}  {'-'*4}  {'-'*5}")
    for td, r in results.items():
        print(f"  {td:>10s}  {r['desc']:<35s}  {r['trades']:>6d}  {r['sl_hits']:>4d}  {r['wins']:>5d}  {r['total_pl']:>+7.2f}%  {r['longest_hold_h']:>6.1f}h  {r['buy_sigs']:>4d}  {r['sell_sigs']:>4d}  {r['breaker_rejects']:>5d}")

    return results

print("Running KN2 backtests (using built-in agent breaker)...")
print()

res_no  = run_audit(False, "BASELINE: No circuit breaker")
res_yes = run_audit(True,  "WITH circuit breaker (built-in)")

print(f"\n{'='*95}")
print(f"  CROSS-DAY DELTA")
print(f"{'='*95}")
print(f"  {'Date':>10s}  {'PnL (no brkr)':>14s}  {'PnL (w/ brkr)':>14s}  {'Delta':>8s}  {'SL no->w/':>10s}  {'BrkRj':>6s}  {'Verdict':>20s}")
print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*6}  {'-'*20}")
for td in TARGET_DATES:
    d = td[0]
    if d not in res_no or d not in res_yes:
        continue
    rn = res_no[d]
    ry = res_yes[d]
    delta = ry["total_pl"] - rn["total_pl"]
    sl_change = f"{rn['sl_hits']}->{ry['sl_hits']}"
    if delta > 0.05:
        v = "Breaker IMPROVED"
    elif delta < -0.05:
        v = "Breaker HURT"
    elif ry["sl_hits"] < rn["sl_hits"]:
        v = "Less SL, similar PnL"
    elif ry["breaker_rejects"] > 0:
        v = "Breaker engaged safely"
    elif ry["breaker_rejects"] == 0:
        v = "Breaker never triggered"
    else:
        v = "Similar"
    print(f"  {d:>10s}  {rn['total_pl']:>+13.2f}%  {ry['total_pl']:>+13.2f}%  {delta:>+7.2f}%  {sl_change:>10s}  {ry['breaker_rejects']:>6d}  {v:<20s}")
print()
