"""
KN2 Circuit Breaker Validation
Runs KN2 backtest on 4 diverse market days (June 2, 10, 11, 12)
Comparing: WITHOUT breaker vs WITH breaker
Circuit breaker: 3 SL hits in 12 bars -> cooldown 24 bars
No parameter tuning - pure structural validation.
"""
import torch, sys, os, json, numpy as np, pandas as pd
from pathlib import Path

INSTALL = Path(r"C:\Program Files\ZhuLong")
APPDATA = Path(os.environ["APPDATA"]) / "ZhuLong"
SRC = r"D:\trae_projects\ZhuLong3_DevPackage_v2\scripts"

# ── Load all data ─────────────────────────────────────────
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

# ── Circuit breaker params (structural, not tuned to any day) ──
SL_LOOKBACK = 12   # bars to look back for SL count
SL_THRESHOLD = 3   # number of SL hits to trigger
COOLDOWN_BARS = 24 # bars to stay flat after trigger

WINDOW = 256

TARGET_DATES = [
    ("2026-06-02", "Sideways (+0.03%, 1.74% range)"),
    ("2026-06-10", "Strong Downtrend (-4.27%, 4.47% range)"),
    ("2026-06-11", "Strong Uptrend (+3.42%, 4.82% range)"),
    ("2026-06-12", "Reversal Day (+0.08%, 1.81% range)"),
]

def run_audit(use_breaker: bool, label: str):
    """Run KN2 backtest on all 4 days. Returns summary dict per day."""
    results = {}
    for target_date, desc in TARGET_DATES:
        agent = TradingAgent(config=cfg, root=str(INSTALL))
        agent.reset_kn2_hidden()

        day_idx = df[df["datetime"].dt.strftime("%Y-%m-%d") == target_date].index.tolist()
        if len(day_idx) == 0:
            continue

        pos = None
        trades = []
        decisions = []
        sl_events = []  # list of bar indices where SL hit
        cooldown_until = -1  # bar index until which cooldown is active

        for idx in day_idx:
            if idx < WINDOW:
                continue
            seg = df.iloc[idx-WINDOW:idx].copy()
            cp = float(seg["close"].iloc[-1])
            dt = str(seg["datetime"].iloc[-1])

            # Check cooldown expiry
            breaker_active = use_breaker and idx < cooldown_until

            # Update position + SL check
            if pos:
                if pos["d"] == "buy":
                    pnl = (cp - pos["ep"]) / pos["ep"]
                else:
                    pnl = (pos["ep"] - cp) / pos["ep"]
                pos["pnl"] = pnl
                pos["mfe"] = max(pos.get("mfe", pnl), pnl)
                pos["mae"] = min(pos.get("mae", pnl), pnl)
                pos["bars"] += 1

                # SL hit
                if pos["d"] == "buy" and cp <= pos["sl"]:
                    trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp,
                                   "xr": "SL_HIT", "pl": pnl*100, "cooldown": breaker_active})
                    sl_events.append(idx)
                    pos = None
                    # Circuit breaker: check recent SL count
                    if use_breaker:
                        recent_sl = sum(1 for s in sl_events if s > idx - SL_LOOKBACK)
                        if recent_sl >= SL_THRESHOLD:
                            cooldown_until = idx + COOLDOWN_BARS
                    continue
                elif pos["d"] == "sell" and cp >= pos["sl"]:
                    trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp,
                                   "xr": "SL_HIT", "pl": pnl*100, "cooldown": breaker_active})
                    sl_events.append(idx)
                    pos = None
                    if use_breaker:
                        recent_sl = sum(1 for s in sl_events if s > idx - SL_LOOKBACK)
                        if recent_sl >= SL_THRESHOLD:
                            cooldown_until = idx + COOLDOWN_BARS
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
                act = r.get("action")
                dir_ = sig.get("direction", "flat")
                conf = sig.get("confidence", 0)
                sl_price = sig.get("sl", 0)
                reject = sig.get("reject_reason", "")

                decisions.append({"a": act, "d": dir_, "c": conf, "breaker": breaker_active})

                # ── Entry: skip if breaker active ──
                if breaker_active:
                    # Force flat during cooldown
                    if pos and dir_ != "flat" and pos["d"] != dir_:
                        # KN2 wants to flip direction - close position
                        trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp,
                                       "xr": "BREAKER_EXIT", "pl": pos["pnl"]*100})
                        pos = None
                    continue

                if not pos and dir_ in ("buy", "sell"):
                    pos = {"d": dir_, "eb": idx, "edt": dt, "ep": cp, "bars": 0,
                           "mfe": 0, "mae": 0, "pnl": 0, "sl": sl_price}

                # Exit on KN2 flip
                if pos and dir_ != "flat" and pos["d"] != dir_:
                    trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp,
                                   "xr": "KN2_FLIP", "pl": pos["pnl"]*100})
                    pos = None

            except Exception as e:
                break

        # Close open at EOD
        if pos:
            lp = float(df["close"].iloc[day_idx[-1]])
            if pos["d"] == "buy":
                pnl = (lp - pos["ep"]) / pos["ep"]
            else:
                pnl = (pos["ep"] - lp) / pos["ep"]
            pos["pnl"] = pnl
            trades.append({**pos, "xbar": "EOD", "xdt": "EOD", "xp": lp,
                           "xr": "EOD(open)", "pl": pnl*100})

        # Stats
        buy_sigs = sum(1 for d in decisions if d["d"] == "buy")
        sell_sigs = sum(1 for d in decisions if d["d"] == "sell")
        total_pl = sum(t["pl"] for t in trades)
        sl_trades = [t for t in trades if t["xr"] == "SL_HIT"]
        win_trades = [t for t in trades if t["pl"] > 0]
        breaker_triggers = sum(1 for d in decisions if d.get("breaker"))
        longest_hold = max((t["bars"] for t in trades), default=0)

        results[target_date] = {
            "desc": desc,
            "trades": len(trades),
            "sl_hits": len(sl_trades),
            "wins": len(win_trades),
            "total_pl": total_pl,
            "longest_hold_h": longest_hold * 5 / 60,
            "buy_sigs": buy_sigs,
            "sell_sigs": sell_sigs,
            "breaker_bars": breaker_triggers,
        }

    # Print comparison table
    print(f"\n{'='*95}")
    print(f"  {label}")
    print(f"  Breaker: {'ON' if use_breaker else 'OFF'} | SL_LOOKBACK={SL_LOOKBACK} SL_THRESHOLD={SL_THRESHOLD} COOLDOWN={COOLDOWN_BARS}")
    print(f"{'='*95}")
    print(f"  {'Date':>10s}  {'Market Type':<35s}  {'Trades':>6s}  {'SL Hit':>6s}  {'Wins':>5s}  {'PnL%':>8s}  {'MaxHld':>7s}  {'Buy':>4s}  {'Sell':>4s}  {'BrkBrs':>6s}")
    print(f"  {'-'*10}  {'-'*35}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*4}  {'-'*4}  {'-'*6}")
    for td, r in results.items():
        print(f"  {td:>10s}  {r['desc']:<35s}  {r['trades']:>6d}  {r['sl_hits']:>6d}  {r['wins']:>5d}  {r['total_pl']:>+7.2f}%  {r['longest_hold_h']:>6.1f}h  {r['buy_sigs']:>4d}  {r['sell_sigs']:>4d}  {r['breaker_bars']:>6d}")

    return results

# ── Run both ───────────────────────────────────────────────
print("Running KN2 backtests on 4 diverse market days...")
print("(This takes ~2-3 min, loading KN2 model once per run)")
print()

res_no = run_audit(use_breaker=False, label="BASELINE: No circuit breaker")
res_yes = run_audit(use_breaker=True, label="WITH circuit breaker")

# ── Cross-day comparison ───────────────────────────────────
print(f"\n{'='*95}")
print(f"  CROSS-DAY COMPARISON")
print(f"{'='*95}")
print(f"  {'Date':>10s}  {'PnL (no brkr)':>14s}  {'PnL (w/ brkr)':>14s}  {'Delta':>8s}  {'SL no->w/':>10s}  {'Brkr Bars':>10s}  {'Verdict':>20s}")
print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*20}")
for td in TARGET_DATES:
    d = td[0]
    if d not in res_no or d not in res_yes:
        continue
    rn = res_no[d]
    ry = res_yes[d]
    delta = ry["total_pl"] - rn["total_pl"]
    sl_change = f"{rn['sl_hits']}->{ry['sl_hits']}"
    # Verdict
    if delta > 0.1:
        v = "Breaker IMPROVED"
    elif delta < -0.1:
        v = "Breaker HURT"
    elif ry["sl_hits"] < rn["sl_hits"]:
        v = "Less SL, similar PnL"
    elif ry["breaker_bars"] > 0 and ry["sl_hits"] < rn["sl_hits"]:
        v = "Breaker REDUCED SL"
    elif ry["breaker_bars"] == 0:
        v = "Breaker NEVER TRIGGERED"
    else:
        v = "Similar"
    print(f"  {d:>10s}  {rn['total_pl']:>+13.2f}%  {ry['total_pl']:>+13.2f}%  {delta:>+7.2f}%  {sl_change:>10s}  {ry['breaker_bars']:>10d}  {v:<20s}")

print()
