import torch, sys, os, json, numpy as np, pandas as pd
from pathlib import Path

INSTALL = Path(r"C:\Program Files\ZhuLong")
APPDATA = Path(os.environ["APPDATA"]) / "ZhuLong"

# Load June 11+12 data from WCG MT5 export
df = pd.read_csv(r"D:\trae_projects\ZhuLong3_DevPackage_v2\scripts\_june12_ctx_bars.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# Mark which rows are June 12
df["is_june12"] = df["datetime"].dt.strftime("%Y%m%d") == "20260612"

print(f"Total bars: {len(df)}")
print(f"  June 11 bars: {len(df[~df['is_june12']])}")
print(f"  June 12 bars: {len(df[df['is_june12']])}")
print(f"Range: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
print(f"Price: {df['close'].iloc[0]:.1f} -> {df['close'].iloc[-1]:.1f}")
print(f"June12 range: {df[df['is_june12']]['datetime'].iloc[0]} -> {df[df['is_june12']]['datetime'].iloc[-1]}")
print(f"June12 OHLC: {df[df['is_june12']]['open'].iloc[0]:.1f} Open -> Close {df[df['is_june12']]['close'].iloc[-1]:.1f}")
print(f"June12 Hi/Lo: {df[df['is_june12']]['high'].max():.1f} / {df[df['is_june12']]['low'].min():.1f}")
print()

def m5(sl):
    idx = pd.DatetimeIndex(sl["datetime"], tz="UTC")
    return pd.DataFrame({
        "open": sl["open"].values,
        "high": sl["high"].values,
        "low": sl["low"].values,
        "close": sl["close"].values,
        "volume": sl["volume"].fillna(0).values
    }, index=idx)

cfg = json.loads((APPDATA / "config_agent.json").read_text(encoding="utf-8-sig"))
sys.path.insert(0, str(INSTALL))
from zhulong.agent.trading_agent import TradingAgent

agent = TradingAgent(config=cfg, root=str(INSTALL))
agent.reset_kn2_hidden()

WINDOW = 256
pos = None
trades = []
decisions = []
june12_indices = df[df["is_june12"]].index.tolist()

print(f"Processing June 12 only ({len(june12_indices)} bars)...")
print(f"{'Time':>10s}  {'Close':>8s}  {'Act':>5s}  {'Dir':>4s}  {'Conf':>6s}  {'SL':>8s}  {'Position':>30s}")
print("-" * 95)

for idx in june12_indices:
    if idx < WINDOW:
        continue
    seg = df.iloc[idx-WINDOW:idx].copy()
    cp = float(seg["close"].iloc[-1])
    dt = str(seg["datetime"].iloc[-1])

    m5t = m5(seg)

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

        # SL hit check
        if pos["d"] == "buy" and cp <= pos["sl"]:
            trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp, "xr": "SL_HIT", "pl": pnl*100})
            pos = None
            # Still log this bar's decision (flat since we exited)
        elif pos["d"] == "sell" and cp >= pos["sl"]:
            trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp, "xr": "SL_HIT", "pl": pnl*100})
            pos = None

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

        # Position display
        if pos and pos["d"] == "buy":
            ps = f"LONG({pos['bars']}b +{pos['pnl']*100:.2f}%)"
        elif pos:
            ps = f"SHORT({pos['bars']}b +{pos['pnl']*100:.2f}%)"
        else:
            ps = "FLAT"

        em = " *** ENTRY ***" if (not pos and dir_ in ("buy","sell")) else ""
        rj = f"  [{reject[:20]}]" if reject else ""
        bar_t = dt[-8:]
        print(f"{bar_t:>10s}  {cp:>8.2f}  {act:>5s}  {dir_:>4s}  {conf:.4f}  {sl_price:>8.2f}  {ps:>30s}{em}{rj}")

        decisions.append({"t": bar_t, "a": act, "d": dir_, "c": conf, "hp": bool(pos)})

        # Entry
        if not pos and dir_ in ("buy", "sell"):
            pos = {"d": dir_, "eb": idx, "edt": dt, "ep": cp, "bars": 0,
                   "mfe": 0, "mae": 0, "pnl": 0, "sl": sl_price}

        # Exit on KN2 direction flip (but NOT on flat)
        if pos and dir_ != "flat" and pos["d"] != dir_:
            trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp,
                           "xr": "KN2_FLIP", "pl": pos["pnl"]*100})
            pos = None

    except Exception as e:
        print(f"  ERR at {dt}: {e}")
        break

# Close open position at end of June 12
if pos:
    lp = float(df["close"].iloc[june12_indices[-1]])
    if pos["d"] == "buy":
        pnl = (lp - pos["ep"]) / pos["ep"]
    else:
        pnl = (pos["ep"] - lp) / pos["ep"]
    pos["pnl"] = pnl
    trades.append({**pos, "xbar": "EOD", "xdt": "EOD", "xp": lp,
                   "xr": "EOD(open)", "pl": pnl*100})

print()
print("=" * 70)
print("TRADE SUMMARY (June 12, 2026)")
print("=" * 70)

acts = {"hold": 0, "long": 0, "short": 0}
dirs = {"flat": 0, "buy": 0, "sell": 0}
for d in decisions:
    a = d["a"]
    if a in acts: acts[a] += 1
    dd = d["d"]
    if dd in dirs: dirs[dd] += 1

print(f"  Bars processed: {len(decisions)}  |  Trades executed: {len(trades)}")
print(f"  KN2 decisions: hold={acts['hold']} long={acts['long']} short={acts['short']}")
print(f"  Signals sent:  flat={dirs['flat']} buy={dirs['buy']} sell={dirs['sell']}")
print()

if trades:
    total_pl = 0
    for i, t in enumerate(trades, 1):
        h = t["bars"]
        print(f"  Trade #{i}: {t['d'].upper():5s}  entry={t['ep']:.1f}  exit={t['xp']:.1f}  "
              f"hold={h:>3d} bars ({h*5/60:>5.1f}h)  PnL={t['pl']:>+7.2f}%  "
              f"MFE={t['mfe']*100:>+7.2f}%  MAE={t['mae']*100:>+7.2f}%  reason={t['xr']}")
        total_pl += t['pl']
    print(f"\n  Total PnL: {total_pl:+.2f}%")
    if total_pl > 0:
        print("  VERDICT: Profitable day")
    elif total_pl < 0:
        print("  VERDICT: Losing day")
    else:
        print("  VERDICT: Breakeven")
else:
    print("  No trades executed — KN2 stayed flat all day on June 12")
