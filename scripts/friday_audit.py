import torch, sys, os, json, numpy as np, pandas as pd
from pathlib import Path

INSTALL = Path(r"C:\Program Files\ZhuLong")
APPDATA = Path(os.environ["APPDATA"]) / "ZhuLong"

csv = r"C:\Users\xiaomi\Desktop\XAUUSD5.csv"
df = pd.read_csv(csv, header=None, names=["date","time","open","high","low","close","volume"])
df = df.dropna(subset=["open","high","low","close"])
df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
df = df.sort_values("datetime")

def m5(sl):
    idx = pd.DatetimeIndex(sl["datetime"], tz="UTC")
    return pd.DataFrame({"open":sl["open"].values,"high":sl["high"].values,
        "low":sl["low"].values,"close":sl["close"].values,
        "volume":sl["volume"].fillna(0).values}, index=idx)

cfg = json.loads((APPDATA / "config_agent.json").read_text(encoding="utf-8-sig"))
sys.path.insert(0, str(INSTALL))
from zhulong.agent.trading_agent import TradingAgent

agent = TradingAgent(config=cfg, root=str(INSTALL))
agent.reset_kn2_hidden()

# Find last Friday in dataset
df["ymd"] = df["datetime"].dt.strftime("%Y%m%d")
all_dates = sorted(df["ymd"].unique())
friday_d = None
for d in reversed(all_dates):
    wd = pd.Timestamp(d[:4]+"-"+d[4:6]+"-"+d[6:]).weekday()
    if wd == 4:
        friday_d = d
        break

fri = df[df["ymd"] == friday_d]
print(f"Friday: {friday_d}  ({len(fri)} bars)")
print(f"  Time: {fri['datetime'].iloc[0]} -> {fri['datetime'].iloc[-1]}")

WINDOW = 256
fidx = fri.index.tolist()
pos = None       # position state
trades = []
decisions = []

for idx in fidx:
    if idx < WINDOW:
        continue
    seg = df.iloc[idx-WINDOW:idx].copy()
    cp = float(seg["close"].iloc[-1])
    dt = str(seg["datetime"].iloc[-1])
    m5t = m5(seg)

    # Update position PnL
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
            trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp, "xr": "SL_HIT", "pl": pnl*100})
            pos = None
            continue
        elif pos["d"] == "sell" and cp >= pos["sl"]:
            trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp, "xr": "SL_HIT", "pl": pnl*100})
            pos = None
            continue

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
        sl = sig.get("sl", 0)
        reject = sig.get("reject_reason", "")

        # Position display
        if pos and pos["d"] == "buy":
            ps = f"LONG({pos['bars']}b +{pos['pnl']*100:.2f}%)"
        elif pos:
            ps = f"SHORT({pos['bars']}b +{pos['pnl']*100:.2f}%)"
        else:
            ps = "FLAT"

        em = " *** ENTRY ***" if (not pos and dir_ in ("buy","sell")) else ""
        rj = f"  [{reject[:25]}]" if reject else ""
        bar_t = dt[-8:]
        print(f"  {bar_t}  {act:>5s}  {dir_:>4s}  c={conf:.4f}  sl={sl:>8.2f}  {ps:>28s}{em}{rj}")

        decisions.append({"t": bar_t, "a": act, "d": dir_, "c": conf, "hp": bool(pos)})

        # Entry
        if not pos and dir_ in ("buy", "sell"):
            pos = {"d": dir_, "eb": idx, "edt": dt, "ep": cp, "bars": 0,
                   "mfe": 0, "mae": 0, "pnl": 0, "sl": sl}

        # Exit on KN2 direction flip
        if pos and pos["d"] != dir_:
            trades.append({**pos, "xbar": idx, "xdt": dt, "xp": cp,
                           "xr": "KN2_FLIP", "pl": pos["pnl"]*100})
            pos = None

    except Exception as e:
        print(f"  ERR {dt}: {e}")
        import traceback
        traceback.print_exc()
        break

# Close open position at EOD
if pos:
    lp = float(df["close"].iloc[fidx[-1]])
    if pos["d"] == "buy":
        pnl = (lp - pos["ep"]) / pos["ep"]
    else:
        pnl = (pos["ep"] - lp) / pos["ep"]
    pos["pnl"] = pnl
    trades.append({**pos, "xbar": "EOD", "xdt": "EOD", "xp": lp,
                   "xr": "EOD(open)", "pl": pnl*100})

print()

# Summary
acts = {"hold": 0, "long": 0, "short": 0}
dirs = {"flat": 0, "buy": 0, "sell": 0}
for d in decisions:
    a = d["a"]
    if a in acts: acts[a] += 1
    dd = d["d"]
    if dd in dirs: dirs[dd] += 1

print(f"  Bars: {len(decisions)}  |  Trades: {len(trades)}")
print(f"  KN2 internal: hold={acts['hold']} long={acts['long']} short={acts['short']}")
print(f"  Signals sent: flat={dirs['flat']} buy={dirs['buy']} sell={dirs['sell']}")
print()

if trades:
    for i, t in enumerate(trades, 1):
        h = t["bars"]
        print(f"  Trade #{i}: {t['d'].upper()}  entry={t['ep']:.1f}  exit={t['xp']:.1f}  "
              f"hold={h} bars ({h*5/60:.1f}h)  PnL={t['pl']:+.2f}%  "
              f"MFE={t['mfe']*100:+.2f}%  MAE={t['mae']*100:+.2f}%  r={t['xr']}")
else:
    print("  No trades today — KN2 stayed flat all day")
