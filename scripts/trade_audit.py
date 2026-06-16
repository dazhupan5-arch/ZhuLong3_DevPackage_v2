import torch  # MUST be first import (DLL env fix)
import sys, os, json, numpy as np, pandas as pd
from pathlib import Path

INSTALL = Path(r"C:\Program Files\ZhuLong")
APPDATA = Path(os.environ["APPDATA"]) / "ZhuLong"
PY = r"C:\Users\xiaomi\AppData\Local\Programs\Python\Python311\python.exe"

csv_path = r"C:\Users\xiaomi\Desktop\XAUUSD5.csv"
df = pd.read_csv(csv_path, header=None, names=["date","time","open","high","low","close","volume"])
df = df.dropna(subset=["open","high","low","close"])
df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
df = df.sort_values("datetime")

def m5_of(sl):
    idx = pd.DatetimeIndex(sl["datetime"], tz="UTC")
    return pd.DataFrame({"open":sl["open"].values,"high":sl["high"].values,
        "low":sl["low"].values,"close":sl["close"].values,
        "volume":sl["volume"].fillna(0).values}, index=idx)

# ═══ CLOSED-LOOP VERIFICATION ═══
print("=" * 70)
print("  v3.1.7 CLOSED-LOOP AUDIT + Trade Analysis")
print("=" * 70)
print()

cfg_a = json.loads((APPDATA / "config_agent.json").read_text(encoding="utf-8-sig"))
cfg_r = json.loads((INSTALL / "config.json").read_text(encoding="utf-8-sig"))

ok = lambda n,c: print(f"  {'[OK]' if c else '[BUG]'} {n}")

ok("kn2.enabled=true", cfg_a["kn2"]["enabled"] is True)
ok("kn2.shadow_mode=false", cfg_a["kn2"]["shadow_mode"] is False)
ok("kn2.min_confidence=0.48", cfg_a["kn2"]["min_confidence"] == 0.48)
ok("primary=XAUUSD", cfg_a["primary_symbol"] == "XAUUSD")
ok("model exists", (INSTALL / "models/kn2_trader.pth").is_file())
ok("causal_graph.json", (INSTALL / "config/causal_graph.json").is_file())
trading_src = (INSTALL / "zhulong/agent/trading_agent.py").read_text(encoding="utf-8")
ok("filter_reason guard", "not kn2_dictator_active and filter_reason" in trading_src)
ok("scheduler guard", "not kn2_dictator_active" in trading_src)

print()

# ═══ LOAD AGENT ═══
sys.path.insert(0, str(INSTALL))
from zhulong.agent.trading_agent import TradingAgent

agent = TradingAgent(config=cfg_a, root=str(INSTALL))
agent.reset_kn2_hidden()
print(f"Agent: kn2_mode={agent.kn2_mode} shadow={agent.kn2_shadow} ready={agent._kn2.is_ready}")
print()

# ═══ SIMULATION ═══
WINDOW, START, NUM_BARS = 256, len(df) - 500, 500
bars_log, trades = [], []
pos = None  # {dir, entry_bar, entry_dt, entry_price, bars_held, mfe, mae, pnl_pct}
print(f"Simulating {NUM_BARS} bars (500→{START+NUM_BARS-1})...")
print()

for i in range(START, START+NUM_BARS):
    seg = df.iloc[i-WINDOW:i].copy()
    cp = float(seg["close"].iloc[-1])
    hp = float(seg["high"].iloc[-1])
    lp = float(seg["low"].iloc[-1])
    dt = str(seg["datetime"].iloc[-1])
    m5t = m5_of(seg)

    # Update position
    if pos:
        pnl = (cp-pos["ep"])/pos["ep"] if pos["d"]=="buy" else (pos["ep"]-cp)/pos["ep"]
        pos["pnl"]=pnl; pos["mfe"]=max(pos.get("mfe",pnl),pnl)
        pos["mae"]=min(pos.get("mae",pnl),pnl); pos["bars"]+=1
        # SL check using KN2's SL price
        if pos["d"]=="buy" and cp<=pos["sl"]:
            trades.append({**pos,"xbar":i,"xdt":dt,"xp":cp,"xr":"SL","pl":pnl*100})
            pos=None; continue
        elif pos["d"]=="sell" and cp>=pos["sl"]:
            trades.append({**pos,"xbar":i,"xdt":dt,"xp":cp,"xr":"SL","pl":pnl*100})
            pos=None; continue

    acct={"balance":10000,"equity":10000}
    if pos:
        acct["equity"]=10000*(1+pos["pnl"]*0.1)
        acct["_positions"]=[{
            "symbol":"XAUUSD","direction":pos["d"],"open_price":pos["ep"],
            "sl":pos["ep"]*.98 if pos["d"]=="buy" else pos["ep"]*1.02,
            "tp":pos["ep"]*1.02 if pos["d"]=="buy" else pos["ep"]*.98,
            "_bars_held":pos["bars"],
            "_mfe":pos["ep"]*(1+pos["mfe"]) if pos["d"]=="buy" else pos["ep"]*(1-pos["mfe"]),
            "_mae":pos["ep"]*(1+pos["mae"]) if pos["d"]=="buy" else pos["ep"]*(1-pos["mae"]),
        }]
    else:
        acct["_positions"]=[]

    try:
        results = agent.tick_symbols({"XAUUSD":m5t},["XAUUSD"],acct)
        if not results: continue
        r=results[0]; sig=r.get("signal") or {}
        act=r.get("action"); dir_=sig.get("direction","flat")
        conf=sig.get("confidence",0); sl=sig.get("sl",0); tp=sig.get("tp",0)

        # Entry
        if not pos and dir_ in ("buy","sell"):
            pos={"d":dir_,"eb":i,"edt":dt,"ep":cp,"bars":0,"mfe":0,"mae":0,"pnl":0,"sl":sl,"tp":tp}

        # Exit: KN2 switches from long to hold when we have a long position
        if pos and pos["d"]!=dir_:
            trades.append({**pos,"xbar":i,"xdt":dt,"xp":cp,"xr":"KN2_SWITCH","pl":pos["pnl"]*100})
            pos=None

        bars_log.append({"bar":i,"act":act,"dir":dir_,"conf":conf,"has_pos":bool(pos)})

    except Exception as ex:
        print(f"  BAR {i} ERROR: {ex}"); break

# Close open position at simulation end
if pos:
    last_price = float(df["close"].iloc[min(START+NUM_BARS-1, len(df)-1)])
    pos["pnl"] = (last_price-pos["ep"])/pos["ep"] if pos["d"]=="buy" else (pos["ep"]-last_price)/pos["ep"]
    trades.append({**pos,"xbar":START+NUM_BARS-1,"xdt":str(df["datetime"].iloc[START+NUM_BARS-1]),
                   "xp":last_price,"xr":"SIM_END(open)","pl":pos["pnl"]*100})

print()

# ═══════════════════════════════════════════════════════════════════
#  TRADE ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("  TRADE-BY-TRADE ANALYSIS")
print("=" * 70)

active = sum(1 for b in bars_log if b.get("bar"))
print(f"\n  Bars: {active}  |  Trades: {len(trades)}")
print()

if not trades:
    print("  NO TRADES — KN2 never entered")
else:
    for idx, t in enumerate(trades, 1):
        hb = t["bars"]
        print(f"  ╔══ TRADE #{idx} {'═'*55}")
        print(f"  ║  Dir:     {t['d'].upper()}")
        print(f"  ║  Entry:   bar {t['eb']} @ {t['ep']:.1f}")
        print(f"  ║  Time:    {t['edt']}")
        print(f"  ║  Exit:    bar {t['xbar']} @ {t['xp']:.1f}")
        print(f"  ║  Time:    {t['xdt']}")
        print(f"  ║  Reason:  {t['xr']}")
        print(f"  ║  Hold:    {hb} bars = {hb*5/60:.1f} hours")
        print(f"  ║  PnL:     {t['pl']:+.2f}%")
        print(f"  ║  MFE:     {t['mfe']*100:+.2f}%")
        print(f"  ║  MAE:     {t['mae']*100:+.2f}%")

        # Quality
        if hb > 100: q="[A+] Trend-hold — major trend captured"
        elif hb > 50: q="[A] Long-term hold"
        elif hb > 20: q="[B] Medium swing"
        elif hb > 5: q="[C] Intraday"
        else: q="[D] Scalp"

        if t["pl"] > 0 and hb > 50: q+=", profitable trend trade"
        elif t["pl"] > 0: q+=", profitable"
        else: q+=", loss"

        print(f"  ║  Quality: {q}")

        # Price trajectory
        if hb > 3:
            e_bar = t["eb"] - START
            x_bar = t["xbar"] - START if isinstance(t["xbar"],int) else min(active, NUM_BARS)
            if x_bar > e_bar:
                sp = df.iloc[e_bar:x_bar+1]["close"].values
                pct = (sp - t["ep"])/t["ep"]*100
                print(f"  ║  Path:    {pct[0]:+.2f}% → {pct[-1]:+.2f}%  "
                      f"(peak={pct.max():+.2f}%  trough={pct.min():+.2f}%)")
        print(f"  ╚{'═'*65}")
        print()

    # Summary
    pls = [t["pl"] for t in trades]
    hbs = [t["bars"] for t in trades]
    wins = sum(1 for p in pls if p>0)
    print(f"  ┌─ SUMMARY {'─'*55}")
    print(f"  │  Trades:      {len(trades)}")
    print(f"  │  Wins:        {wins}/{len(trades)} ({wins/len(trades)*100:.0f}%)" if len(trades)>0 else "  │  Wins:  N/A")
    print(f"  │  Avg PnL:     {np.mean(pls):+.2f}%")
    print(f"  │  Total PnL:   {sum(pls):+.2f}%")
    print(f"  │  Avg hold:    {np.mean(hbs):.1f} bars = {np.mean(hbs)*5/60:.1f}h")
    print(f"  │  Max hold:    {max(hbs)} bars = {max(hbs)*5/60:.1f}h")
    print(f"  └{'─'*65}")
    print()

    mx = max(hbs)
    if mx > 200: print("  [A+] Super hold system — KN2 captured and held the trend")
    elif mx > 100: print("  [A] Excellent hold system — trend-following good")
    elif mx > 50: print("  [B] Medium hold system")
    else: print("  [C] Short-term system — not ideal hold")

# Action stats
act_dist={"hold":0,"long":0,"short":0}
for b in bars_log:
    a=b.get("act","hold")
    if a in act_dist: act_dist[a]+=1
print(f"\n  KN2: hold={act_dist['hold']} long={act_dist['long']} short={act_dist['short']}")

dir_dist={"flat":0,"buy":0,"sell":0}
for b in bars_log:
    d=b.get("dir","flat")
    if d in dir_dist: dir_dist[d]+=1
print(f"  Sig:  flat={dir_dist['flat']} buy={dir_dist['buy']} sell={dir_dist['sell']}")

print()
print("=" * 70)
