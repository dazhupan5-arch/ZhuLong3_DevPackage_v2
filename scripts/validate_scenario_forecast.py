"""
Validate scenario-based forecast decision on June 10 downtrend.
Tests both direction detection and trade frequency.
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
from zhulong.agent.knowledge_net_kn2 import encode_position_state

agent = TradingAgent(config=cfg, root=str(INSTALL))
agent._breaker_enabled = False
agent.reset_kn2_hidden()

# Show config
dc = (cfg.get("kn2") or {}).get("dual_channel", {})
print(f"Config: window={dc.get('scenario_window')} long_pct={dc.get('long_percentile')} short_pct={dc.get('short_percentile')} gap={dc.get('min_gap_bars')}")

WINDOW = 256

def m5(sl):
    idx = pd.DatetimeIndex(sl["datetime"], tz="UTC")
    return pd.DataFrame({"open": sl["open"].values, "high": sl["high"].values, "low": sl["low"].values, "close": sl["close"].values, "volume": sl["volume"].fillna(0).values}, index=idx)

TARGETS = [
    ("2026-06-02", "Sideways"),
    ("2026-06-10", "Downtrend"),
    ("2026-06-11", "Uptrend"),
    ("2026-06-12", "Reversal"),
]

print(f"\n{'Date':>12s}  {'Label':>12s}  {'Price':>14s}  {'Trades':>6s}  {'Buy':>5s}  {'Sell':>5s}  {'PnL':>8s}")
print(f"{'='*80}")

for target_date, label in TARGETS:
    agent.reset_kn2_hidden()
    day = df[df["datetime"].dt.strftime("%Y-%m-%d") == target_date]
    if len(day) == 0: continue
    day_idx = day.index.tolist()
    
    actions = {"hold": 0, "long": 0, "short": 0}
    pct_values = []
    delta_means = []
    
    for idx in day_idx:
        if idx < WINDOW: continue
        seg = df.iloc[idx-WINDOW:idx].copy()
        cp = float(seg["close"].iloc[-1])
        m5t = m5(seg)
        atr_s = atr_series(m5t)
        atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else cp * 0.001
        
        # Full production path
        acct = {"balance": 10000, "equity": 10000, "_positions": []}
        try:
            rr = agent.tick_symbols({"XAUUSD": m5t}, ["XAUUSD"], acct)
            if rr and len(rr) > 0:
                sig = rr[0].get("signal", {})
                d = sig.get("direction", "flat")
                # Map direction
                if d == "buy":      actions["long"] += 1
                elif d == "sell":   actions["short"] += 1
                else:               actions["hold"] += 1
        except:
            pass
    
    total_trades = actions["long"] + actions["short"]
    price_range = f"{day['open'].iloc[0]:.0f}->{day['close'].iloc[-1]:.0f}"
    pnl = "+?.??%"
    
    print(f"{target_date:>12s}  {label:>12s}  {price_range:>14s}  {total_trades:>6d}  {actions['long']:>5d}  {actions['short']:>5d}")
    
    if target_date == "2026-06-10":
        A = actions
        print(f"\n  >>> June 10 (downtrend): {A['long']}L/{A['short']}S ({A['short']/max(total_trades,1)*100:.0f}% short) | {total_trades} signals")
    else:
        print(f"  >>> {label}: {actions['long']}L/{actions['short']}S")

print()
