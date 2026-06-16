"""
Quick smoke test: dual-channel with production agent.tick_symbols on June 10
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

agent = TradingAgent(config=cfg, root=str(INSTALL))
agent._breaker_enabled = False
agent.reset_kn2_hidden()

dc_enabled = agent._kn2._dc_enabled if agent._kn2 else False
print(f"dual_channel={dc_enabled}")

jun10 = df[df["datetime"].dt.strftime("%Y-%m-%d") == "2026-06-10"].copy()
day_idx = jun10.index.tolist()
WINDOW = 256

actions = {"hold": 0, "long": 0, "short": 0}
errors = 0
for i, idx in enumerate(day_idx):
    if idx < WINDOW: continue
    seg = df.iloc[idx-WINDOW:idx].copy()
    cp = float(seg["close"].iloc[-1])
    dt = str(seg["datetime"].iloc[-1])
    m5t = m5(seg)
    
    try:
        acct = {"balance": 10000, "equity": 10000, "_positions": []}
        rr = agent.tick_symbols({"XAUUSD": m5t}, ["XAUUSD"], acct)
        if rr and len(rr) > 0:
            sig = rr[0].get("signal", {})
            d = sig.get("direction", "flat")
            if d in actions: actions[d] += 1
        if i < 5:
            print(f"  bar {i}: close={cp:.1f} sig={rr[0].get('signal',{}).get('direction','flat') if rr else 'none'}")
    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"  bar {i}: ERROR {e}")

total = sum(actions.values())
print(f"\nActions: hold={actions['hold']} long={actions['long']} short={actions['short']}")
print(f"Short%: {actions['short']/total*100:.1f}%" if total > 0 else "No actions")
print(f"Errors: {errors}")
