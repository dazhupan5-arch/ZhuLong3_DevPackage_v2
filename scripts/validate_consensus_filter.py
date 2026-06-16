"""
Validate consensus-based direction filter on all 4 market days.
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

dc_cfg = (cfg.get("kn2") or {}).get("dual_channel", {})
print(f"consensus_bars={dc_cfg.get('consensus_bars')} gap_threshold={dc_cfg.get('gap_threshold')} cooldown={dc_cfg.get('gap_cooldown_bars')}")
print()

WINDOW = 256
TARGETS = [
    ("2026-06-02", "Sideways"),
    ("2026-06-10", "Downtrend"),
    ("2026-06-11", "Uptrend"),
    ("2026-06-12", "Reversal"),
]

for target_date, label in TARGETS:
    agent = TradingAgent(config=cfg, root=str(INSTALL))
    agent._breaker_enabled = False
    agent.reset_kn2_hidden()

    day = df[df["datetime"].dt.strftime("%Y-%m-%d") == target_date]
    day_idx = day.index.tolist()
    if len(day_idx) == 0: continue

    actions = {"hold": 0, "long": 0, "short": 0}
    gaps = []

    for idx in day_idx:
        if idx < WINDOW: continue
        seg = df.iloc[idx-WINDOW:idx].copy()
        cp = float(seg["close"].iloc[-1])
        m5t = m5(seg)

        # Use agent.tick_symbols (production path)
        acct = {"balance": 10000, "equity": 10000, "_positions": []}
        try:
            rr = agent.tick_symbols({"XAUUSD": m5t}, ["XAUUSD"], acct)
            if rr and len(rr) > 0:
                sig = rr[0].get("signal", {})
                d = sig.get("direction", "flat")
                if d == "buy": actions["long"] += 1
                elif d == "sell": actions["short"] += 1
                else: actions["hold"] += 1
        except Exception as e:
            pass

        # Also collect raw gap for analysis
        if agent._kn2 and agent._kn2.is_ready:
            if hasattr(agent._kn2, '_dc_gap_history') and agent._kn2._dc_gap_history:
                gaps.append(agent._kn2._dc_gap_history[-1])

    total = actions["long"] + actions["short"]
    pl_display = f"{actions['long']}L/{actions['short']}S"
    if total > 0:
        ratio_short = actions["short"] / total * 100
    else:
        ratio_short = 0
    pct_str = f"{ratio_short:.0f}%S" if ratio_short > 0 else "noS"

    price_fmt = f"{day['open'].iloc[0]:.0f}->{day['close'].iloc[-1]:.0f}"
    avg_gap = np.mean(gaps) if gaps else 0
    gap_fraction_short_frames = np.mean([g < -0.02 for g in gaps])*100 if gaps else 0

    print(f"{target_date} {label:>10s} {price_fmt:>12s} | sigs={total:>3d} {pl_display:>7s} | gap_avg={avg_gap:+.4f} gap_short={gap_fraction_short_frames:.0f}%")

    if target_date == "2026-06-10":
        if actions["short"] > 0:
            print(f"  >>> KN2 IS SHORTING on downtrend day ({pct_str})")
        else:
            print(f"  >>> KN2 still NOT shorting on downtrend day")
    print()
