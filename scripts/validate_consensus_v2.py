"""
Fast validation: relative-baseline consensus filter.
"""
import torch, sys, os, json, numpy as np, pandas as pd
from pathlib import Path

INSTALL = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2")
APPDATA = Path(os.environ["APPDATA"]) / "ZhuLong"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

df = pd.read_csv(INSTALL / "scripts" / "_june_multi_bars.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

sys.path.insert(0, str(INSTALL))
cfg = json.loads((APPDATA / "config_agent.json").read_text(encoding="utf-8-sig"))
from zhulong.agent.trading_agent import TradingAgent
from zhulong.strategies.indicators import atr_series
from zhulong.agent.knowledge_net_kn2 import encode_position_state

def m5(sl):
    idx = pd.DatetimeIndex(sl["datetime"], tz="UTC")
    return pd.DataFrame({"open": sl["open"].values, "high": sl["high"].values, "low": sl["low"].values, "close": sl["close"].values, "volume": sl["volume"].fillna(0).values}, index=idx)

dc_cfg = (cfg.get("kn2") or {}).get("dual_channel", {})
print(f"consensus_bars={dc_cfg.get('consensus_bars')} gap_threshold={dc_cfg.get('gap_threshold')} cooldown={dc_cfg.get('gap_cooldown_bars')}")
print()

print("Initializing TradingAgent...")
agent = TradingAgent(config=cfg, root=str(INSTALL))
agent._breaker_enabled = False
print()

WINDOW = 256
TARGETS = [
    ("2026-06-02", "Sideways"),
    ("2026-06-10", "Downtrend"),
    ("2026-06-11", "Uptrend"),
    ("2026-06-12", "Reversal"),
]

for target_date, label in TARGETS:
    agent.reset_kn2_hidden()
    
    if agent._kn2 is None:
        print(f"{target_date}: no KN2")
        continue

    day = df[df["datetime"].dt.strftime("%Y-%m-%d") == target_date]
    day_idx = day.index.tolist()
    if len(day_idx) == 0: continue

    actions = {"hold": 0, "long": 0, "short": 0}
    gaps, deviations, bar_count = [], [], 0

    for idx in day_idx:
        if idx < WINDOW: continue
        bar_count += 1
        seg = df.iloc[idx-WINDOW:idx].copy()
        cp = float(seg["close"].iloc[-1])
        m5t = m5(seg)
        atr_s = atr_series(m5t)
        atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else cp * 0.001

        kn_row = agent._build_knowledge_features(m5t, agent.structure.compute_latest({"M5": m5t}))
        struct = agent.structure.compute_latest({"M5": m5t})
        
        struct_30 = np.asarray(struct[:30], dtype=np.float32)
        if struct_30.shape[0] < 30: struct_30 = np.pad(struct_30, (0, 30 - struct_30.shape[0]))
        v14_feat = np.asarray(kn_row, dtype=np.float32).reshape(-1)
        if v14_feat.shape[0] < 68: v14_feat = np.pad(v14_feat, (0, 68 - v14_feat.shape[0]))
        mf = np.concatenate([v14_feat[:68], struct_30[:30]])
        ps = encode_position_state()

        decision = agent._kn2.predict(mf, ps, close=cp, atr=atr)
        a_name = decision["action_name"]
        if a_name == "long": actions["long"] += 1
        elif a_name == "short": actions["short"] += 1
        else: actions["hold"] += 1

        if hasattr(agent._kn2, '_dc_gap_history') and agent._kn2._dc_gap_history:
            g = agent._kn2._dc_gap_history[-1]
            gaps.append(g)
            if hasattr(agent._kn2, '_dc_gap_ema') and agent._kn2._dc_gap_ema is not None:
                deviations.append(g - agent._kn2._dc_gap_ema)

    total = actions["long"] + actions["short"]
    ratio_s = actions["short"] / max(total, 1) * 100
    avg_gap = np.mean(gaps) if gaps else 0
    avg_dev = np.mean(deviations) if deviations else 0
    max_dev = max(abs(np.min(deviations)), abs(np.max(deviations))) if deviations else 0
    price_fmt = f"{day['open'].iloc[0]:.0f}->{day['close'].iloc[-1]:.0f}"

    print(f"{target_date} {label:>10s} {price_fmt:>12s} | {bar_count:>3d}b {actions['long']:>3d}L/{actions['short']:>3d}S/{actions['hold']:>3d}H | gap_avg={avg_gap:+.4f} dev_range=[{np.min(deviations):+.4f},{np.max(deviations):+.4f}]")
    
    if target_date == "2026-06-10":
        if actions["short"] > 0:
            print(f"  >>> SHORTING! ({ratio_s:.0f}% short)")
        else:
            print(f"  >>> Still no shorts (gap threshold {dc_cfg.get('gap_threshold')} vs max_abs_dev={max_dev:.4f})")
    print()
