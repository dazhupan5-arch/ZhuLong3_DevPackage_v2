"""
Diagnostic: print scenario_head delta_price values over June 10
to understand value range and calibration needs.
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

WINDOW = 256

def m5(sl):
    idx = pd.DatetimeIndex(sl["datetime"], tz="UTC")
    return pd.DataFrame({"open": sl["open"].values, "high": sl["high"].values, "low": sl["low"].values, "close": sl["close"].values, "volume": sl["volume"].fillna(0).values}, index=idx)

# June 10 + June 11 for diversity
for label, date_range in [("6/10 downtrend", slice("2026-06-10", "2026-06-10")), ("6/11 uptrend", slice("2026-06-11", "2026-06-11"))]:
    day = df[df["datetime"].dt.strftime("%Y-%m-%d") == date_range.stop].copy()
    day_idx = day.index.tolist()
    
    print(f"\n{'='*80}")
    print(f"  SCENARIO DELTA_PRICE DIAGNOSTIC - {label}")
    print(f"{'='*80}")
    print(f"  {'Bar':>4s} {'close':>8s} {'action_h':>8s} | {'S0':>8s} {'S1':>8s} {'S2':>8s} {'S3':>8s} {'S4':>8s} {'S5':>8s} {'S6':>8s} {'S7':>8s} | {'mean':>8s} {'std':>8s} {'consens':>8s}")
    print(f"  {'-'*4} {'-'*8} {'-'*8} | {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} | {'-'*8} {'-'*8} {'-'*8}")
    
    all_deltas = []
    sampled_bars = list(range(0, 40, 4)) + list(range(50, 70, 4)) + list(range(100, 120, 4)) + list(range(180, 210, 4))
    sampled_bars = sorted(set(sampled_bars))
    
    for j, idx in enumerate(day_idx):
        if idx < WINDOW: continue
        bar_pos = idx - day_idx[0] - WINDOW
        if bar_pos < 0: continue
        
        seg = df.iloc[idx-WINDOW:idx].copy()
        cp = float(seg["close"].iloc[-1])
        m5t = m5(seg)
        atr_s = atr_series(m5t)
        atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else cp * 0.001
        
        kn_row = agent._build_knowledge_features(m5t, agent.structure.compute_latest({"M5": m5t}))
        struct_vals = agent.structure.compute_latest({"M5": m5t})
        struct_30 = np.asarray(struct_vals[:30], dtype=np.float32)
        if struct_30.shape[0] < 30: struct_30 = np.pad(struct_30, (0, 30 - struct_30.shape[0]))
        v14_feat = np.asarray(kn_row, dtype=np.float32).reshape(-1)
        if v14_feat.shape[0] < 68: v14_feat = np.pad(v14_feat, (0, 68 - v14_feat.shape[0]))
        mf = np.concatenate([v14_feat[:68], struct_30[:30]])
        ps = encode_position_state()
        
        with torch.no_grad():
            mf_t = torch.tensor(mf.reshape(1, -1))
            ps_t = torch.tensor(ps.reshape(1, -1))
            h_prev = torch.tensor(agent._kn2._h) if agent._kn2._h is not None else None
            out = agent._kn2.model(mf_t, h_prev, ps_t)
            h_new = out["hidden"]
            
            # Get scenario predictions
            scenarios = out["scenarios"][0].numpy()  # (64,)
            scenario_reshaped = scenarios.reshape(8, 8)
            delta_prices = scenario_reshaped[:, 0]  # first param per scenario
            delta_vols = scenario_reshaped[:, 1] if scenario_reshaped.shape[1] > 1 else np.zeros(8)
            
            all_deltas.append(delta_prices)
            
            # Update hidden state
            agent._kn2._h = h_new.numpy()
        
        if bar_pos in sampled_bars:
            action_head = out["action_logits"].argmax(dim=-1).item()
            mean_d = np.mean(delta_prices)
            std_d = np.std(delta_prices)
            pos_count = int(np.sum(delta_prices > 0))
            neg_count = int(np.sum(delta_prices < 0))
            consensus = f"{pos_count}/8P" if pos_count >= neg_count else f"{neg_count}/8N"
            
            print(f"  {bar_pos:>4d} {cp:>8.1f} {['H','L','S'][min(action_head,2)]:>8s} | "
                  f"{delta_prices[0]:>8.4f} {delta_prices[1]:>8.4f} {delta_prices[2]:>8.4f} {delta_prices[3]:>8.4f} "
                  f"{delta_prices[4]:>8.4f} {delta_prices[5]:>8.4f} {delta_prices[6]:>8.4f} {delta_prices[7]:>8.4f} | "
                  f"{mean_d:>8.4f} {std_d:>8.4f} {consensus:>8s}")
    
    # Summary statistics
    all_d = np.concatenate(all_deltas)
    print(f"\n  DELTA_PRICE STATS:")
    print(f"    count={len(all_d)}  mean={all_d.mean():.6f}  std={all_d.std():.6f}")
    print(f"    min={all_d.min():.6f}  max={all_d.max():.6f}")
    print(f"    P1={np.percentile(all_d, 1):.6f}  P99={np.percentile(all_d, 99):.6f}")
    print(f"    P10={np.percentile(all_d, 10):.6f}  P90={np.percentile(all_d, 90):.6f}")
    print(f"    frac>0={np.mean(all_d > 0)*100:.1f}%  frac<0={np.mean(all_d < 0)*100:.1f}%")
    
    # Reset for next day
    agent.reset_kn2_hidden()
