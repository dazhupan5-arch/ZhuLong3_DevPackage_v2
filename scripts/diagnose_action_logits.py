"""
Deep diagnostic: peek at raw action_logits inside the model
to understand WHY KN2 always outputs "long"
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
    return pd.DataFrame({
        "open": sl["open"].values, "high": sl["high"].values,
        "low": sl["low"].values, "close": sl["close"].values,
        "volume": sl["volume"].fillna(0).values
    }, index=idx)

jun10 = df[df["datetime"].dt.strftime("%Y-%m-%d") == "2026-06-10"].copy()
day_idx = jun10.index.tolist()

# Capture raw logits directly from the model
import torch.nn.functional as F
ACTION_NAMES = ["hold", "long", "short"]  # only 3 classes trained!

samples = []
sample_indices = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 220, 240, 260]

for idx in day_idx:
    if idx < WINDOW:
        continue
    
    bar_pos = idx - day_idx[0] - WINDOW
    if bar_pos < 0:
        continue
    
    seg = df.iloc[idx-WINDOW:idx].copy()
    cp = float(seg["close"].iloc[-1])
    dt = str(seg["datetime"].iloc[-1])
    m5t = m5(seg)
    atr_s = atr_series(m5t)
    atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else cp * 0.001

    if agent._kn2 is not None and agent._kn2.is_ready:
        kn_row = agent._build_knowledge_features(m5t, agent.structure.compute_latest({"M5": m5t}))
        struct_vals = agent.structure.compute_latest({"M5": m5t})
        struct_30 = np.asarray(struct_vals[:30], dtype=np.float32)
        if struct_30.shape[0] < 30:
            struct_30 = np.pad(struct_30, (0, 30 - struct_30.shape[0]))
        v14_feat = np.asarray(kn_row, dtype=np.float32).reshape(-1)
        if v14_feat.shape[0] < 68:
            v14_feat = np.pad(v14_feat, (0, 68 - v14_feat.shape[0]))
        market_feat = np.concatenate([v14_feat[:68], struct_30[:30]])
        pos_state = encode_position_state()

        # Get raw internal model output
        mf_t = torch.tensor(market_feat.reshape(1, -1))
        ps_t = torch.tensor(pos_state.reshape(1, -1))
        
        h_prev = torch.tensor(agent._kn2._h) if agent._kn2._h is not None else None
        
        with torch.no_grad():
            raw = agent._kn2.model(mf_t, h_prev, ps_t)
        
        logits = raw["action_logits"].numpy()[0]
        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        
        h_norm = np.linalg.norm(raw["hidden"].numpy())
        
        # Also track how hidden state evolves
        agent._kn2._h = raw["hidden"].numpy()  # update manually

        if bar_pos in sample_indices:
            samples.append({
                "bar": bar_pos,
                "time": dt[-8:],
                "close": cp,
                "atr": atr,
                "h_norm": h_norm,
                "logits": logits,
                "probs": probs,
            })

print(f"Raw action logits analysis on June 10")
print(f"Model hidden dim: {agent._kn2.hidden_dim}, initial hidden: {'zeros' if agent._kn2._h is None else f'norms={np.linalg.norm(agent._kn2._h):.2f}'}")
print()

print(f"{'Bar':>4s}  {'Time':>8s}  {'Close':>8s}  {'ATR':>6s}  {'h_norm':>7s}  {'hold':>6s}  {'long':>6s}  {'short':>6s}")
print(f"{'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}")

for s in samples:
    n = len(s['probs'])
    print(f"{s['bar']:>4d}  {s['time']:>8s}  {s['close']:>8.1f}  {s['atr']:>6.2f}  {s['h_norm']:>7.2f}  "
          f"{s['probs'][0]:>6.3f}  {s['probs'][1]:>6.3f}  {s['probs'][2]:>6.3f}")

# Summary
print(f"\n{'='*60}")
print(f"ANALYSIS")
print(f"{'='*60}")
probs_arr = np.array([s["probs"] for s in samples])
for i, name in enumerate(ACTION_NAMES):
    if i < probs_arr.shape[1]:
        print(f"  {name:>10s}: mean_p={probs_arr[:, i].mean():.4f}  min={probs_arr[:, i].min():.4f}  max={probs_arr[:, i].max():.4f}")

print(f"\nHidden state norms: min={min(s['h_norm'] for s in samples):.2f}  max={max(s['h_norm'] for s in samples):.2f}  final={samples[-1]['h_norm']:.2f}")
print(f"\nDiagnosis: The model's action logit for 'long' ALWAYS dominates.")
print(f"If long_prob >> short_prob even during a 4% crash, the model has:")
print(f"  (a) Training data heavily biased toward 'long' labels, OR")
print(f"  (b) The model learned 'gold always goes up' as its core prior, OR")
print(f"  (c) The GRU is dead (same output regardless of input)")

# Check if GRU is actually differentiating inputs
first_logits = np.array([samples[0]["logits"]])
last_logits = np.array([samples[-1]["logits"]])
logit_diff = np.abs(first_logits - last_logits)
print(f"\nLogit change from first to last bar (bar 0 vs {samples[-1]['bar']}):")
for i, name in enumerate(ACTION_NAMES):
    if i < logit_diff.shape[1]:
        print(f"  {name}: {samples[0]['logits'][i]:>7.4f} -> {samples[-1]['logits'][i]:>7.4f}  (delta={logit_diff[0][i]:.4f})")
