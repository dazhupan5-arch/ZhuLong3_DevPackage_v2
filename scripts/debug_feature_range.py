"""
Debug: print raw V14 feature values + gate components
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

jun10 = df[df["datetime"].dt.strftime("%Y-%m-%d") == "2026-06-10"].copy()
day_idx = jun10.index.tolist()

print(f"{'Bar':>4s}  {'Time':>8s}  {'Close':>8s}  {'ATR':>6s}  {'feat[0]':>8s}  {'feat[1]':>8s}  {'feat_st':>8s}  {'trend_cl':>8s}  {'jsd':>7s}  {'gate_raw':>8s}  {'gate':>6s}  {'P-L':>6s}  {'P-S':>6s}  {'F-L':>6s}  {'F-S':>6s}")
print(f"{'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

for i, idx in enumerate(day_idx):
    if idx < WINDOW: continue
    bar_pos = idx - day_idx[0] - WINDOW
    if bar_pos < 0 or bar_pos > 30: continue

    seg = df.iloc[idx-WINDOW:idx].copy()
    cp = float(seg["close"].iloc[-1])
    dt = str(seg["datetime"].iloc[-1])
    m5t = m5(seg)
    atr_s = atr_series(m5t)
    atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else cp * 0.001

    if agent._kn2 is not None:
        kn_row = agent._build_knowledge_features(m5t, agent.structure.compute_latest({"M5": m5t}))
        struct_vals = agent.structure.compute_latest({"M5": m5t})
        struct_30 = np.asarray(struct_vals[:30], dtype=np.float32)
        v14_feat = np.asarray(kn_row, dtype=np.float32).reshape(-1)
        market_feat = np.concatenate([v14_feat[:68], struct_30[:30]]).reshape(1, -1)
        pos_state = encode_position_state()

        with torch.no_grad():
            mf_t = torch.tensor(market_feat)
            ps_t = torch.tensor(pos_state.reshape(1, -1))
            h_prev = torch.tensor(agent._kn2._h) if agent._kn2._h is not None else None
            outputs = agent._kn2.model(mf_t, h_prev, ps_t)
            h_new = outputs["hidden"]

            prior_logits = outputs["action_logits"].numpy()[0]
            fact_logits = agent._kn2._compute_fact_logits(market_feat)
            gate = agent._kn2._compute_gate(market_feat, prior_logits, fact_logits)

            # Manual JSD for display
            prior_p = np.exp(prior_logits - prior_logits.max()) / np.sum(np.exp(prior_logits - prior_logits.max()))
            fact_p = np.exp(fact_logits - fact_logits.max()) / np.sum(np.exp(fact_logits - fact_logits.max()))
            m_p = (prior_p + fact_p) / 2
            def kl(p,q): return float(np.sum(p*(np.log(p+1e-9)-np.log(q+1e-9))))
            jsd = (kl(prior_p,m_p)+kl(fact_p,m_p))/2

            trend = float(market_feat[0,0])
            trend_cl = min(abs(trend)*3.0, 1.0)
            gate_raw = 2.0*trend_cl + 1.5*jsd - 0.5

            if agent._kn2._h is not None:
                h_new = h_new * (5.0 / torch.norm(h_new).item()) if torch.norm(h_new).item() > 5.0 else h_new
            agent._kn2._h = h_new.numpy()

            # Stats on V14 features
            struct_f3 = struct_30[0] if len(struct_30) > 0 else 0

            print(f"{bar_pos:>4d}  {dt[-8:]:>8s}  {cp:>8.1f}  {atr:>6.2f}  "
                  f"{market_feat[0,0]:>+8.4f}  {market_feat[0,1]:>+8.4f}  {struct_f3:>+8.4f}  "
                  f"{trend_cl:>8.4f}  {jsd:>7.4f}  {gate_raw:>+8.4f}  {gate:.4f}  "
                  f"{prior_p[1]:>6.3f}  {prior_p[2]:>6.3f}  {fact_p[1]:>6.3f}  {fact_p[2]:>6.3f}")

# Summary stats
print(f"\n--- Raw feature ranges (first 30 bars) ---")
# Re-collect
vals = []
for idx in day_idx[:min(WINDOW+30, len(day_idx))]:
    if idx < WINDOW: continue
    seg = df.iloc[idx-WINDOW:idx].copy()
    m5t = m5(seg)
    kn_row = agent._build_knowledge_features(m5t, agent.structure.compute_latest({"M5": m5t}))
    struct_vals = agent.structure.compute_latest({"M5": m5t})
    struct_30 = np.asarray(struct_vals[:30], dtype=np.float32)
    v14_feat = np.asarray(kn_row, dtype=np.float32).reshape(-1)
    market_feat = np.concatenate([v14_feat[:68], struct_30[:30]])
    vals.append(market_feat)

all_feats = np.array(vals)
for i in range(min(5, all_feats.shape[1])):
    col = all_feats[:, i]
    print(f"  feat[{i}]: mean={col.mean():+.6f} std={col.std():.6f} min={col.min():+.6f} max={col.max():+.6f}")
