"""
Dual-Channel Architecture Validation v2
Tests KN2 with price-based fact signal on June 10 downtrend.
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

print(f"June 10: {len(jun10)} bars, {jun10['open'].iloc[0]:.1f} -> {jun10['close'].iloc[-1]:.1f}")
print(f"\n{'Bar':>4s}  {'Time':>8s}  {'Close':>8s}  {'ATR':>6s}  {'ret_atr':>8s}  {'trend_cl':>8s}  {'jsd':>7s}  {'gate':>6s}  {'prior':>6s}  {'fact':>6s}  {'ACT':>6s}  {'h_norm':>6s}")
print(f"{'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

actions = {"hold": 0, "long": 0, "short": 0}
gates = []
show_set = set(range(0, 40, 3))

bar_count = 0
for i, idx in enumerate(day_idx):
    if idx < WINDOW: continue
    bar_pos = idx - day_idx[0] - WINDOW
    if bar_pos < 0: continue
    bar_count += 1
    
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
        if struct_30.shape[0] < 30: struct_30 = np.pad(struct_30, (0, 30 - struct_30.shape[0]))
        v14_feat = np.asarray(kn_row, dtype=np.float32).reshape(-1)
        if v14_feat.shape[0] < 68: v14_feat = np.pad(v14_feat, (0, 68 - v14_feat.shape[0]))
        market_feat = np.concatenate([v14_feat[:68], struct_30[:30]])
        pos_state = encode_position_state()

        ret_atr = agent._kn2._update_return_in_atr(cp, atr)

        with torch.no_grad():
            mf_t = torch.tensor(market_feat.reshape(1, -1))
            ps_t = torch.tensor(pos_state.reshape(1, -1))
            h_prev = torch.tensor(agent._kn2._h) if agent._kn2._h is not None else None
            outputs = agent._kn2.model(mf_t, h_prev, ps_t)
            h_new = outputs["hidden"]

            prior_logits = outputs["action_logits"].numpy()[0]
            fact_logits = agent._kn2._compute_fact_logits_from_price(ret_atr)
            gate = agent._kn2._compute_gate_from_return(ret_atr, prior_logits, fact_logits)
            blended = (1 - gate) * prior_logits + gate * fact_logits
            action = int(np.argmax(blended))

            h_norm = torch.norm(h_new).item()
            if h_norm > agent._kn2._dc_h_norm_max:
                h_new = h_new * (agent._kn2._dc_h_norm_max / h_norm)
            agent._kn2._h = h_new.numpy()

        a_name = ["hold", "long", "short"][action]
        actions[a_name] += 1
        gates.append(gate)

        prior_a = ["H","L","S"][np.argmax(prior_logits)]
        fact_a = ["H","L","S"][np.argmax(fact_logits)]

        # Compute jsd for display
        prior_p = np.exp(prior_logits - prior_logits.max()) / np.sum(np.exp(prior_logits - prior_logits.max()))
        fact_p = np.exp(fact_logits - fact_logits.max()) / np.sum(np.exp(fact_logits - fact_logits.max()))
        m_p = (prior_p + fact_p) / 2
        def kl(p,q): return float(np.sum(p*(np.log(p+1e-9)-np.log(q+1e-9))))
        jsd_val = (kl(prior_p,m_p)+kl(fact_p,m_p))/2
        trend_cl = min(abs(ret_atr)*3.0, 1.0)

        if bar_pos in show_set:
            print(f"{bar_pos:>4d}  {dt[-8:]:>8s}  {cp:>8.1f}  {atr:>6.2f}  {ret_atr:>+8.3f}  {trend_cl:>8.4f}  {jsd_val:>7.4f}  {gate:.4f}  {prior_a:>6s}  {fact_a:>6s}  {a_name:>6s}  {h_norm:>6.2f}")

print(f"\n{'='*60}")
print(f"ACTION DISTRIBUTION ({bar_count} bars)")
print(f"{'='*60}")
for k, v in actions.items():
    print(f"  {k:>6s}: {v:>4d} ({v/bar_count*100:>5.1f}%)")
print(f"\nGate stats: min={min(gates):.4f} max={max(gates):.4f} mean={np.mean(gates):.4f}")
gate_open = sum(1 for g in gates if g > 0.5)
print(f"Gate > 0.5: {gate_open}/{len(gates)} ({gate_open/len(gates)*100:.1f}%)")
print(f"\n{'>>> KN2 IS NOW SHORTING!' if actions['short'] > 0 else '>>> KN2 still never shorts'}")
print()
