"""
Dual-Channel Architecture Validation
Tests if KN2 with fact/prior separation can take short positions
during June 10 downtrend.
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

# Verify dual_channel is in config
dc = (cfg.get("kn2") or {}).get("dual_channel", {})
print(f"Dual-channel config: {json.dumps(dc, indent=2)}")
print()

from zhulong.agent.trading_agent import TradingAgent
from zhulong.strategies.indicators import atr_series
from zhulong.agent.knowledge_net_kn2 import encode_position_state

agent = TradingAgent(config=cfg, root=str(INSTALL))
agent._breaker_enabled = False
agent.reset_kn2_hidden()

# Verify dual-channel is active
if agent._kn2 is not None:
    print(f"KN2 loaded: ready={agent._kn2.is_ready}")
    print(f"Dual-channel enabled: {agent._kn2._dc_enabled}")
    print(f"  gate_sensitivity={agent._kn2._dc_gate_sensitivity}")
    print(f"  gate_bias={agent._kn2._dc_gate_bias}")
    print(f"  h_norm_max={agent._kn2._dc_h_norm_max}")
else:
    print("KN2 NOT LOADED!")
    sys.exit(1)

WINDOW = 256

def m5(sl):
    idx = pd.DatetimeIndex(sl["datetime"], tz="UTC")
    return pd.DataFrame({
        "open": sl["open"].values, "high": sl["high"].values,
        "low": sl["low"].values, "close": sl["close"].values,
        "volume": sl["volume"].fillna(0).values
    }, index=idx)

# Focus on June 10 with detailed gate output
jun10 = df[df["datetime"].dt.strftime("%Y-%m-%d") == "2026-06-10"].copy()
day_idx = jun10.index.tolist()

print(f"\nJune 10 bars: {len(jun10)}")
print(f"Price: {jun10['open'].iloc[0]:.1f} -> {jun10['close'].iloc[-1]:.1f}")
print()

# Run with detailed per-bar diagnostic
print(f"{'Bar':>4s}  {'Time':>8s}  {'Close':>8s}  {'ATR':>6s}  {'Prior':>6s}  {'Fact':>6s}  {'Gate':>6s}  {'Action':>7s}  {'-Norm':>6s}")
print(f"{'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}")

actions = {"hold": 0, "long": 0, "short": 0}
gate_values = []
show_idx = list(range(0, 20, 2)) + list(range(50, 70, 2)) + list(range(100, 120, 2)) + list(range(180, 200, 2)) + list(range(250, 270, 1))
show_set = set(show_idx)

for i, idx in enumerate(day_idx):
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

        # Get raw model outputs + dual-channel blending
        with torch.no_grad():
            mf_t = torch.tensor(market_feat.reshape(1, -1))
            ps_t = torch.tensor(pos_state.reshape(1, -1))
            h_prev = torch.tensor(agent._kn2._h) if agent._kn2._h is not None else None
            outputs = agent._kn2.model(mf_t, h_prev, ps_t)
            h_new = outputs["hidden"]

            prior_logits = outputs["action_logits"].numpy()[0]
            fact_logits = agent._kn2._compute_fact_logits(market_feat.reshape(1, -1))
            gate = agent._kn2._compute_gate(market_feat.reshape(1, -1), prior_logits, fact_logits)
            blended_logits = (1 - gate) * prior_logits + gate * fact_logits
            action = int(np.argmax(blended_logits))

            # Apply h_norm clipping
            h_norm = torch.norm(h_new).item()
            if h_norm > agent._kn2._dc_h_norm_max:
                h_new = h_new * (agent._kn2._dc_h_norm_max / h_norm)
            agent._kn2._h = h_new.numpy()

        a_name = ["hold", "long", "short"][action]
        actions[a_name] += 1
        gate_values.append(gate)

        prior_a = ["H", "L", "S"][np.argmax(prior_logits)]
        fact_a = ["H", "L", "S"][np.argmax(fact_logits)]

        if bar_pos in show_set:
            print(f"{bar_pos:>4d}  {dt[-8:]:>8s}  {cp:>8.1f}  {atr:>6.2f}  {prior_a:>6s}  {fact_a:>6s}  {gate:.4f}  {a_name:>7s}  {h_norm:>6.2f}")

print(f"\n{'='*60}")
print(f"ACTION DISTRIBUTION")
print(f"{'='*60}")
total = sum(actions.values())
for k, v in actions.items():
    print(f"  {k:>6s}: {v:>4d} ({v/total*100:>5.1f}%)")
print(f"\nGate stats: min={min(gate_values):.4f} max={max(gate_values):.4f} mean={np.mean(gate_values):.4f} median={np.median(gate_values):.4f}")
gate_open = sum(1 for g in gate_values if g > 0.5)
print(f"Gate > 0.5 (trusting facts): {gate_open}/{len(gate_values)} ({gate_open/len(gate_values)*100:.1f}%)")
print(f"\nVERDICT: {'KN2 IS TAKING SHORT POSITIONS' if actions['short'] > 0 else 'KN2 still never shorts'}")
print()
