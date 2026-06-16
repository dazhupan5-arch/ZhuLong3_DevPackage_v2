"""
Diagnostic: Why doesn't KN2 short during June 10 downtrend?
Uses the agent directly to capture raw decision outputs.
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

# Focus on June 10
jun10 = df[df["datetime"].dt.strftime("%Y-%m-%d") == "2026-06-10"].copy()
day_idx = jun10.index.tolist()

raw_decisions = []
logit_samples = []

for idx in day_idx:
    if idx < WINDOW:
        continue
    seg = df.iloc[idx-WINDOW:idx].copy()
    cp = float(seg["close"].iloc[-1])
    dt = seg["datetime"].iloc[-1]
    m5t = m5(seg)
    atr_s = atr_series(m5t)
    atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else cp * 0.001

    # Get KN2 raw prediction
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

        kn2_raw = agent._kn2.predict(market_feat, pos_state)

        raw_decisions.append({
            "time": str(dt)[-8:],
            "close": cp,
            "atr": atr,
            "action": kn2_raw["action_name"],
            "action_id": kn2_raw["action"],
            "confidence": kn2_raw["confidence"],
            "should_trade": kn2_raw["should_trade"],
            "sl_atr_mult": kn2_raw["sl_atr_mult"],
            "tp_atr_mult": kn2_raw["tp_atr_mult"],
        })

rd = pd.DataFrame(raw_decisions)
print(f"Total decisions: {len(rd)}")
print(f"\nAction distribution (6 action classes):")
for a in ["hold", "long", "short", "short_50", "short_100", "close"]:
    c = sum(rd["action_id"].apply(lambda x: a == ["hold", "long", "short", "short_50", "short_100", "close"][int(x)]))
    pct = c/len(rd)*100
    print(f"  {a:>10s}: {c:>4d} ({pct:>5.1f}%)")

print(f"\nshould_trade:")
print(f"  True  : {sum(rd['should_trade'])} ({sum(rd['should_trade'])/len(rd)*100:.1f}%)")
print(f"  False : {sum(~rd['should_trade'])} ({sum(~rd['should_trade'])/len(rd)*100:.1f}%)")

print(f"\nConfidence:")
print(f"  min: {rd['confidence'].min():.4f}  max: {rd['confidence'].max():.4f}")
print(f"  mean: {rd['confidence'].mean():.4f}  median: {rd['confidence'].median():.4f}")

print(f"\nSL/TP:")
print(f"  SL atr_mult mean: {rd['sl_atr_mult'].mean():.2f} (range {rd['sl_atr_mult'].min():.1f}-{rd['sl_atr_mult'].max():.1f})")
print(f"  TP atr_mult mean: {rd['tp_atr_mult'].mean():.2f} (range {rd['tp_atr_mult'].min():.1f}-{rd['tp_atr_mult'].max():.1f})")

# Show decisions around price turning points
print(f"\nDecisions at key moments:")
print(f"  {'Time':>10s}  {'Close':>8s}  {'ATR':>7s}  {'Action':>10s}  {'Conf':>6s}  {'Trade':>6s}  {'SLx':>5s}")
print(f"  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*5}")
# Show first 10, middle 10, last 10
for i in list(range(0, 10)) + list(range(len(rd)//2-5, len(rd)//2+5)) + list(range(len(rd)-10, len(rd))):
    r = raw_decisions[i]
    print(f"  {r['time']:>10s}  {r['close']:>8.1f}  {r['atr']:>7.2f}  {r['action']:>10s}  {r['confidence']:.4f}  {str(r['should_trade']):>6s}  {r['sl_atr_mult']:>5.2f}")

# Now inspect training data
meta_path = INSTALL / "models" / "kn2_trader.meta.json"
meta = json.loads(meta_path.read_text())
print(f"\n{'='*60}")
print(f"TRAINING DATA ANALYSIS")
print(f"{'='*60}")
print(f"Model: {meta.get('model_type', 'unknown')}")
print(f"Hidden dim: {meta.get('hidden_dim', '?')}")
print(f"Training bars: {meta.get('total_bars', '?')}")
print(f"Date range: {meta.get('date_range', '?')}")

if "label_counts" in meta:
    lc = meta["label_counts"]
    total = sum(lc.values())
    print(f"\nTraining label distribution:")
    for k, v in lc.items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")
    if "short" not in str(lc).lower():
        # Might be numeric keys
        action_names = ["hold", "long", "short", "short_50", "short_100", "close"]
        for i, name in enumerate(action_names):
            if str(i) in lc:
                print(f"  [{i}] {name}: {lc[str(i)]} ({lc[str(i)]/total*100:.1f}%)")

print(f"\nOther meta keys: {list(meta.keys())}")
# Show any relevant training info
for k in ["train_loss", "val_accuracy", "best_val_acc", "epoch", "training_time"]:
    if k in meta:
        print(f"  {k}: {meta[k]}")

# Also check what heuristic fallback would do
print(f"\n{'='*60}")
print(f"HEURISTIC FALLBACK CHECK")
print(f"{'='*60}")
print(f"Heuristic threshold: trend > 0.3 -> long, < -0.3 -> short")
print(f"(This is what happens when KN2 model is not available)")
# Check first feature dimension trend on June 10
trends = []
for idx in day_idx:
    if idx < WINDOW:
        continue
    seg = df.iloc[idx-WINDOW:idx].copy()
    m5t = m5(seg)
    kn_row = agent._build_knowledge_features(m5t, agent.structure.compute_latest({"M5": m5t}))
    trends.append(("time", dt, kn_row[0, 0] if kn_row.ndim > 1 else kn_row[0]))
print(f"(Cannot recalculate - feature dim mismatch, but this is the V14 trend signal that would drive the heuristic)")

# GRU hidden state analysis
print(f"\n{'='*60}")
print(f"GRU HIDDEN STATE")
print(f"{'='*60}")
print(f"Hidden state shape: {agent._kn2_hidden.shape if agent._kn2_hidden is not None else 'None'}")
if agent._kn2_hidden is not None:
    h = agent._kn2_hidden.numpy()
    print(f"Hidden norm: {np.linalg.norm(h):.4f}")
    print(f"Hidden mean: {h.mean():.4f}")
    print(f"Hidden max: {h.max():.4f}, min: {h.min():.4f}")
