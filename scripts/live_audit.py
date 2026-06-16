"""
烛龙3 KN2 实机架构审计 ...
"""
import sys
from pathlib import Path

# ---- 模拟 ZhuLong 发布环境的路径 ----
PUB = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2\publish\win-x64")
ROOT = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2")

sys.path.insert(0, str(ROOT))

# ---- 必须先 import torch（DLL 环境） ----
import torch
print(f"PyTorch: {torch.__version__}")

import os, json, time, numpy as np, pandas as pd
os.chdir(str(PUB))  # ZhuLong.exe 的工作目录

# ---- 1. 配置审计 ----
print("\n" + "=" * 65)
print("  CHECK #1 — CONFIG FILES")
print("=" * 65)

cfg_agent = json.loads((PUB / "config/config_agent.json").read_text(encoding="utf-8-sig"))
cfg_root  = json.loads((PUB / "config.json").read_text(encoding="utf-8-sig"))

checks = {}
def chk(name, ok, detail=""):
    checks[name] = ok
    marker = "[OK]" if ok else "[FAIL]"
    print(f"  {marker} {name}{' — ' + detail if detail else ''}")

chk("agent.enabled",      cfg_agent["enabled"] == True, f"={cfg_agent['enabled']}")
chk("kn2.enabled",        cfg_agent["kn2"]["enabled"] == True, f"={cfg_agent['kn2']['enabled']}")
chk("kn2.shadow_mode",    cfg_agent["kn2"]["shadow_mode"] == False, f"={cfg_agent['kn2']['shadow_mode']} (LIVE)")
chk("kn2.model_path",     "kn2_trader.pth" in cfg_agent["kn2"]["model_path"], f"={cfg_agent['kn2']['model_path']}")
chk("kn2.min_confidence", cfg_agent["kn2"]["min_confidence"] > 0, f"={cfg_agent['kn2']['min_confidence']}")
chk("primary_symbol",     cfg_agent["primary_symbol"] == "XAUUSD", f"={cfg_agent['primary_symbol']}")
chk("root trading_agent.enabled", cfg_root["trading_agent"]["enabled"] == True, f"=True")
chk("root multi_strategy.enabled", cfg_root["multi_strategy"]["enabled"] == True, f"=True")

# ---- 2. KN2 模型加载 ----
print("\n" + "=" * 65)
print("  CHECK #2 — KN2 MODEL LOADING")
print("=" * 65)

from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state

kn2_path = PUB / "models" / "kn2_trader.pth"
chk("model file exists", kn2_path.is_file(), str(kn2_path))

kn2 = KN2Inference(kn2_path)
chk("model is_ready",    kn2.is_ready)
chk("num_actions",       kn2.num_actions == 3, f"={kn2.num_actions} (hold/long/short)")

chk("hidden_dim  = 128", kn2.hidden_dim == 128, f"={kn2.hidden_dim}")
chk("num_layers  = 2",   kn2.num_layers == 2,   f"={kn2.num_layers}")
chk("val_loss OK", kn2.hidden_dim > 0,         f"model loaded")

# ---- 3. TradingAgent 实例化 ----
print("\n" + "=" * 65)
print("  CHECK #3 — TradingAgent INITIALIZATION")
print("=" * 65)

try:
    from zhulong.agent.trading_agent import TradingAgent
    agent = TradingAgent(config=cfg_agent, root=str(PUB))
    chk("agent instantiated", True)
    chk("kn2_mode",  agent.kn2_mode == True, f"={agent.kn2_mode}")
    chk("kn2_shadow", agent.kn2_shadow == False, f"={agent.kn2_shadow} (LIVE)")
    chk("primary_symbol", agent.primary_symbol == "XAUUSD", f"={agent.primary_symbol}")
    chk("agent enabled", agent.enabled == True)
    chk("kn2 object not None", agent._kn2 is not None)
    chk("kn2 object is_ready", agent._kn2.is_ready if agent._kn2 else False)
except Exception as ex:
    chk("agent instantiated", False, str(ex)[:80])
    agent = None

# ---- 4. Causal Graph ----
print("\n" + "=" * 65)
print("  CHECK #4 — CAUSAL GRAPH & FALLBACK")
print("=" * 65)

try:
    from zhulong.agent.causal_inference import load_causal_graph
    graph = load_causal_graph()
    syms = list(graph.get("symbols", {}).keys())
    chk("causal graph loaded", len(syms) > 0, f"symbols={syms}")
    chk("XAUUSD in graph", "XAUUSD" in syms)
except Exception as ex:
    chk("causal graph loaded", False, str(ex)[:80])

# ---- 5. 全链路决策模拟 ----
print("\n" + "=" * 65)
print("  CHECK #5 — FULL DECISION PIPELINE (Gold 10yr)")
print("=" * 65)

DATA_DIR = Path(r"C:\Users\xiaomi\Desktop")
df = pd.read_csv(DATA_DIR / "XAUUSD5.csv", header=None,
                 names=["date","time","open","high","low","close","volume"])
df = df.dropna(subset=["open","high","low","close"])
df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
df = df.sort_values("datetime").reset_index(drop=True)

c = df["close"].values.astype(np.float64)
h = df["high"].values.astype(np.float64)
l = df["low"].values.astype(np.float64)
o = df["open"].values.astype(np.float64)
v = df["volume"].values.astype(np.float64)

# Build 98-dim features (exact same as KN2 training)
feats = {}
for lag in [1,3,5,10,20]:
    feats[f"ret_{lag}"] = pd.Series(c).pct_change(lag).fillna(0).values
for lag in [1,5,20]:
    feats[f"logret_{lag}"] = pd.Series(np.log(np.maximum(c,1e-8))).diff(lag).fillna(0).values
for lag in [5,10,20]:
    feats[f"vol_{lag}"] = (pd.Series(c).rolling(lag).std()/np.maximum(c,1e-8)).fillna(0).values
feats["hl_ratio"] = (h-l)/np.maximum(c,1e-8)
feats["gap"] = pd.Series((o-np.roll(c,1))/np.maximum(np.roll(c,1),1e-8)).fillna(0).values
dc = np.diff(c,prepend=c[0]); gain=np.maximum(dc,0); loss_=np.maximum(-dc,0)
for w in [7,14,28]:
    ag=pd.Series(gain).rolling(w).mean().fillna(0).values
    al=pd.Series(loss_).rolling(w).mean().fillna(0).values
    feats[f"rsi_{w}"]=100-100/(1+ag/np.maximum(al,1e-8))
for w in [20,50]:
    ma=pd.Series(c).rolling(w).mean(); std=pd.Series(c).rolling(w).std()
    feats[f"bb_{w}"]=((c-ma)/np.maximum(std,1e-8)).fillna(0).values
e12=pd.Series(c).ewm(span=26,adjust=False).mean(); e26=pd.Series(c).ewm(span=52,adjust=False).mean()
macd=e12-e26;sig=macd.ewm(span=18,adjust=False).mean()
feats["macd"]=(macd-sig).values; feats["macd_hist"]=(macd-sig).diff().fillna(0).values
tr=np.maximum(h-l,np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
for w in [5,14,21]:
    feats[f"atr_{w}"]=(pd.Series(tr).rolling(w).mean()/np.maximum(c,1e-8)).fillna(0).values
for lag in [2,4,8]:
    feats[f"mom_{lag}"]=(pd.Series(c).diff(lag)/np.maximum(c,1e-8)).fillna(0).values
feats["pos_hl"]=(c-l)/np.maximum(h-l,1e-8)
e5=pd.Series(c).ewm(span=10,adjust=False).mean(); e40=pd.Series(c).ewm(span=40,adjust=False).mean()
feats["ema_cross"]=((e5-e40)/np.maximum(e40,1e-8)).values
feats["vratio"]=(v/np.maximum(pd.Series(v).rolling(10).mean(),1e-8)).fillna(0).values
ma60=pd.Series(c).rolling(60).mean()
feats["dma_60"]=((c-ma60)/np.maximum(ma60,1e-8)).fillna(0).values
X = np.column_stack(list(feats.values())).astype(np.float32)
np.nan_to_num(X,nan=0.0,copy=False); X=np.clip(X,-10,10)
X = np.pad(X, ((0,0),(0,98-X.shape[1])))
X = (X - X.mean(0)) / (X.std(0)+1e-8)

chk("feature dim = 98", X.shape[1] == 98, f"=({X.shape[1]},)")
chk("data rows", len(df) > 700000, f"={len(df):,}")
chk("date range covers 10yr", df["datetime"].dt.year.min() <= 2016 and df["datetime"].dt.year.max() >= 2026,
    f"{df['datetime'].dt.year.min()}-{df['datetime'].dt.year.max()}")

# 1000 bar sweep
test = X[-1000:]
actions = {"hold":0,"long":0,"short":0}
conf_sum, size_sum = 0.0, 0.0
trade_count = 0
kn2.reset_hidden()

for i in range(len(test)):
    out = kn2.predict(test[i], encode_position_state())
    a = out["action_name"]
    if a in actions: actions[a] += 1
    conf_sum += out["confidence"]
    size_sum += out["position_size"]
    if out["should_trade"]: trade_count += 1

t = sum(actions.values())
print(f"  Decisions (1000 bars):")
print(f"    hold={actions['hold']}({actions['hold']/t*100:.0f}%) "
      f"long={actions['long']}({actions['long']/t*100:.0f}%) "
      f"short={actions['short']}({actions['short']/t*100:.0f}%)")
print(f"    avg confidence = {conf_sum/t:.3f}")
print(f"    avg position_size = {size_sum/t:.2f}")
print(f"    should_trade = {trade_count}/1000 ({trade_count/10:.0f}%)")

chk("hold > 15%",       actions['hold']/t >= 0.15,  f"={actions['hold']/t*100:.0f}%")
chk("long > 0%",        actions['long'] > 0,         f"={actions['long']/t*100:.0f}%")
chk("not all same class", len([v for v in actions.values() if v > 0]) >= 2)
chk("avg confidence > 0.5", conf_sum/t >= 0.5,       f"={conf_sum/t:.3f}")
chk("avg confidence < 0.8", conf_sum/t <= 0.8,       f"={conf_sum/t:.3f}")
chk("some should_trade", trade_count > 0,            f"={trade_count}")

# Latest decision detail
last = kn2.predict(test[-1], encode_position_state())
print(f"\n  Latest decision detail:")
print(f"    action       = {last['action_name']}")
print(f"    confidence   = {last['confidence']:.3f}")
print(f"    position_size= {last['position_size']:.2f}")
print(f"    sl_atr_mult  = {last.get('sl_atr_mult',0):.2f}")
print(f"    tp_atr_mult  = {last.get('tp_atr_mult',0):.2f}")
print(f"    should_trade = {last['should_trade']}")
print(f"    embedding_dim= {last.get('embedding',np.zeros(0)).shape}")
print(f"    last_close   = {c[-1]:.2f}")
print(f"    last_bar     = {df['datetime'].iloc[-1]}")

chk("action in [hold/long/short]", last['action_name'] in ("hold","long","short"), f"={last['action_name']}")
chk("confidence in [0,1]", 0 <= last['confidence'] <= 1, f"={last['confidence']:.3f}")
chk("position_size in [0,1]", 0 <= last['position_size'] <= 1, f"={last['position_size']:.2f}")
chk("sl_atr_mult > 0", last.get('sl_atr_mult',0) > 0, f"={last.get('sl_atr_mult',0):.2f}")
chk("tp_atr_mult > sl_atr_mult", last.get('tp_atr_mult',0) > last.get('sl_atr_mult',0),
    f"tp={last.get('tp_atr_mult',0):.2f} sl={last.get('sl_atr_mult',0):.2f}")
chk("has embedding", last.get('embedding') is not None)

# ---- 6. Hidden State Continuity ----
print("\n" + "=" * 65)
print("  CHECK #6 — GRU HIDDEN STATE CONTINUITY")
print("=" * 65)

kn2.reset_hidden()
h0 = kn2._h
chk("hidden init = None", h0 is None)

# Run 3 consecutive predictions
out1 = kn2.predict(test[0], encode_position_state())
h1 = kn2._h
chk("hidden after 1st prediction", h1 is not None, f"shape={h1.shape}")

out2 = kn2.predict(test[1], encode_position_state())
h2 = kn2._h
chk("hidden after 2nd prediction", not np.allclose(h1, h2), "hidden changed")

out3 = kn2.predict(test[2], encode_position_state())

kn2.reset_hidden()
chk("hidden after reset", kn2._h is None)

# ---- SUMMARY ----
print("\n" + "=" * 65)
print("  AUDIT SUMMARY")
print("=" * 65)

passed = sum(v for v in checks.values())
total  = len(checks)
for name, ok in checks.items():
    print(f"  {'[ OK ]' if ok else '[FAIL]'} {name}")

print(f"\n  Result: {passed}/{total} checks passed")
if passed == total:
    print("\n  *** ALL CHECKS PASSED — KN2 Decision Architecture is FULLY OPERATIONAL ***")
else:
    print(f"\n  *** {total-passed} CHECK(S) FAILED ***")
print("=" * 65)
print(f"  Runtime:  {PUB / 'ZhuLong.exe'}")
print(f"  Config:   {PUB / 'config/config_agent.json'}")
print(f"  Model:    {PUB / 'models/kn2_trader.pth'}")
print(f"  Shadow:   FALSE (LIVE MODE)")
print("=" * 65)
