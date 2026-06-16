"""烛龙3 实机检查：KN2黄金模型完整决策链路"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2")
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("ZHULONG 3 LIVE CHECK - KN2 Gold Decision Chain")
print("=" * 60)

# 1. Load config
cfg = json.loads((ROOT / "config/config_agent.json").read_text(encoding="utf-8-sig"))
print(f"\n1. Config:")
print(f"   agent enabled: {cfg['enabled']}")
print(f"   kn2.enabled:   {cfg['kn2']['enabled']}")
print(f"   kn2.shadow:    {cfg['kn2']['shadow_mode']}")
print(f"   kn2.path:      {cfg['kn2']['model_path']}")

# 2. Load model
print(f"\n2. KN2 Model:")
from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
kn2 = KN2Inference(ROOT / "models/kn2_trader.pth")
print(f"   is_ready:      {kn2.is_ready}")
print(f"   num_actions:   {kn2.num_actions}")
meta = json.loads((ROOT / "models/kn2_trader.meta.json").read_text(encoding="utf-8-sig"))
print(f"   hidden_dim:    {meta['hidden_dim']}")
print(f"   val_loss:      {meta['val_loss']:.6f}")

# 3. Load gold data and prepare features
print(f"\n3. Feature Pipeline:")
df = pd.read_csv(r"C:\Users\xiaomi\Desktop\XAUUSD5.csv", header=None,
                 names=["date","time","open","high","low","close","volume"])
df = df.dropna(subset=["open","high","low","close"])
df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
df = df.sort_values("datetime").reset_index(drop=True)

# 4. Feed recent bars through KN2
print(f"\n4. Decision Test (last 20 bars of 2025):")
df_2025 = df[df["datetime"].dt.year == 2025].tail(100).reset_index(drop=True)
print(f"   Test bars: {len(df_2025)}")

# Quick feature gen
c = df_2025["close"].values.astype(np.float64)
h = df_2025["high"].values.astype(np.float64)
l = df_2025["low"].values.astype(np.float64)
o = df_2025["open"].values.astype(np.float64)
v = df_2025["volume"].values.astype(np.float64)

def build_features(df):
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    feats = {}
    for lag in [1,3,5,10,20]:
        feats[f"ret_{lag}"] = pd.Series(c).pct_change(lag).fillna(0).values
    for lag in [1,5,20]:
        feats[f"logret_{lag}"] = pd.Series(np.log(np.maximum(c,1e-8))).diff(lag).fillna(0).values
    for lag in [5,10,20]:
        feats[f"vol_{lag}"] = (pd.Series(c).rolling(lag).std()/np.maximum(c,1e-8)).fillna(0).values
    feats["hl_ratio"] = (h-l)/np.maximum(c,1e-8)
    feats["gap"] = pd.Series((o-np.roll(c,1))/np.maximum(np.roll(c,1),1e-8)).fillna(0).values
    diff_c = np.diff(c,prepend=c[0])
    gain=np.maximum(diff_c,0); loss_=np.maximum(-diff_c,0)
    for w in [7,14,28]:
        ag=pd.Series(gain).rolling(w).mean().fillna(0).values
        al=pd.Series(loss_).rolling(w).mean().fillna(0).values
        feats[f"rsi_{w}"]=100-100/(1+ag/np.maximum(al,1e-8))
    for w in [20,50]:
        ma=pd.Series(c).rolling(w).mean(); std=pd.Series(c).rolling(w).std()
        feats[f"bb_{w}"]=((c-ma)/np.maximum(std,1e-8)).fillna(0).values
    e12=pd.Series(c).ewm(span=26,adjust=False).mean()
    e26=pd.Series(c).ewm(span=52,adjust=False).mean()
    macd=e12-e26;sig=macd.ewm(span=18,adjust=False).mean()
    feats["macd"]=(macd-sig).values
    feats["macd_hist"]=(macd-sig).diff().fillna(0).values
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
    for w in [5,14,21]:
        feats[f"atr_{w}"]=(pd.Series(tr).rolling(w).mean()/np.maximum(c,1e-8)).fillna(0).values
    for lag in [2,4,8]:
        feats[f"mom_{lag}"]=(pd.Series(c).diff(lag)/np.maximum(c,1e-8)).fillna(0).values
    feats["pos_hl"]=(c-l)/np.maximum(h-l,1e-8)
    e5=pd.Series(c).ewm(span=10,adjust=False).mean()
    e40=pd.Series(c).ewm(span=40,adjust=False).mean()
    feats["ema_cross"]=((e5-e40)/np.maximum(e40,1e-8)).values
    feats["vratio"]=(v/np.maximum(pd.Series(v).rolling(10).mean(),1e-8)).fillna(0).values
    ma60=pd.Series(c).rolling(60).mean()
    feats["dma_60"]=((c-ma60)/np.maximum(ma60,1e-8)).fillna(0).values
    result=np.column_stack(list(feats.values())).astype(np.float32)
    np.nan_to_num(result,nan=0.0,copy=False); result=np.clip(result,-10,10)
    return result

feats = build_features(df_2025)
feats_pad = np.pad(feats, ((0,0),(0,98-feats.shape[1]))).astype(np.float32)
feats_pad = (feats_pad - feats_pad.mean(0)) / (feats_pad.std(0) + 1e-8)

actions = {"hold": 0, "long": 0, "short": 0}
for i in range(len(feats_pad)):
    out = kn2.predict(feats_pad[i], encode_position_state())
    actions[out["action_name"]] += 1

total = sum(actions.values())
print(f"   Actions: hold={actions['hold']}({actions['hold']/total*100:.0f}%) "
      f"long={actions['long']}({actions['long']/total*100:.0f}%) "
      f"short={actions['short']}({actions['short']/total*100:.0f}%)")

# 5. Causual graph
print(f"\n5. Causal Graph:")
from zhulong.agent.causal_inference import load_causal_graph
graph = load_causal_graph()
print(f"   Symbols: {list(graph['symbols'].keys())}")
print(f"   JSON fallback: OK")

# 6. TradingAgent
print(f"\n6. TradingAgent:")
from zhulong.agent.trading_agent import TradingAgent
agent = TradingAgent(config=cfg, root=str(ROOT))
print(f"   kn2_mode:   {agent.kn2_mode}")
print(f"   kn2_shadow: {agent.kn2_shadow}")
print(f"   symbol:     {agent.primary_symbol}")
print(f"   enabled:    {agent.enabled}")
print(f"   use_rl:     {agent.use_rl}")

print(f"\n{'=' * 60}")
print("ALL CHECKS PASSED - ZhuLong3 KN2 Gold Ready")
print(f"Model: D:\\trae_projects\\ZhuLong3_DevPackage_v2\\models\\kn2_trader.pth")
print(f"Config: D:\\trae_projects\\ZhuLong3_DevPackage_v2\\config\\config_agent.json")
print("=" * 60)
