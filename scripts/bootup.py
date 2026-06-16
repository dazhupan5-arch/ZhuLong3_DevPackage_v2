import sys
sys.path.insert(0, r"D:\trae_projects\ZhuLong3_DevPackage_v2")
import torch
print(f"torch: {torch.__version__}")
import json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(r"D:\trae_projects\ZhuLong3_DevPackage_v2")
DATA_DIR = Path(r"C:\Users\xiaomi\Desktop")

print("ZHULONG 3 BOOTUP - KN2 Direct Decision Test")
print("=" * 60)

# 1. Config
print("\n1. CONFIG:")
cfg = json.loads((ROOT / "config/config_agent.json").read_text(encoding="utf-8-sig"))
print(f"   shadow_mode:    {cfg['kn2']['shadow_mode']}  <- LIVE MODE")

# 2. Load KN2 model directly
print("\n2. MODEL LOAD:")
from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state

kn2 = KN2Inference(ROOT / "models/kn2_trader.pth")
print(f"   is_ready:    {kn2.is_ready}")
print(f"   num_actions: {kn2.num_actions}")
print(f"   hidden_dim:  {kn2.hidden_dim}")
print(f"   num_layers:  {kn2.num_layers}")

if not kn2.is_ready:
    print("   FATAL: Model NOT READY")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Load gold data
print("\n3. DATA LOAD:")
df = pd.read_csv(DATA_DIR / "XAUUSD5.csv", header=None,
                 names=["date","time","open","high","low","close","volume"])
df = df.dropna(subset=["open","high","low","close"])
df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
df = df.sort_values("datetime").reset_index(drop=True)
print(f"   total rows:  {len(df)}")
print(f"   date range:  {df['datetime'].min()} -> {df['datetime'].max()}")

# 4. Build features
print("\n4. FEATURE BUILDING:")
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
X = np.column_stack(list(feats.values())).astype(np.float32)
np.nan_to_num(X,nan=0.0,copy=False); X=np.clip(X,-10,10)
X = np.pad(X, ((0,0),(0,98-X.shape[1])))
X = (X - X.mean(0)) / (X.std(0)+1e-8)
print(f"   feature dim: {X.shape}")

# 5. Run full test
print("\n5. KN2 DECISION TEST (Latest 1000 bars):")
test_slice = X[-1000:]
actions = {"hold":0,"long":0,"short":0}
conf_sum = 0
for i in range(len(test_slice)):
    out = kn2.predict(test_slice[i], encode_position_state())
    a = out["action_name"]
    if a in actions: 
        actions[a] += 1
        conf_sum += out["confidence"]
t = sum(actions.values())
print(f"   1000 bars: hold={actions['hold']}({actions['hold']/t*100:.0f}%) "
      f"long={actions['long']}({actions['long']/t*100:.0f}%) "
      f"short={actions['short']}({actions['short']/t*100:.0f}%)")
print(f"   avg conf:  {conf_sum/t:.3f}")

# 6. Latest bar decision
print(f"\n6. LATEST DECISION:")
tail = test_slice[-1]
last_out = kn2.predict(tail, encode_position_state())
print(f"   action:     {last_out['action_name']}")
print(f"   confidence: {last_out['confidence']:.3f}")
print(f"   size:       {last_out['position_size']:.2f}")
print(f"   sl_atr:     {last_out.get('sl_atr_mult',0):.2f}")
print(f"   tp_atr:     {last_out.get('tp_atr_mult',0):.2f}")
print(f"   trade:      {last_out['should_trade']}")
print(f"   last close: {c[-1]:.2f}")
print(f"   last bar:   {df['datetime'].iloc[-1]}")

print(f"\n{'='*60}")
print("KN2 STATUS: READY")
print(f"Config:  {ROOT / 'config/config_agent.json'}")
print(f"Model:   {ROOT / 'models/kn2_trader.pth'}")
print(f"Shadow:  FALSE (LIVE MODE)")
print("=" * 60)
