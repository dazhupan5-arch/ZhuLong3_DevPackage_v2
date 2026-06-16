#!/usr/bin/env python3
"""KN2 baseline: XGBoost & feature diagnostic.
   Determine the upper bound of 3-class action prediction."""

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_NPZ = ROOT / "data" / "kn2_training_data.npz"

OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_YEARS = [2021, 2022, 2023, 2024]
VAL_YEAR = 2025
TP, SL, H = 2.0, 1.5, 48

def log(msg): print(msg, flush=True)

def gen_labels(df):
    n=len(df); c=df["close"].values.astype(np.float64)
    hi=df["high"].values.astype(np.float64); lo=df["low"].values.astype(np.float64)
    av=np.maximum(df["atr"].values,c*0.0005); ut=c+TP*av; ls=c-SL*av
    us=c+SL*av; lt=c-TP*av
    a=np.zeros(n,dtype=np.int32); ch=5000
    for s in range(0,n-H,ch):
        e=min(s+ch,n-H); m=e-s
        hi_=np.clip(np.arange(s,s+m)[:,None]+np.arange(1,H+1),0,n-1)
        li_=np.clip(np.arange(s,s+m)[:,None]+np.arange(1,H+1),0,n-1)
        ltp=hi[hi_]>=ut[s:e,None]; lsl=lo[li_]<=ls[s:e,None]
        stp=lo[li_]<=lt[s:e,None]; ssl=hi[hi_]>=us[s:e,None]
        ltf=np.argmax(ltp,1); lsf=np.argmax(lsl,1); stf=np.argmax(stp,1); ssf=np.argmax(ssl,1)
        lta=ltp.any(1); lsa=lsl.any(1); sta=stp.any(1); ssa=ssl.any(1)
        for i in range(m):
            t=s+i
            if lta[i] and (not lsa[i] or ltf[i]<lsf[i]): a[t]=1
            if a[t]==0 and sta[i] and (not ssa[i] or stf[i]<ssf[i]): a[t]=2
    return a[:n]

log("="*60)
log("KN2 Baseline: XGBoost + Feature Diagnostic")
log("="*60)

t0 = time.perf_counter()
log("\nLoading data...")
raw = np.load(DATA_NPZ, allow_pickle=True); dd = {k:raw[k] for k in raw.files}
mf = dd["market_feat"].astype(np.float32)
df = pd.DataFrame({"close":dd["close"],"high":dd["high"],"low":dd["low"],
                   "atr":dd["atr"],"year":pd.to_datetime(dd["time"]).year})
log(f"  {len(df):,} bars, features: {mf.shape}")

# Feature quality
stds = np.std(mf, axis=0)
good = stds > 0.01
log(f"\nFeature dims with std>0.01: {good.sum()}/{len(stds)}")

# Use ALL features (XGBoost can handle noise)
log("\nGenerating labels...")
labels = gen_labels(df)
dist = np.bincount(labels, minlength=3)
log(f"  hold={dist[0]} long={dist[1]} short={dist[2]}")

tr_mask = df["year"].isin(TRAIN_YEARS).values
vl_mask = (df["year"]==VAL_YEAR).values

X_tr, Y_tr = mf[tr_mask], labels[tr_mask]
X_vl, Y_vl = mf[vl_mask], labels[vl_mask]
# Remove last H bars (no label)
X_tr, Y_tr = X_tr[:len(X_tr)-H], Y_tr[:len(X_tr)-H]
X_vl, Y_vl = X_vl[:len(X_vl)-H], Y_vl[:len(X_vl)-H]

log(f"  Train: {len(X_tr):,} | Val: {len(X_vl):,}")

# XGBoost with class weights
log("\nTraining XGBoost...")
import xgboost as xgb
dtrain = xgb.DMatrix(X_tr, label=Y_tr)
dval = xgb.DMatrix(X_vl, label=Y_vl)

# Scale weight inversely proportional to class frequency
n_total = len(Y_tr)
scale = n_total / 3
params = {
    "objective": "multi:softmax", "num_class": 3,
    "max_depth": 8, "eta": 0.1, "subsample": 0.8,
    "colsample_bytree": 0.8, "eval_metric": "mlogloss",
    "min_child_weight": 1, "nthread": 4,
    "scale_pos_weight": None,  # handled per-class
}

# Class weights
w = np.ones(len(Y_tr))
for c in range(3):
    cnt = np.sum(Y_tr==c)
    if cnt > 0:
        w[Y_tr==c] = scale / cnt
dtrain.set_weight(w)

model = xgb.train(params, dtrain, num_boost_round=200,
                  evals=[(dval, "val")], early_stopping_rounds=20, verbose_eval=10)

Yp = model.predict(dval).astype(int)
acc = accuracy_score(Y_vl, Yp)
cm = confusion_matrix(Y_vl, Yp)
log(f"\n=== XGBoost Results ===")
log(f"  Accuracy: {acc*100:.1f}%")
log(f"  Confusion Matrix:")
log(f"         hold  long short")
for i, nm in enumerate(["hold","long","short"]):
    log(f"    {nm:5s} {cm[i,0]:5d} {cm[i,1]:5d} {cm[i,2]:5d}")
pd = np.bincount(Yp, minlength=3) / len(Yp)
log(f"  Pred dist: hold={pd[0]*100:.1f}% long={pd[1]*100:.1f}% short={pd[2]*100:.1f}%")

# Feature importance
imp = model.get_score(importance_type="gain")
top = sorted(imp.items(), key=lambda x:x[1], reverse=True)[:10]
log(f"\n  Top 10 features (by gain):")
for k, v in top:
    log(f"    f{k}: {v:.1f}")

# Shuffle test: if features are useless, shuffled labels should give same accuracy
np.random.seed(42)
Y_shuf = Y_tr.copy(); np.random.shuffle(Y_shuf)
dtrain_shuf = xgb.DMatrix(X_tr, label=Y_shuf)
model_shuf = xgb.train(params, dtrain_shuf, num_boost_round=200,
                       evals=[(dval, "shuf_val")], early_stopping_rounds=20, verbose_eval=0)
Yp_shuf = model_shuf.predict(dval).astype(int)
acc_shuf = accuracy_score(Y_vl, Yp_shuf)
log(f"\n  Shuffle test (labels randomized): {acc_shuf*100:.1f}%")
log(f"  Signal/noise ratio: {acc/acc_shuf if acc_shuf>0 else 'inf':.2f}x")

# 3 separate binary classifiers
log(f"\n=== Per-class binary XGBoost ===")
for c, nm in enumerate(["hold","long","short"]):
    Yb_tr = (Y_tr==c).astype(int); Yb_vl = (Y_vl==c).astype(int)
    pos = Yb_tr.sum(); neg = len(Yb_tr)-pos
    dt = xgb.DMatrix(X_tr, label=Yb_tr)
    dv = xgb.DMatrix(X_vl, label=Yb_vl)
    scale_pos = neg / max(pos, 1)
    bp = {"objective":"binary:logistic","max_depth":6,"eta":0.1,
          "scale_pos_weight":scale_pos,"eval_metric":"auc","nthread":4}
    m = xgb.train(bp, dt, num_boost_round=100, verbose_eval=0)
    pred = (m.predict(dv) > 0.5).astype(int)
    acc_c = accuracy_score(Yb_vl, pred)
    recall = np.sum(pred * Yb_vl) / max(Yb_vl.sum(), 1)
    precision = np.sum(pred * Yb_vl) / max(pred.sum(), 1)
    log(f"  {nm}: acc={acc_c*100:.1f}% recall={recall*100:.1f}% prec={precision*100:.1f}% scale_pos={scale_pos:.1f}")

tt = time.perf_counter()-t0
log(f"\nDone in {tt:.0f}s")
