#!/usr/bin/env python3
"""KN2 training — fast vectorized features from desktop CSV."""

import sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
TP, SL, H = 2.0, 1.5, 48
SEQ_LEN, EPOCHS, BATCH_SEQ, PATIENCE = 64, 80, 20, 30
HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 128, 2, 64
NUM_ACTIONS, LR = 3, 0.0003
OUT_PATH = ROOT / "models" / "kn2_trader.pth"

def log(msg): print(msg, flush=True)

def build_features_fast(df):
    """Vectorized feature computation — O(n) per indicator, no Python loops."""
    n = len(df)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    feats = {}

    # Price returns
    for lag in [1, 3, 5, 10, 20]:
        feats[f"ret_{lag}"] = pd.Series(c).pct_change(lag).fillna(0).values
    for lag in [1, 5, 20]:
        feats[f"logret_{lag}"] = pd.Series(np.log(c)).diff(lag).fillna(0).values

    # Volatility
    for lag in [5, 10, 20]:
        feats[f"vol_{lag}"] = (pd.Series(c).rolling(lag).std() / c).fillna(0).values

    # HL ratio, gap
    feats["hl_ratio"] = (h - l) / np.maximum(c, 1e-8)
    feats["gap"] = pd.Series((o - np.roll(c, 1)) / np.roll(c, 1)).fillna(0).values

    # RSI
    diff = np.diff(c, prepend=c[0])
    gain = np.maximum(diff, 0); loss = np.maximum(-diff, 0)
    for w in [7, 14, 28]:
        ag = pd.Series(gain).rolling(w).mean().fillna(0).values
        al = pd.Series(loss).rolling(w).mean().fillna(0).values
        rs = ag / np.maximum(al, 1e-8)
        feats[f"rsi_{w}"] = 100 - 100 / (1 + rs)

    # Bollinger
    for w in [20, 50]:
        ma = pd.Series(c).rolling(w).mean()
        std = pd.Series(c).rolling(w).std()
        feats[f"bb_{w}"] = ((c - ma) / np.maximum(std, 1e-8)).fillna(0).values

    # Trend slope
    for w in [10, 30]:
        x = np.arange(w)
        sx = x.sum(); sxx = (x*x).sum()
        denom = w*sxx - sx*sx
        sxy = pd.Series(c).rolling(w).apply(lambda y: (x*y).sum(), raw=True)
        sy = pd.Series(c).rolling(w).sum()
        slope = (w*sxy - sx*sy) / max(denom, 1e-8)
        feats[f"slope_{w}"] = (slope / np.maximum(c, 1e-8)).fillna(0).values

    # ATR
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    for w in [5, 14, 21]:
        feats[f"atr_{w}"] = (pd.Series(tr).rolling(w).mean() / c).fillna(0).values

    # Volume ratio
    vma10 = pd.Series(v).rolling(10).mean()
    feats["vratio"] = (v / np.maximum(vma10, 1e-8)).fillna(0).values

    # Price position
    feats["pos_hl"] = (c - l) / np.maximum(h - l, 1e-8)

    # EMA cross
    e5 = pd.Series(c).ewm(span=10, adjust=False).mean()
    e20 = pd.Series(c).ewm(span=40, adjust=False).mean()
    feats["ema_cross"] = ((e5 - e20) / np.maximum(e20, 1e-8)).values

    # MACD
    e12 = pd.Series(c).ewm(span=26, adjust=False).mean()
    e26 = pd.Series(c).ewm(span=52, adjust=False).mean()
    macd = e12 - e26
    signal = macd.ewm(span=9*2, adjust=False).mean()
    feats["macd"] = (macd - signal).values
    feats["macd_hist"] = (macd - signal).diff().fillna(0).values

    # Momentum
    for lag in [2, 4, 8]:
        feats[f"mom_{lag}"] = pd.Series(c).diff(lag).fillna(0).values / np.maximum(c, 1e-8)

    # Distance from MA
    for w in [10, 30, 60]:
        ma = pd.Series(c).rolling(w).mean()
        feats[f"dma_{w}"] = ((c - ma) / np.maximum(ma, 1e-8)).fillna(0).values

    result = np.column_stack(list(feats.values())).astype(np.float32)
    np.nan_to_num(result, nan=0.0, copy=False)
    result = np.clip(result, -10, 10)
    return result, list(feats.keys())

def gen_labels(df):
    n=len(df); c=df["close"].values.astype(np.float64)
    hi=df["high"].values.astype(np.float64); lo=df["low"].values.astype(np.float64)
    tr = np.maximum(hi-lo, np.maximum(np.abs(hi-np.roll(c,1)), np.abs(lo-np.roll(c,1))))
    atr_v = pd.Series(tr).rolling(14).mean().fillna(tr[14] if n>14 else tr[-1]).values
    av=np.maximum(atr_v,c*0.0005); ut=c+TP*av; ls=c-SL*av; us=c+SL*av; lt=c-TP*av
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
    return a

def evaluate(path, vm, vl):
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
    from sklearn.metrics import accuracy_score, confusion_matrix
    k=KN2Inference(path)
    if not k.is_ready: return {"pass":False,"acc":0}
    p=np.array([k.predict(vm[i],encode_position_state())["action"] for i in range(len(vm))])
    ll=vl[:len(p)]; p=np.clip(p,0,2)
    acc=accuracy_score(ll,p); cm=confusion_matrix(ll,p,labels=[0,1,2])
    pr=[]; pd=np.bincount(p,minlength=3)/len(p)
    for c in range(3): tp=cm[c,c]; fp=cm[:,c].sum()-tp; pr.append(tp/max(tp+fp,1))
    ok=acc>0.50 and min(pr)>0.30 and all(p>0.10 for p in pd)
    log(f"  Acc: {acc*100:.1f}% | Prec: h={pr[0]*100:.0f}% l={pr[1]*100:.0f}% s={pr[2]*100:.0f}%")
    log(f"  Dist: h={pd[0]*100:.0f}% l={pd[1]*100:.0f}% s={pd[2]*100:.0f}% | PASS: {ok}")
    return {"pass":ok,"acc":acc,"precs":pr,"cm":cm}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=r"C:\Users\xiaomi\Desktop\XAUUSD5.csv")
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    log("="*60)
    log(f"KN2 Fast: {HIDDEN_DIM}x{NUM_LAYERS} GRU | Desktop CSV -> real features")
    log(f"  Train: {TRAIN_YEARS} | Val: {VAL_YEAR}")
    log("="*60)

    t0 = time.perf_counter()

    log("\n[1/5] Loading CSV...")
    df = pd.read_csv(args.csv, header=None,
                     names=["date","time","open","high","low","close","tvol","volume","spread"])
    df = df.dropna(subset=["open","high","low","close"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
    df = df.sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year
    log(f"  {len(df):,} bars")

    log("\n[2/5] Computing features (vectorized)...")
    t1 = time.perf_counter()
    feats, feat_names = build_features_fast(df)
    log(f"  Shape: {feats.shape} | Names: {feat_names[:5]}... ({time.perf_counter()-t1:.1f}s)")
    stds = np.std(feats, axis=0)
    log(f"  Dims with std>0.001: {np.sum(stds>0.001)}/{len(stds)}")

    log("\n[3/5] Labels...")
    t2 = time.perf_counter()
    labels = gen_labels(df)
    d = np.bincount(labels[:len(df)-H], minlength=3)
    log(f"  hold={d[0]} long={d[1]} short={d[2]} ({time.perf_counter()-t2:.1f}s)")

    tr_mask = df["year"].isin(TRAIN_YEARS).values
    vl_mask = (df["year"]==VAL_YEAR).values
    tr_mf = feats[tr_mask][:-H]; tr_l = labels[tr_mask][:-H]
    vl_mf = feats[vl_mask][:-H]; vl_l = labels[vl_mask][:-H]
    log(f"  Train: {len(tr_mf):,} | Val: {len(vl_mf):,}")

    # XGBoost baseline
    log("\n[4/5] XGBoost baseline...")
    import xgboost as xgb
    dtrain = xgb.DMatrix(tr_mf, label=tr_l)
    dval = xgb.DMatrix(vl_mf, label=vl_l)
    params = {"objective":"multi:softmax","num_class":3,"max_depth":6,"eta":0.1,
              "eval_metric":"mlogloss","nthread":4}
    m = xgb.train(params, dtrain, num_boost_round=200,
                  evals=[(dval,"val")], early_stopping_rounds=20, verbose_eval=0)
    xp = m.predict(dval).astype(int); xa = np.mean(xp==vl_l)*100
    log(f"  XGBoost: {xa:.1f}% | dist: {np.bincount(xp,minlength=3)/len(xp)*100}")

    # GRU training
    log(f"\n[5/5] GRU ({args.attempts} attempts)...")
    from zhulong.agent.knowledge_net_kn2 import train_kn2_fast
    pad_tr = np.pad(tr_mf, ((0,0),(0,max(0,98-tr_mf.shape[1]))))[:,:98].astype(np.float32)
    pad_vl = np.pad(vl_mf, ((0,0),(0,max(0,98-vl_mf.shape[1]))))[:,:98].astype(np.float32)
    pos_tr = np.zeros((len(pad_tr),6), dtype=np.float32); best = None

    for at in range(1, args.attempts+1):
        lr = LR*(0.8+0.4*np.random.random())
        log(f"\n  Attempt {at}/{args.attempts} | LR={lr:.5f}")
        r = train_kn2_fast(
            market_features=pad_tr, position_states=pos_tr,
            targets={"action":tr_l,"position_size":np.ones(len(tr_l),dtype=np.float32),
                     "sl_atr_mult":np.full(len(tr_l),SL,dtype=np.float32),
                     "tp_atr_mult":np.full(len(tr_l),TP,dtype=np.float32),
                     "should_trade":(tr_l>0).astype(np.float32)},
            val_ratio=0.1,epochs=EPOCHS,batch_size=BATCH_SEQ,lr=lr,patience=PATIENCE,
            class_weights=[1.0, 1.0, 1.0], num_actions=NUM_ACTIONS,
            hidden_dim=HIDDEN_DIM,num_layers=NUM_LAYERS,embed_dim=EMBED_DIM,
            out_path=OUT_PATH,device="cpu",sequence_length=SEQ_LEN,
        )
        ev = evaluate(OUT_PATH, pad_vl, vl_l); ev["val_loss"]=r["val_loss"]
        if ev["pass"]: log("\n*** PASSED! ***"); best=ev; break
        if best is None or ev["acc"]>best["acc"]: best=ev; log(f"  Best: {ev['acc']*100:.1f}%")

    tt=time.perf_counter()-t0
    st="PASSED" if (best and best["pass"]) else f"BEST({best['acc']*100:.1f}%)" if best else "FAIL"
    log(f"\n{'='*60}\n{st} | XGB={xa:.1f}% | {tt:.0f}s\n{'='*60}")
    return 0 if (best and best["pass"]) else 1

if __name__=="__main__": sys.exit(main())
