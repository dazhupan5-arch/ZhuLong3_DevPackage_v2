#!/usr/bin/env python3
"""KN2 training from raw desktop CSV — compute real features + XGBoost baseline + GRU training."""

import sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
TP, SL, H = 2.0, 1.5, 48
SEQ_LEN = 64
EPOCHS, BATCH_SEQ, PATIENCE = 80, 20, 30
HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 128, 2, 64
NUM_ACTIONS = 3
LR = 0.0003
OUT_PATH = ROOT / "models" / "kn2_trader.pth"

def log(msg): print(msg, flush=True)

# ===== Feature Engineering =====
def build_features(df):
    """Compute simple, meaningful features from OHLCV."""
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    n = len(df)

    feats = []
    # Price returns (multiple horizons)
    for lag in [1, 3, 5, 10, 20]:
        ret = np.zeros(n); ret[lag:] = (c[lag:] - c[:-lag]) / c[:-lag]; feats.append(ret)
    # Log returns
    for lag in [1, 5, 20]:
        lr = np.zeros(n); lr[lag:] = np.log(c[lag:]) - np.log(c[:-lag]); feats.append(lr)
    # Volatility
    for lag in [5, 10, 20]:
        vol = np.zeros(n)
        for i in range(lag, n): vol[i] = np.std(c[i-lag:i]) / c[i]
        feats.append(vol)
    # HL ratio
    hl = (h - l) / np.maximum(c, 1e-8); feats.append(hl)
    # Gap
    gap = np.zeros(n); gap[1:] = (o[1:] - c[:-1]) / c[:-1]; feats.append(gap)
    # RSI proxy
    for w in [7, 14, 28]:
        rsi = np.zeros(n)
        for i in range(w, n):
            up = np.sum(np.maximum(c[i-w:i] - np.roll(c,1)[i-w:i], 0))
            dn = np.sum(np.maximum(np.roll(c,1)[i-w:i] - c[i-w:i], 0))
            rsi[i] = 100 * up / max(up+dn, 1e-8)
        feats.append(rsi)
    # Bollinger position
    for w in [20, 50]:
        ma = np.zeros(n); std = np.zeros(n); bb = np.zeros(n)
        for i in range(w, n):
            ma[i] = np.mean(c[i-w:i]); std[i] = np.std(c[i-w:i])
        mask = std > 1e-8; bb[mask] = (c[mask] - ma[mask]) / std[mask]
        feats.append(bb)
    # Trend strength (linear slope)
    for w in [10, 30]:
        ts = np.zeros(n)
        for i in range(w, n):
            x = np.arange(w); y = c[i-w:i]
            slope = (w*np.sum(x*y) - np.sum(x)*np.sum(y)) / max(w*np.sum(x*x) - np.sum(x)**2, 1e-8)
            ts[i] = slope / max(c[i], 1e-8)
        feats.append(ts)
    # ATR proxy
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))
    for w in [5, 14, 21]:
        atr = np.zeros(n)
        for i in range(w, n): atr[i] = np.mean(tr[i-w:i]) / c[i]
        feats.append(atr)
    # Volume features
    v = df["volume"].values.astype(np.float64)
    vma = np.zeros(n)
    for i in range(10, n): vma[i] = np.mean(v[i-10:i])
    vratio = np.zeros(n); vratio[10:] = v[10:] / np.maximum(vma[10:], 1e-8)
    feats.append(vratio)
    # Price position (high-low range)
    pos_hl = np.zeros(n); pos_hl = (c - l) / np.maximum(h - l, 1e-8); feats.append(pos_hl)
    # EMA crosses
    ema5 = c.copy(); ema20 = c.copy()
    for i in range(1, n):
        ema5[i] = ema5[i-1]*0.8 + c[i]*0.2
        ema20[i] = ema20[i-1]*0.95 + c[i]*0.05
    ema_cross = (ema5 - ema20) / np.maximum(ema20, 1e-8); feats.append(ema_cross)

    result = np.column_stack(feats).astype(np.float32)
    # Clip outliers
    np.nan_to_num(result, nan=0.0)
    result = np.clip(result, -10, 10)
    return result

# ===== Labels =====
def gen_labels(df):
    n=len(df); c=df["close"].values.astype(np.float64)
    hi=df["high"].values.astype(np.float64); lo=df["low"].values.astype(np.float64)
    tr = np.maximum(hi-lo, np.maximum(np.abs(hi-np.roll(c,1)), np.abs(lo-np.roll(c,1))))
    atr_v = np.zeros(n); atr_v[14:]=np.convolve(tr, np.ones(14)/14, mode='same')[14:]
    atr_v[:14]=atr_v[14]; av=np.maximum(atr_v,c*0.0005)
    ut=c+TP*av; ls=c-SL*av; us=c+SL*av; lt=c-TP*av
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

def evaluate(model_path, vm, vl):
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
    from sklearn.metrics import accuracy_score, confusion_matrix
    k=KN2Inference(model_path)
    if not k.is_ready: return {"pass":False,"acc":0}
    p=np.array([k.predict(vm[i],encode_position_state())["action"] for i in range(len(vm))])
    ll=vl[:len(p)]; p=np.clip(p,0,2)
    acc=accuracy_score(ll,p); cm=confusion_matrix(ll,p,labels=[0,1,2])
    pr=[]; pd=np.bincount(p,minlength=3)/len(p)
    for c in range(3): tp=cm[c,c]; fp=cm[:,c].sum()-tp; pr.append(tp/max(tp+fp,1))
    ok=acc>0.50 and min(pr)>0.30 and all(p>0.10 for p in pd)
    log(f"  Acc: {acc*100:.1f}% | Prec: h={pr[0]*100:.0f}% l={pr[1]*100:.0f}% s={pr[2]*100:.0f}%")
    log(f"  Dist: h={pd[0]*100:.0f}% l={pd[1]*100:.0f}% s={pd[2]*100:.0f}% | PASS: {ok}")
    log(f"  CM:     hold  long short")
    for i,nm in enumerate(["hold","long","short"]):
        log(f"    {nm:5s} {cm[i,0]:5d} {cm[i,1]:5d} {cm[i,2]:5d}")
    return {"pass":ok,"acc":acc,"precs":pr,"cm":cm}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=r"C:\Users\xiaomi\Desktop\XAUUSD5.csv")
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()

    log("="*60)
    log(f"KN2 Final: Real Features from Desktop CSV")
    log(f"  Dims: {HIDDEN_DIM}x{NUM_LAYERS} | Feature dim: to be computed")
    log(f"  Train: {TRAIN_YEARS} | Val: {VAL_YEAR}")
    log("="*60)

    t0 = time.perf_counter()

    # Load CSV
    log("\n[1/6] Loading raw CSV...")
    df = pd.read_csv(args.csv, header=None,
                     names=["date","time","open","high","low","close","tvol","volume","spread"])
    df = df.dropna(subset=["open","high","low","close"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
    df = df.sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year
    log(f"  {len(df):,} bars")

    # Compute features
    log("\n[2/6] Computing features...")
    t1 = time.perf_counter()
    feats = build_features(df)
    log(f"  Feature shape: {feats.shape} ({time.perf_counter()-t1:.0f}s)")

    # Feature quality check
    stds = np.std(feats, axis=0)
    good = stds > 0.001
    log(f"  Dims with std>0.001: {good.sum()}/{len(stds)}")

    # Labels
    log("\n[3/6] Generating labels...")
    t2 = time.perf_counter()
    labels = gen_labels(df)
    dist = np.bincount(labels[:len(df)-H], minlength=3)
    log(f"  hold={dist[0]} long={dist[1]} short={dist[2]} ({time.perf_counter()-t2:.1f}s)")

    # Split
    log("\n[4/6] Splitting...")
    tr_mask = df["year"].isin(TRAIN_YEARS).values
    vl_mask = (df["year"]==VAL_YEAR).values
    tr_mf = feats[tr_mask]; vl_mf = feats[vl_mask]
    tr_l = labels[tr_mask]; vl_l = labels[vl_mask]
    # Trim H
    tr_mf = tr_mf[:-H]; tr_l = tr_l[:-H]
    vl_mf = vl_mf[:-H]; vl_l = vl_l[:-H]
    log(f"  Train: {len(tr_mf):,} | Val: {len(vl_mf):,}")

    # XGBoost baseline
    log("\n[5/6] XGBoost baseline...")
    import xgboost as xgb
    dtrain = xgb.DMatrix(tr_mf, label=tr_l)
    dval = xgb.DMatrix(vl_mf, label=vl_l)
    params = {"objective":"multi:softmax","num_class":3,"max_depth":6,"eta":0.1,
              "eval_metric":"mlogloss","nthread":4}
    xgb_model = xgb.train(params, dtrain, num_boost_round=200,
                          evals=[(dval,"val")], early_stopping_rounds=20, verbose_eval=0)
    xgb_pred = xgb_model.predict(dval).astype(int)
    xgb_acc = np.mean(xgb_pred==vl_l)*100
    xgb_cm = np.bincount(xgb_pred, minlength=3)/len(xgb_pred)
    log(f"  XGBoost accuracy: {xgb_acc:.1f}% | dist: {xgb_cm}")
    # Feature importance
    imp = xgb_model.get_score(importance_type="gain")
    top10 = sorted(imp.items(), key=lambda x:x[1], reverse=True)[:10]
    log(f"  Top features: {', '.join(f'f{k}={v:.0f}' for k,v in top10[:5])}")

    # GRU training
    log(f"\n[6/6] GRU training ({args.attempts} attempts)...")
    from zhulong.agent.knowledge_net_kn2 import train_kn2_fast
    # Pad features to 98 dims for GRU compatibility
    pad_tr = np.pad(tr_mf, ((0,0),(0,max(0,98-tr_mf.shape[1]))), mode='constant')[:,:98].astype(np.float32)
    pad_vl = np.pad(vl_mf, ((0,0),(0,max(0,98-vl_mf.shape[1]))), mode='constant')[:,:98].astype(np.float32)
    pos_tr = np.zeros((len(pad_tr),6), dtype=np.float32)
    best = None

    for at in range(1, args.attempts+1):
        lr = LR * (0.8+0.4*np.random.random())
        log(f"\n  Attempt {at}/{args.attempts} | LR={lr:.5f}")
        r = train_kn2_fast(
            market_features=pad_tr, position_states=pos_tr,
            targets={"action":tr_l,"position_size":np.ones(len(tr_l),dtype=np.float32),
                     "sl_atr_mult":np.full(len(tr_l),SL,dtype=np.float32),
                     "tp_atr_mult":np.full(len(tr_l),TP,dtype=np.float32),
                     "should_trade":(tr_l>0).astype(np.float32)},
            val_ratio=0.1,epochs=EPOCHS,batch_size=BATCH_SEQ,lr=lr,patience=PATIENCE,
            class_weights=[0.8, 1.5, 1.5],
            num_actions=NUM_ACTIONS,
            hidden_dim=HIDDEN_DIM,num_layers=NUM_LAYERS,embed_dim=EMBED_DIM,
            out_path=OUT_PATH,device="cpu",sequence_length=SEQ_LEN,
        )
        ev = evaluate(OUT_PATH, pad_vl, vl_l)
        ev["val_loss"] = r["val_loss"]
        if ev["pass"]: log("\n*** PASSED! ***"); best=ev; break
        if best is None or ev["acc"]>best["acc"]: best=ev; log(f"  Best: {ev['acc']*100:.1f}%")
        if at < args.attempts: log("  Retrying...")

    tt = time.perf_counter()-t0
    st = "PASSED" if (best and best["pass"]) else f"BEST({best['acc']*100:.1f}%)" if best else "FAIL"
    log(f"\n{'='*60}\n{st} in {tt:.0f}s | XGBoost baseline: {xgb_acc:.1f}%\n{'='*60}")
    return 0 if (best and best["pass"]) else 1

if __name__=="__main__": sys.exit(main())
