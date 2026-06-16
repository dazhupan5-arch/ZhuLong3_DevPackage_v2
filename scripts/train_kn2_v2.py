#!/usr/bin/env python3
"""KN 2.0 v2 training — pre-computed features, fast 128x2 GRU, class weights, 5 retries."""

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_NPZ = ROOT / "data" / "kn2_training_data.npz"
OUT_PATH = ROOT / "models" / "kn2_trader.pth"

HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 128, 2, 64
EPOCHS, BATCH_SEQ, PATIENCE, SEQ_LEN = 80, 20, 20, 64
LR_BASE = 0.0005
CLASS_WEIGHTS = [2.0, 1.0, 1.5, 1.0, 1.0, 1.0]
TRAIN_YEARS = [2021, 2022, 2023, 2024]
VAL_YEAR = 2025
TP, SL, H = 2.0, 1.5, 48
BARS_LIMIT = 250000

def log(msg): print(msg, flush=True)

def gen_labels(df, atr_v):
    n=len(df); c=df["close"].values.astype(np.float64)
    hi=df["high"].values.astype(np.float64); lo=df["low"].values.astype(np.float64)
    av=np.maximum(atr_v,c*0.0005); ut=c+TP*av; ls=c-SL*av; us=c+SL*av; lt=c-TP*av
    a=np.zeros(n,dtype=np.int32); sz=np.zeros(n,dtype=np.float32); st=np.zeros(n,dtype=np.float32)
    ch=5000
    log(f"  Labeling {n:,} bars...")
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
            if lta[i] and (not lsa[i] or ltf[i]<lsf[i]): a[t]=1; sz[t]=1.0; st[t]=1.0
            if a[t]==0 and sta[i] and (not ssa[i] or stf[i]<ssf[i]): a[t]=2; sz[t]=1.0; st[t]=1.0
        if (s//ch)%10==0: log(f"    {s:,}/{n-H:,}")
    d=np.bincount(a,minlength=6); log(f"  Done: {d} trade={st.mean()*100:.1f}%")
    return {"action":a,"position_size":sz,"sl_atr_mult":np.full(n,SL,dtype=np.float32),
            "tp_atr_mult":np.full(n,TP,dtype=np.float32),"should_trade":st}

def eval_model(path, vm, vl):
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
    from sklearn.metrics import accuracy_score, confusion_matrix
    k=KN2Inference(path)
    if not k.is_ready: return {"pass":False,"acc":0}
    p=np.array([k.predict(vm[i],encode_position_state())["action"] for i in range(len(vm))])
    ll=vl["action"][:len(p)]; acc=accuracy_score(np.clip(ll,0,2),np.clip(p,0,2))
    cm=confusion_matrix(ll,p,labels=list(range(6)))
    pr=[]; pd=np.bincount(p,minlength=6)/len(p)
    for c in range(3): tp=cm[c,c]; fp=cm[:,c].sum()-tp; pr.append(tp/max(tp+fp,1))
    ok=acc>0.50 and min(pr)>0.30 and all(p>0.08 for p in pd[:3])
    log(f"  Acc: {acc*100:.1f}% | Prec: h={pr[0]*100:.0f}% l={pr[1]*100:.0f}% s={pr[2]*100:.0f}%")
    log(f"  Dist: h={pd[0]*100:.0f}% l={pd[1]*100:.0f}% s={pd[2]*100:.0f}% | PASS: {ok}")
    return {"pass":ok,"acc":acc,"precs":pr,"cm":cm}

def main():
    log("="*60)
    log(f"KN2 v2 | {HIDDEN_DIM}x2 GRU | Train: {TRAIN_YEARS} | Val: {VAL_YEAR}")
    log(f"  Weights: {CLASS_WEIGHTS[:3]} | Pass: top3_acc>50% prec>30% dist>8%")
    log("="*60)

    t0=time.perf_counter()

    log("\n[1/5] Load data...")
    raw=np.load(DATA_NPZ,allow_pickle=True); dd={k:raw[k] for k in raw.files}
    mf=dd["market_feat"].astype(np.float32)
    df=pd.DataFrame({"close":dd["close"],"high":dd["high"],"low":dd["low"],
                     "atr":dd["atr"],"year":pd.to_datetime(dd["time"]).year})
    log(f"  Total: {len(df):,} | Features: {mf.shape}")

    tm=df["year"].isin(TRAIN_YEARS).values; vm=df["year"]==VAL_YEAR
    # Limit
    idx=np.where(tm)[0]
    if len(idx)>BARS_LIMIT:
        idx=idx[::max(len(idx)//BARS_LIMIT,1)][:BARS_LIMIT]
        tm=np.zeros(len(df),dtype=bool); tm[idx]=True
    log(f"  Train: {tm.sum():,} | Val: {vm.sum():,}")

    tdf=df[tm]; vdf=df[vm]; tmf=mf[tm]; vmf=mf[vm]

    log("\n[2/5] Labels..."); t2=time.perf_counter()
    tl=gen_labels(tdf,tdf["atr"].values); vl=gen_labels(vdf,vdf["atr"].values)
    log(f"  {time.perf_counter()-t2:.1f}s")

    log(f"\n[3/5] Training...")
    from zhulong.agent.knowledge_net_kn2 import train_kn2_fast
    pt=np.zeros((len(tmf),6),dtype=np.float32); best=None

    for at in range(1,6):
        lr=LR_BASE*(0.8+0.4*np.random.random())
        log(f"\n  Attempt {at}/5 | LR={lr:.5f}")
        r = train_kn2_fast(
            market_features=tmf,position_states=pt,
            targets={"action":tl["action"],"position_size":tl["position_size"],
                     "sl_atr_mult":tl["sl_atr_mult"],"tp_atr_mult":tl["tp_atr_mult"],
                     "should_trade":tl["should_trade"]},
            val_ratio=0.1,epochs=EPOCHS,batch_size=BATCH_SEQ,lr=lr,patience=PATIENCE,
            class_weights=CLASS_WEIGHTS,
            hidden_dim=HIDDEN_DIM,num_layers=NUM_LAYERS,embed_dim=EMBED_DIM,
            out_path=OUT_PATH,device="cpu",sequence_length=SEQ_LEN,
        )

        log(f"\n[4/5] Eval...")
        ev=eval_model(OUT_PATH,vmf,vl); ev["val_loss"]=r["val_loss"]
        if ev["pass"]: log("\n*** PASSED! ***"); best=ev; break
        if best is None or ev["acc"]>best["acc"]: best=ev; log(f"  Best: {ev['acc']*100:.1f}%")

    tt=time.perf_counter()-t0
    st = "PASSED" if (best and best["pass"]) else f"BEST({best['acc']*100:.1f}%)" if best else "FAIL"
    log(f"\n{'='*60}\n{st} in {tt:.0f}s | {OUT_PATH}\n{'='*60}")
    return 0 if (best and best["pass"]) else 1

if __name__=="__main__": sys.exit(main())
