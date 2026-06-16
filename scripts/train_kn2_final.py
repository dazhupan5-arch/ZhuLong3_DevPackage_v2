#!/usr/bin/env python3
"""
KN 2.0 Final Training — class-balanced, deep GRU, full desktop data.
Gold: 738k bars   Oil: 734k bars

PASS CRITERIA (must meet ALL):
  - val_accuracy > 50%   (random baseline = 33%)
  - each class precision > 30%
  - each class predicted > 10%
"""

import sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 256, 2, 64
EPOCHS, BATCH_SEQ, PATIENCE, SEQ_LEN = 80, 16, 25, 64
LR_BASE = 0.0003
CLASS_WEIGHTS = [2.5, 1.0, 1.5, 1.0, 1.0, 1.0]
TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
TP_ATR, SL_ATR, MAX_HOLD = 2.0, 1.5, 48


def log(msg):
    print(msg, flush=True)


def prepare_data(csv_path):
    log(f"  Reading {csv_path.name}...")
    df = pd.read_csv(csv_path, header=None, names=["date","time","open","high","low","close","tvol","vol","spread"])
    df = df.dropna(subset=["open","high","low","close"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
    df = df.sort_values("datetime").reset_index(drop=True)
    from zhulong.strategies.indicators import atr_series
    atr_v = atr_series(pd.DataFrame({"high":df["high"],"low":df["low"],"close":df["close"]})).bfill().fillna(df["close"]*0.001).values
    return df, atr_v


def generate_labels(df, atr_v):
    n = len(df); H = MAX_HOLD
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    av = np.maximum(atr_v, c*0.0005)
    ut = c + TP_ATR*av; ls = c - SL_ATR*av
    us = c + SL_ATR*av; lt = c - TP_ATR*av
    actions = np.zeros(n, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.float32)
    should_trade = np.zeros(n, dtype=np.float32)
    chunk = 5000
    log(f"  Labeling {n:,} bars...")
    for start in range(0, n-H, chunk):
        end = min(start+chunk, n-H); m = end-start
        hi = np.clip(np.arange(start,start+m)[:,None]+np.arange(1,H+1),0,n-1)
        li = np.clip(np.arange(start,start+m)[:,None]+np.arange(1,H+1),0,n-1)
        ltp = h[hi] >= ut[start:end,None]; lsl = l[li] <= ls[start:end,None]
        stp = l[li] <= lt[start:end,None]; ssl = h[hi] >= us[start:end,None]
        ltf = np.argmax(ltp, axis=1); lsf = np.argmax(lsl, axis=1)
        stf = np.argmax(stp, axis=1); ssf = np.argmax(ssl, axis=1)
        lta = ltp.any(1); lsa = lsl.any(1); sta = stp.any(1); ssa = ssl.any(1)
        for i in range(m):
            t = start+i
            if lta[i] and (not lsa[i] or ltf[i]<lsf[i]):
                actions[t]=1; sizes[t]=1.0; should_trade[t]=1.0
            if actions[t]==0 and sta[i] and (not ssa[i] or stf[i]<ssf[i]):
                actions[t]=2; sizes[t]=1.0; should_trade[t]=1.0
        if (start//chunk)%10==0: log(f"    {start:,}/{n-H:,}")
    d = np.bincount(actions, minlength=3)
    log(f"  Done: hold={d[0]} long={d[1]} short={d[2]} trade={should_trade.mean()*100:.1f}%")
    return {"action":actions,"position_size":sizes,
            "sl_atr_mult":np.full(n,SL_ATR,dtype=np.float32),
            "tp_atr_mult":np.full(n,TP_ATR,dtype=np.float32),
            "should_trade":should_trade}


def build_v14(df):
    from zhulong.training.lgb.features import compute_features, FEATURE_COLUMNS_LGB_V13
    n = len(df); feats = np.zeros((n,68), dtype=np.float32); cs = 20000
    log(f"  Computing V14 for {n:,} bars...")
    for s in range(0,n,cs):
        e = min(s+cs,n)
        chunk = df.iloc[:e].copy()
        try:
            fc = compute_features(chunk, include_mtf=True, include_reversal=True)
            cols = [c for c in FEATURE_COLUMNS_LGB_V13 if c in fc.columns]
            arr = fc[cols].iloc[s:e].to_numpy(dtype=np.float32)
            if arr.shape[1]<68: arr=np.concatenate([arr,np.zeros((arr.shape[0],68-arr.shape[1]),dtype=np.float32)],1)
            feats[s:e,:]=arr[:,:68]
        except Exception as ex:
            log(f"  WARN: V14 chunk [{s}:{e}] {ex}")
        if (s//cs)%5==0: log(f"    {min(e,n):,}/{n:,}")
    np.nan_to_num(feats,nan=0.0,posinf=10.0,neginf=-10.0,copy=False)
    return feats


def evaluate(model_path, val_mf, val_labels):
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
    from sklearn.metrics import accuracy_score, confusion_matrix
    kn2 = KN2Inference(model_path)
    if not kn2.is_ready: return {"pass":False,"acc":0}
    preds=[]; nv=len(val_mf)
    for i in range(nv):
        d = kn2.predict(val_mf[i], encode_position_state())
        preds.append(d["action"])
    preds=np.array(preds); labels=val_labels["action"][:nv]
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0,1,2])
    precs=[]; pd= np.bincount(preds, minlength=3)/nv
    for c in [0,1,2]:
        tp=cm[c,c]; fp=cm[:,c].sum()-tp
        precs.append(tp/max(tp+fp,1))
    passed = acc>0.50 and min(precs)>0.30 and all(p>0.10 for p in pd)
    log(f"  Accuracy: {acc*100:.1f}%")
    log(f"  Precision: hold={precs[0]*100:.1f}% long={precs[1]*100:.1f}% short={precs[2]*100:.1f}%")
    log(f"  Pred dist: hold={pd[0]*100:.1f}% long={pd[1]*100:.1f}% short={pd[2]*100:.1f}%")
    log(f"  CM:        hold  long short")
    for i,nm in enumerate(["hold","long","short"]):
        log(f"    {nm:5s} {cm[i,0]:5d} {cm[i,1]:5d} {cm[i,2]:5d}")
    log(f"  PASS: {passed}")
    return {"pass":passed,"acc":acc,"precs":precs,"cm":cm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()
    sym = "XAUUSD" if args.symbol.upper() in ("XAUUSD",) else "USOIL"
    out_path = ROOT / "models" / f"kn2_trader_{sym.lower()}.pth"
    csv_path = Path(r"C:\Users\xiaomi\Desktop\XAUUSD5.csv") if sym=="XAUUSD" else Path(r"C:\Users\xiaomi\Desktop\XTIUSD5.csv")

    log("="*65)
    log(f"KN 2.0 Final | {sym} | {HIDDEN_DIM}d GRU x{NUM_LAYERS} | {EPOCHS} epochs")
    log(f"  Train: {TRAIN_YEARS} | Val: {VAL_YEAR} | Pass: acc>50%,prec>30%,balanced")
    log("="*65)

    t0 = time.perf_counter()

    # Data
    log(f"\n[1/5] Loading {csv_path.name}...")
    df, atr_v = prepare_data(csv_path)
    df["atr"] = atr_v
    df["year"] = pd.to_datetime(df["datetime"]).dt.year
    tr_mask = df["year"].isin(TRAIN_YEARS).values
    vl_mask = (df["year"]==VAL_YEAR).values
    tr_df = df[tr_mask].copy(); vl_df = df[vl_mask].copy()
    log(f"  Train: {len(tr_df):,} | Val: {len(vl_df):,}")

    # V14
    log("\n[2/5] V14 features...")
    t1 = time.perf_counter()
    v14_tr = build_v14(tr_df); v14_vl = build_v14(vl_df)
    tr_mf = np.concatenate([v14_tr[:,:68], np.zeros((len(v14_tr),30),dtype=np.float32)],1).astype(np.float32)
    vl_mf = np.concatenate([v14_vl[:,:68], np.zeros((len(v14_vl),30),dtype=np.float32)],1).astype(np.float32)
    log(f"  Time: {time.perf_counter()-t1:.0f}s | Train: {tr_mf.shape} | Val: {vl_mf.shape}")

    # Labels
    log("\n[3/5] Labels..."); t2=time.perf_counter()
    tr_l = generate_labels(tr_df, tr_df["atr"].values)
    vl_l = generate_labels(vl_df, vl_df["atr"].values)
    log(f"  Time: {time.perf_counter()-t2:.1f}s")

    # Train with retries
    log(f"\n[4/5] Training ({args.attempts} attempts max)...")
    from zhulong.agent.knowledge_net_kn2 import train_kn2_fast
    pos_tr = np.zeros((len(tr_mf),6), dtype=np.float32)
    best_pass = None
    for at in range(1, args.attempts+1):
        lr = LR_BASE * (0.8 + 0.4*np.random.random())
        log(f"\n  Attempt {at}/{args.attempts} | LR={lr:.5f}")
        result = train_kn2_fast(
            market_features=tr_mf, position_states=pos_tr,
            targets={"action":tr_l["action"],"position_size":tr_l["position_size"],
                     "sl_atr_mult":tr_l["sl_atr_mult"],"tp_atr_mult":tr_l["tp_atr_mult"],
                     "should_trade":tr_l["should_trade"]},
            val_ratio=0.1, epochs=EPOCHS, batch_size=BATCH_SEQ, lr=lr, patience=PATIENCE,
            class_weights=[2.5, 1.0, 1.5, 1.0, 1.0, 1.0],  # hold=2.5 long=1.0 short=1.5 s50/s100/close=1.0
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
            out_path=out_path, device="cpu", sequence_length=SEQ_LEN,
        )
        log(f"  val_loss={result['val_loss']:.4f}")

        log("\n[5/5] Evaluation...")
        ev = evaluate(out_path, vl_mf, vl_l)
        ev["val_loss"] = result["val_loss"]
        if ev["pass"]:
            log("\n*** TRAINING PASSED! ***"); best_pass = ev; break
        if best_pass is None or ev["acc"] > best_pass["acc"]:
            best_pass = ev

    total = time.perf_counter()-t0
    log(f"\n{'='*65}")
    log(f"FINAL: {'PASSED' if best_pass and best_pass['pass'] else 'BEST EFFORT'}")
    log(f"  Model: {out_path} | Time: {total:.0f}s ({total/60:.0f}m)")
    log(f"  Accuracy: {best_pass['acc']*100:.1f}% | val_loss: {best_pass.get('val_loss','N/A')}")
    log("="*65)
    return 0 if (best_pass and best_pass["pass"]) else 1

if __name__=="__main__":
    sys.exit(main())
