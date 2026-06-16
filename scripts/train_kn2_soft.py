#!/usr/bin/env python3
"""
KN2 Soft-target training: future returns -> soft action probabilities -> GRU.
Much richer signal than hard 3-class Triple Barrier labels.
"""

import sys, time, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
TP, SL, H, SEQ_LEN = 2.0, 1.5, 48, 64
EPOCHS, BATCH_SEQ, PATIENCE = 120, 20, 40
HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 128, 2, 64
NUM_ACTIONS, LR = 3, 0.0005
OUT_PATH = ROOT / "models" / "kn2_trader.pth"

def log(msg): print(msg, flush=True)

# ===== Features (same fast vectorized) =====
def build_features(df):
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64); l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64); v = df["volume"].values.astype(np.float64)
    feats = {}
    for lag in [1,3,5,10,20]:
        feats[f"ret_{lag}"] = pd.Series(c).pct_change(lag).fillna(0).values
    for lag in [1,5,20]:
        feats[f"logret_{lag}"] = pd.Series(np.log(c)).diff(lag).fillna(0).values
    for lag in [5,10,20]:
        feats[f"vol_{lag}"] = (pd.Series(c).rolling(lag).std()/c).fillna(0).values
    feats["hl_ratio"] = (h-l)/np.maximum(c,1e-8)
    feats["gap"] = pd.Series((o-np.roll(c,1))/np.roll(c,1)).fillna(0).values
    diff = np.diff(c,prepend=c[0]); g=np.maximum(diff,0); lo=np.maximum(-diff,0)
    for w in [7,14,28]:
        ag=pd.Series(g).rolling(w).mean().fillna(0).values
        al=pd.Series(lo).rolling(w).mean().fillna(0).values
        feats[f"rsi_{w}"]=100-100/(1+ag/np.maximum(al,1e-8))
    for w in [20,50]:
        ma=pd.Series(c).rolling(w).mean(); std=pd.Series(c).rolling(w).std()
        feats[f"bb_{w}"]=((c-ma)/np.maximum(std,1e-8)).fillna(0).values
    for w in [10,30]:
        x=np.arange(w); sx=x.sum(); sxx=(x*x).sum(); denom=w*sxx-sx*sx
        sxy=pd.Series(c).rolling(w).apply(lambda y:(x*y).sum(),raw=True)
        feats[f"slope_{w}"]=((w*sxy-sx*pd.Series(c).rolling(w).sum())/max(denom,1e-8)/np.maximum(c,1e-8)).fillna(0).values
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
    for w in [5,14,21]:
        feats[f"atr_{w}"]=(pd.Series(tr).rolling(w).mean()/c).fillna(0).values
    feats["vratio"]=(v/np.maximum(pd.Series(v).rolling(10).mean(),1e-8)).fillna(0).values
    feats["pos_hl"]=(c-l)/np.maximum(h-l,1e-8)
    e5=pd.Series(c).ewm(span=10,adjust=False).mean()
    e40=pd.Series(c).ewm(span=40,adjust=False).mean()
    feats["ema_cross"]=((e5-e40)/np.maximum(e40,1e-8)).values
    e12=pd.Series(c).ewm(span=26,adjust=False).mean()
    e26=pd.Series(c).ewm(span=52,adjust=False).mean()
    macd=e12-e26; sig=macd.ewm(span=18,adjust=False).mean()
    feats["macd"]=(macd-sig).values; feats["macd_hist"]=(macd-sig).diff().fillna(0).values
    for lag in [2,4,8]:
        feats[f"mom_{lag}"]=pd.Series(c).diff(lag).fillna(0).values/np.maximum(c,1e-8)
    for w in [10,30,60]:
        ma=pd.Series(c).rolling(w).mean()
        feats[f"dma_{w}"]=((c-ma)/np.maximum(ma,1e-8)).fillna(0).values
    result=np.column_stack(list(feats.values())).astype(np.float32)
    np.nan_to_num(result,nan=0.0,copy=False)
    return np.clip(result,-10,10), list(feats.keys())

# ===== Soft labels from future returns =====
def generate_soft_labels(df):
    """Future H-bar return -> soft action probabilities via temperature-scaled softmax."""
    n = len(df); c = df["close"].values.astype(np.float64)
    fwd_ret = np.zeros(n, dtype=np.float32)
    fwd_ret[:n-H] = (c[H:] - c[:-H]) / c[:-H]

    # Temperature-scaled: higher temp = more entropy (smoother)
    # typical_move = median absolute return over H bars
    typical = np.median(np.abs(fwd_ret[:n-H]))
    temperature = max(typical * 0.12, 0.00015)  # sharper: make soft targets more decisive

    # Raw scores: long_score = fwd_ret, short_score = -fwd_ret, hold_score = 0
    long_score = fwd_ret.copy()
    short_score = -fwd_ret.copy()
    hold_score = np.zeros(n, dtype=np.float32)

    # Softmax
    scores = np.column_stack([hold_score, long_score, short_score])
    # Stability: clip before exp
    scores = np.clip(scores / temperature, -20, 20)
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # Hard label: argmax of soft probs
    hard_action = np.argmax(probs, axis=1).astype(np.int32)

    return {
        "action": hard_action,
        "action_probs": probs.astype(np.float32),
        "fwd_ret": fwd_ret,
    }

# ===== Evaluation =====
def evaluate(path, vm, vl):
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
    from sklearn.metrics import accuracy_score, confusion_matrix
    k = KN2Inference(path)
    if not k.is_ready: return {"pass":False,"acc":0}
    p = np.array([k.predict(vm[i],encode_position_state())["action"] for i in range(len(vm))])
    ll = vl[:len(p)]; p = np.clip(p,0,2)
    acc = accuracy_score(ll,p)
    cm = confusion_matrix(ll,p,labels=[0,1,2])
    pr=[]; pd=np.bincount(p,minlength=3)/len(p)
    for c in range(3): tp=cm[c,c]; fp=cm[:,c].sum()-tp; pr.append(tp/max(tp+fp,1))
    ok = acc>0.50 and min(pr)>0.30 and all(x>0.10 for x in pd)
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
    log("KN2 Soft-target: future returns -> action probabilities -> GRU")
    log(f"  {HIDDEN_DIM}x{NUM_LAYERS} GRU | KL-div loss | Train:{TRAIN_YEARS} | Val:{VAL_YEAR}")
    log(f"  Pass: acc>50% prec>30% dist>10%")
    log("="*60)

    t0 = time.perf_counter()

    log("\n[1/5] Load + Features...")
    df = pd.read_csv(args.csv, header=None,
                     names=["date","time","open","high","low","close","tvol","volume","spread"])
    df = df.dropna(subset=["open","high","low","close"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
    df = df.sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year
    feats, names = build_features(df)
    log(f"  {len(df):,} bars | Features: {feats.shape}")

    log("\n[2/5] Soft labels from future returns...")
    labels = generate_soft_labels(df)
    d = np.bincount(labels["action"][:len(df)-H], minlength=3)
    log(f"  hold={d[0]} long={d[1]} short={d[2]}")
    # Show sample soft targets
    for i in [1000, 50000, 100000, 200000]:
        if i < len(labels["action_probs"]):
            log(f"  bar {i}: probs={labels['action_probs'][i].round(3)} fwd_ret={labels['fwd_ret'][i]:.4f}")

    tr_mask = df["year"].isin(TRAIN_YEARS).values
    vl_mask = (df["year"]==VAL_YEAR).values
    tr_feats = feats[tr_mask][:-H]
    vl_feats = feats[vl_mask][:-H]
    tr_l = {k:v[tr_mask][:-H] for k,v in labels.items()}
    vl_l = {k:v[vl_mask][:-H] for k,v in labels.items()}
    log(f"  Train: {len(tr_feats):,} | Val: {len(vl_feats):,}")

    # Pad to 98
    pad_tr = np.pad(tr_feats, ((0,0),(0,max(0,98-tr_feats.shape[1]))))[:,:98].astype(np.float32)
    pad_vl = np.pad(vl_feats, ((0,0),(0,max(0,98-vl_feats.shape[1]))))[:,:98].astype(np.float32)
    pos_tr = np.zeros((len(pad_tr),6), dtype=np.float32)

    log("\n[3/5] XGBoost baseline...")
    import xgboost as xgb
    dtrain = xgb.DMatrix(pad_tr, label=tr_l["action"])
    dval = xgb.DMatrix(pad_vl, label=vl_l["action"])
    params = {"objective":"multi:softmax","num_class":3,"max_depth":6,"eta":0.1,
              "eval_metric":"mlogloss","nthread":4}
    m = xgb.train(params, dtrain, num_boost_round=200,
                  evals=[(dval,"val")], early_stopping_rounds=20, verbose_eval=0)
    xp = m.predict(dval).astype(int); xa = np.mean(xp==vl_l["action"])*100
    xpd = np.bincount(xp,minlength=3)/len(xp)
    log(f"  XGBoost: {xa:.1f}% | dist: h={xpd[0]*100:.0f}% l={xpd[1]*100:.0f}% s={xpd[2]*100:.0f}%")

    log(f"\n[4/5] GRU training ({args.attempts} attempts)...")
    from zhulong.agent.knowledge_net_kn2 import train_kn2_fast
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best = None

    for at in range(1, args.attempts+1):
        lr = LR*(0.8+0.4*np.random.random())
        log(f"\n  Attempt {at}/{args.attempts} | LR={lr:.5f}")
        r = train_kn2_fast(
            market_features=pad_tr, position_states=pos_tr,
            targets={
                "action": tr_l["action"],
                "action_probs": tr_l["action_probs"],
                "position_size": np.ones(len(tr_l["action"]), dtype=np.float32),
                "sl_atr_mult": np.full(len(tr_l["action"]), SL, dtype=np.float32),
                "tp_atr_mult": np.full(len(tr_l["action"]), TP, dtype=np.float32),
                "should_trade": (tr_l["action"]>0).astype(np.float32),
            },
            val_ratio=0.1, epochs=EPOCHS, batch_size=BATCH_SEQ, lr=lr, patience=PATIENCE,
            class_weights=[1.0, 1.0, 1.0], num_actions=NUM_ACTIONS,
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
            out_path=OUT_PATH, device="cpu", sequence_length=SEQ_LEN,
        )

        log("\n[5/5] Evaluation...")
        ev = evaluate(OUT_PATH, pad_vl, vl_l["action"])
        ev["val_loss"] = r["val_loss"]
        if ev["pass"]: log("\n*** PASSED! ***"); best=ev; break
        if best is None or ev["acc"]>best["acc"]:
            best=ev; log(f"  New best: {ev['acc']*100:.1f}%")
        if at<args.attempts: log("  Retrying...")

    tt = time.perf_counter()-t0
    st = "PASSED" if (best and best["pass"]) else f"BEST({best['acc']*100:.1f}%)" if best else "FAIL"
    log(f"\n{'='*60}\n{st} | XGB={xa:.1f}% | {tt:.0f}s\n{'='*60}")
    return 0 if (best and best["pass"]) else 1

if __name__ == "__main__":
    sys.exit(main())
