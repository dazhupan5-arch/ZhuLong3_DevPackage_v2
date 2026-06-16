#!/usr/bin/env python3
"""
KN 2.0 Scenario Training — train scenario_head to predict future delta_price.
Uses the same data as train_kn2_final.py but adds scenario prediction loss.

The scenario_head learns to predict future price changes at 8 horizons,
giving KN2 a clean, unbiased directional signal separate from action_head.
"""
import sys, time, argparse, json
from pathlib import Path

# CRITICAL: Import torch DLLs BEFORE any other heavy library (pandas/numpy/sklearn)
# to avoid WinError 1114 DLL initialization conflicts on Windows.
import torch  # noqa: E402

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zhulong.agent.knowledge_net_kn2 import (
    train_kn2_fast, generate_scenario_labels, SCENARIO_HORIZONS, SCENARIO_PARAMS_PER,
)

HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 256, 2, 64
EPOCHS, BATCH_SEQ, PATIENCE, SEQ_LEN = 50, 16, 15, 64
LR_BASE = 0.0003
CLASS_WEIGHTS = [2.5, 1.0, 1.5, 1.0, 1.0, 1.0]
TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
TP_ATR, SL_ATR, MAX_HOLD = 2.0, 1.5, 48


def log(msg):
    print(msg, flush=True)


def prepare_data(csv_path):
    log(f"  Reading {csv_path.name}...")
    df = pd.read_csv(csv_path, header=None,
                     names=["date","time","open","high","low","close","tvol","vol","spread"])
    df = df.dropna(subset=["open","high","low","close"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df = df.sort_values("datetime").reset_index(drop=True)
    from zhulong.strategies.indicators import atr_series
    atr_v = atr_series(pd.DataFrame({
        "high": df["high"], "low": df["low"], "close": df["close"]
    })).bfill().fillna(df["close"] * 0.001).values
    return df, atr_v


def generate_labels(df, atr_v):
    n = len(df); H = MAX_HOLD
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    av = np.maximum(atr_v, c * 0.0005)
    ut = c + TP_ATR * av; ls = c - SL_ATR * av
    us = c + SL_ATR * av; lt = c - TP_ATR * av
    actions = np.zeros(n, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.float32)
    should_trade = np.zeros(n, dtype=np.float32)
    chunk = 5000
    log(f"  Labeling {n:,} bars...")
    for start in range(0, n - H, chunk):
        end = min(start + chunk, n - H); m = end - start
        hi = np.clip(np.arange(start, start+m)[:,None] + np.arange(1, H+1), 0, n-1)
        li = np.clip(np.arange(start, start+m)[:,None] + np.arange(1, H+1), 0, n-1)
        ltp = h[hi] >= ut[start:end, None]; lsl = l[li] <= ls[start:end, None]
        stp = l[li] <= lt[start:end, None]; ssl = h[hi] >= us[start:end, None]
        ltf = np.argmax(ltp, axis=1); lsf = np.argmax(lsl, axis=1)
        stf = np.argmax(stp, axis=1); ssf = np.argmax(ssl, axis=1)
        lta = ltp.any(1); lsa = lsl.any(1); sta = stp.any(1); ssa = ssl.any(1)
        for i in range(m):
            t = start + i
            if lta[i] and (not lsa[i] or ltf[i] < lsf[i]):
                actions[t] = 1; sizes[t] = 1.0; should_trade[t] = 1.0
            if actions[t] == 0 and sta[i] and (not ssa[i] or stf[i] < ssf[i]):
                actions[t] = 2; sizes[t] = 1.0; should_trade[t] = 1.0
        if (start // chunk) % 10 == 0: log(f"    {start:,}/{n-H:,}")
    d = np.bincount(actions, minlength=3)
    log(f"  Done: hold={d[0]} long={d[1]} short={d[2]} trade={should_trade.mean()*100:.1f}%")
    return {"action": actions, "position_size": sizes,
            "sl_atr_mult": np.full(n, SL_ATR, dtype=np.float32),
            "tp_atr_mult": np.full(n, TP_ATR, dtype=np.float32),
            "should_trade": should_trade}


def build_v14(df):
    from zhulong.training.lgb.features import compute_features, FEATURE_COLUMNS_LGB_V13
    n = len(df); feats = np.zeros((n, 68), dtype=np.float32); cs = 20000
    log(f"  Computing V14 for {n:,} bars...")
    for s in range(0, n, cs):
        e = min(s + cs, n)
        chunk = df.iloc[:e].copy()
        try:
            fc = compute_features(chunk, include_mtf=True, include_reversal=True)
            cols = [c for c in FEATURE_COLUMNS_LGB_V13 if c in fc.columns]
            arr = fc[cols].iloc[s:e].to_numpy(dtype=np.float32)
            if arr.shape[1] < 68:
                arr = np.concatenate([arr, np.zeros((arr.shape[0], 68 - arr.shape[1]), dtype=np.float32)], 1)
            feats[s:e, :] = arr[:, :68]
        except Exception as ex:
            log(f"  WARN: V14 chunk [{s}:{e}] {ex}")
        if (s // cs) % 5 == 0: log(f"    {min(e, n):,}/{n:,}")
    np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0, copy=False)
    return feats


def evaluate_scenario(model_path, val_mf, val_scn):
    """Check if scenario_head delta_price predictions correlate with actual future returns."""
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
    kn2 = KN2Inference(model_path)
    if not kn2.is_ready:
        return {"corr": 0, "sign_acc": 0}

    preds, truths = [], []
    n = len(val_mf[:2000])  # First 2000 for quick eval
    for i in range(n):
        d = kn2.predict(val_mf[i], encode_position_state(), close=0.0, atr=1.0)
        scn = d.get("scenarios")
        if scn is not None:
            # delta_price is at position 0, 8, 16, ... in the flattened array
            dp = scn.reshape(8, 8)[:, 0]  # 8 delta_prices
            preds.append(dp[0])  # Use shortest horizon (1 bar)
            truths.append(val_scn[i, 0])  # True delta_price at horizon 1

    preds = np.array(preds); truths = np.array(truths)
    corr = np.corrcoef(preds, truths)[0, 1] if len(preds) > 1 else 0
    sign_acc = np.mean(np.sign(preds) == np.sign(truths)) if len(preds) > 1 else 0

    log(f"  Scenario eval (n={len(preds)}): corr={corr:+.4f} sign_acc={sign_acc*100:.1f}%")
    return {"corr": corr, "sign_acc": sign_acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    sym = "XAUUSD" if args.symbol.upper() in ("XAUUSD",) else "USOIL"
    out_path = ROOT / "models" / f"kn2_scenario_{sym.lower()}.pth"
    csv_path = Path(r"C:\Users\xiaomi\Desktop\XAUUSD5.csv") if sym == "XAUUSD" \
        else Path(r"C:\Users\xiaomi\Desktop\XTIUSD5.csv")

    log("=" * 65)
    log(f"KN 2.0 SCENARIO TRAINING | {sym} | {HIDDEN_DIM}d GRU x{NUM_LAYERS}")
    log(f"  Horizons: {SCENARIO_HORIZONS}")
    log(f"  Train: {TRAIN_YEARS} | Val: {VAL_YEAR}")
    log("=" * 65)

    t0 = time.perf_counter()

    # Data
    log(f"\n[1/6] Loading {csv_path.name}...")
    df, atr_v = prepare_data(csv_path)
    df["atr"] = atr_v
    df["year"] = pd.to_datetime(df["datetime"]).dt.year
    tr_mask = df["year"].isin(TRAIN_YEARS).values
    vl_mask = (df["year"] == VAL_YEAR).values
    tr_df = df[tr_mask].copy(); vl_df = df[vl_mask].copy()
    log(f"  Train: {len(tr_df):,} | Val: {len(vl_df):,}")

    # V14 features
    log("\n[2/6] V14 features...")
    t1 = time.perf_counter()
    v14_tr = build_v14(tr_df); v14_vl = build_v14(vl_df)
    tr_mf = np.concatenate([v14_tr[:, :68], np.zeros((len(v14_tr), 30), dtype=np.float32)], 1).astype(np.float32)
    vl_mf = np.concatenate([v14_vl[:, :68], np.zeros((len(v14_vl), 30), dtype=np.float32)], 1).astype(np.float32)
    log(f"  Time: {time.perf_counter() - t1:.0f}s | Train: {tr_mf.shape} | Val: {vl_mf.shape}")

    # Action labels
    log("\n[3/6] Action labels..."); t2 = time.perf_counter()
    tr_l = generate_labels(tr_df, tr_df["atr"].values)
    vl_l = generate_labels(vl_df, vl_df["atr"].values)
    log(f"  Time: {time.perf_counter() - t2:.1f}s")

    # Scenario labels
    log("\n[4/6] Scenario labels..."); t3 = time.perf_counter()
    tr_scn = generate_scenario_labels(tr_df, tp_atr_mult=TP_ATR, sl_atr_mult=SL_ATR, max_hold_bars=MAX_HOLD)
    vl_scn = generate_scenario_labels(vl_df, tp_atr_mult=TP_ATR, sl_atr_mult=SL_ATR, max_hold_bars=MAX_HOLD)
    log(f"  Train shape: {tr_scn.shape} | Val shape: {vl_scn.shape}")
    log(f"  Time: {time.perf_counter() - t3:.1f}s")

    # Train
    log(f"\n[5/6] Training ({args.epochs} epochs)...")
    pos_tr = np.zeros((len(tr_mf), 6), dtype=np.float32)
    targets = {
        "action": tr_l["action"],
        "position_size": tr_l["position_size"],
        "sl_atr_mult": tr_l["sl_atr_mult"],
        "tp_atr_mult": tr_l["tp_atr_mult"],
        "should_trade": tr_l["should_trade"],
        "scenarios": tr_scn,  # <-- NEW: scenario prediction targets
    }

    result = train_kn2_fast(
        market_features=tr_mf, position_states=pos_tr, targets=targets,
        val_ratio=0.1, epochs=args.epochs, batch_size=BATCH_SEQ, lr=LR_BASE,
        patience=PATIENCE,
        class_weights=CLASS_WEIGHTS,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
        out_path=out_path, device="cpu", sequence_length=SEQ_LEN,
    )

    # Evaluate
    log(f"\n[6/6] Evaluation...")
    log(f"  val_loss={result['val_loss']:.4f}")
    ev = evaluate_scenario(out_path, vl_mf, vl_scn)
    total = time.perf_counter() - t0
    log(f"\n{'='*65}")
    log(f"DONE | Model: {out_path} | Time: {total:.0f}s ({total/60:.0f}m)")
    log(f"  Scenario corr={ev['corr']:+.4f} sign_acc={ev['sign_acc']*100:.1f}%")
    log(f"{'='*65}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
