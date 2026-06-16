#!/usr/bin/env python3
"""
KN 2.0 Regime-Aware Training — fix the "never short" bias at its root.

Three changes vs train_kn2_final.py:
  1. Regime-aware label filtering: discard short labels when price is in a bull
     regime (above SMA48), and discard long labels in a bear regime (below SMA48).
     This prevents the model from learning "short = noise" during bull runs.
  2. Real struct_30 features: trend (SMA20 vs SMA48), ADX, vol_ratio,
     close-to-SMA distance, instead of 30 zeros.
  3. Same model architecture (128d GRU, 3 actions) — same features, cleaner labels.
"""
import sys, time, argparse, json
from pathlib import Path

# CRITICAL on Windows: import torch before numpy/pandas
import torch
import torch.nn as nn

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zhulong.agent.knowledge_net_kn2 import train_kn2_fast

# ── 配置 ──────────────────────────────────────────────────────
HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 128, 2, 64
EPOCHS, BATCH_SEQ, PATIENCE, SEQ_LEN = 40, 32, 15, 64
LR_BASE = 0.0005
CLASS_WEIGHTS = [1.0, 3.0, 3.0]  # push hard toward long/short
TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
TP_ATR, SL_ATR, MAX_HOLD = 2.0, 1.5, 48
REGIME_SMA = 200       # long-term trend SMA
REGIME_SHORT_SMA = 20  # short-term trend alignment
REGIME_THRESH = 0.02   # ±2% around SMA48 = sideways zone


def log(msg: str) -> None:
    print(msg, flush=True)


# ═════════════════════════════════════════════════════════════════
#  1. 数据加载
# ═════════════════════════════════════════════════════════════════
def prepare_data(csv_path: Path):
    log(f"  Reading {csv_path.name}...")
    df = pd.read_csv(csv_path, header=None,
                     names=["date", "time", "open", "high", "low", "close",
                            "tvol", "vol", "spread"])
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df = df.sort_values("datetime").reset_index(drop=True)

    from zhulong.strategies.indicators import atr_series
    atr_v = atr_series(pd.DataFrame({
        "high": df["high"], "low": df["low"], "close": df["close"]
    })).bfill().fillna(df["close"] * 0.001).values

    df["year"] = pd.to_datetime(df["datetime"]).dt.year
    df["atr"] = atr_v
    return df


# ═════════════════════════════════════════════════════════════════
#  2. Regime-Aware 标签生成
# ═════════════════════════════════════════════════════════════════
def generate_regime_labels(df: pd.DataFrame) -> dict:
    """Triple Barrier labels with light regime filtering.

    只过滤"高度逆势"信号：
      - 做空被过滤：价格在 SMA200 上方 2+ ATR → 典型的"牛市顶部追空"
      - 做多被过滤：价格在 SMA200 下方 2+ ATR → 典型的"熊市底部抄底"
      - 中间地带（SMA200±2ATR内）：保留所有标签，让模型自己学
    这样保留 ~70% 标签，避免模型全部预测 hold。
    """
    n = len(df)
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    av = np.maximum(df["atr"].values.astype(np.float64), c * 0.0005)

    # ── Regime: distance to SMA200 in ATR units ──
    cs = pd.Series(c)
    sma200 = cs.rolling(200, min_periods=50).mean().values
    d_sma200 = np.zeros(n)
    for t in range(50, n):
        if sma200[t] > 0 and av[t] > 0:
            d_sma200[t] = (c[t] - sma200[t]) / av[t]  # in ATR units

    deep_bull = d_sma200 > 2.0   # price > SMA200 + 2ATR → 强烈牛市
    deep_bear = d_sma200 < -2.0  # price < SMA200 - 2ATR → 强烈熊市

    n_bull = np.sum(deep_bull); n_bear = np.sum(deep_bear)
    log(f"  Deep bull (d>SMA200+2ATR): {n_bull:,} ({n_bull/n*100:.1f}%)")
    log(f"  Deep bear (d<SMA200-2ATR): {n_bear:,} ({n_bear/n*100:.1f}%)")
    log(f"  Transition zone (middle):   {n - n_bull - n_bear:,} ({100 - (n_bull+n_bear)/n*100:.1f}%)")

    # ── Triple Barrier ──
    ut = c + TP_ATR * av;  ls = c - SL_ATR * av
    lt = c - TP_ATR * av;  us = c + SL_ATR * av

    actions = np.zeros(n, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.float32)
    should_trade = np.zeros(n, dtype=np.float32)

    rejected_long = 0;  rejected_short = 0
    raw_long = 0;  raw_short = 0

    chunk = 5000
    log(f"  Labeling {n:,} bars (light regime filter)...")
    for start in range(0, n - MAX_HOLD, chunk):
        end = min(start + chunk, n - MAX_HOLD);  m = end - start
        hi = np.clip(np.arange(start, start+m)[:,None] + np.arange(1, MAX_HOLD+1), 0, n-1)
        li = np.clip(np.arange(start, start+m)[:,None] + np.arange(1, MAX_HOLD+1), 0, n-1)

        ltp = h[hi] >= ut[start:end, None];  lsl = l[li] <= ls[start:end, None]
        stp = l[li] <= lt[start:end, None];  ssl = h[hi] >= us[start:end, None]
        ltf = np.argmax(ltp, axis=1);  lsf = np.argmax(lsl, axis=1)
        stf = np.argmax(stp, axis=1);  ssf = np.argmax(ssl, axis=1)
        lta = ltp.any(1);  lsa = lsl.any(1)
        sta = stp.any(1);  ssa = ssl.any(1)

        for i in range(m):
            t = start + i

            if lta[i] and (not lsa[i] or ltf[i] < lsf[i]):
                raw_long += 1
                if not deep_bear[t]:  # 不要熊市底部抄底
                    actions[t] = 1;  sizes[t] = 1.0;  should_trade[t] = 1.0
                else:
                    rejected_long += 1

            if actions[t] == 0 and sta[i] and (not ssa[i] or stf[i] < ssf[i]):
                raw_short += 1
                if not deep_bull[t]:  # 不要牛市顶部追空
                    actions[t] = 2;  sizes[t] = 1.0;  should_trade[t] = 1.0
                else:
                    rejected_short += 1

        if (start // chunk) % 10 == 0:
            log(f"    {start:,}/{n - MAX_HOLD:,}")

    d = np.bincount(actions, minlength=3)
    log(f"  Labels: hold={d[0]:,} long={d[1]:,} short={d[2]:,} trade={(d[1]+d[2])/n*100:.1f}%")
    log(f"  Rejected: long={rejected_long:,} (of {raw_long:,}) short={rejected_short:,} (of {raw_short:,})")
    log(f"  Short/Long ratio: {d[2]/max(d[1], 1):.2f}")

    return {
        "action": actions, "position_size": sizes,
        "sl_atr_mult": np.full(n, SL_ATR, dtype=np.float32),
        "tp_atr_mult": np.full(n, TP_ATR, dtype=np.float32),
        "should_trade": should_trade,
    }


# ═════════════════════════════════════════════════════════════════
#  3. V14 特征
# ═════════════════════════════════════════════════════════════════
def build_v14(df: pd.DataFrame) -> np.ndarray:
    from zhulong.training.lgb.features import compute_features, FEATURE_COLUMNS_LGB_V13
    n = len(df);  feats = np.zeros((n, 68), dtype=np.float32);  cs = 20000
    log(f"  Computing V14 for {n:,} bars...")
    for s in range(0, n, cs):
        e = min(s + cs, n)
        chunk = df.iloc[:e].copy()
        try:
            fc = compute_features(chunk, include_mtf=True, include_reversal=True)
            cols = [c for c in FEATURE_COLUMNS_LGB_V13 if c in fc.columns]
            arr = fc[cols].iloc[s:e].to_numpy(dtype=np.float32)
            if arr.shape[1] < 68:
                arr = np.concatenate([arr, np.zeros((arr.shape[0], 68-arr.shape[1]), dtype=np.float32)], 1)
            feats[s:e, :] = arr[:, :68]
        except Exception:
            pass
        if (s // cs) % 5 == 0:
            log(f"    {min(e, n):,}/{n:,}")
    np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0, copy=False)
    return feats


# ═════════════════════════════════════════════════════════════════
#  4. 真实 Struct 特征（30 维）
# ═════════════════════════════════════════════════════════════════
def build_struct(df: pd.DataFrame, atr_v: np.ndarray) -> np.ndarray:
    """Fast struct features using rolling windows only.

    Row layout (30 dims):
      0   = trend           (sma20 - sma200) / sma200, clipped ±0.5
      1   = adx_approx      |trend| * 2, clipped [0, 1]
      2   = d_sma20         (close - sma20) / atr
      3   = d_sma200        (close - sma200) / atr  ← regime signal
      4   = sma20_slope     (sma20[t] - sma20[t-20]) / atr
      5   = vol_ratio       vol / vol_sma20
      6   = atr_ratio       atr / atr_sma20
      7..29 = 0.0 (reserved)
    """
    n = len(df)
    c = df["close"].values.astype(np.float64)
    v = df["vol"].values.astype(np.float64) if "vol" in df.columns else np.ones(n)
    av = np.maximum(atr_v, c * 0.0005)

    cs = pd.Series(c)
    sma20 = cs.rolling(20, min_periods=5).mean().bfill().values
    sma200 = cs.rolling(200, min_periods=50).mean().bfill().values
    vs = pd.Series(v)
    vsma20 = vs.rolling(20, min_periods=5).mean().bfill().values

    struct = np.zeros((n, 30), dtype=np.float32)

    # trend (sma20 - sma200): the core regime signal
    denom = np.maximum(np.abs(sma200), 1e-9)
    struct[:, 0] = np.clip((sma20 - sma200) / denom, -0.5, 0.5)

    # adx_approx
    struct[:, 1] = np.clip(np.abs(struct[:, 0]) * 3.0, 0.0, 1.0)

    # distance to SMAs
    struct[:, 2] = np.clip((c - sma20) / np.maximum(av, 1e-9), -5, 5)
    struct[:, 3] = np.clip((c - sma200) / np.maximum(av, 1e-9), -8, 8)

    # sma20 slope
    sma20_shifted = np.roll(sma20, 20);  sma20_shifted[:20] = sma20[:20]
    struct[:, 4] = np.clip((sma20 - sma20_shifted) / np.maximum(av, 1e-9), -5, 5)

    # vol_ratio
    struct[:, 5] = np.clip(v / np.maximum(vsma20, 1e-9), 0.1, 5.0)

    # atr_ratio
    atr_s = pd.Series(av)
    atr_sma20 = atr_s.rolling(20, min_periods=5).mean().bfill().values
    struct[:, 6] = np.clip(av / np.maximum(atr_sma20, 1e-9), 0.2, 3.0)

    np.nan_to_num(struct, nan=0.0, copy=False)
    return struct


# ═════════════════════════════════════════════════════════════════
#  5. 评估
# ═════════════════════════════════════════════════════════════════
def evaluate(model_path: Path, val_mf: np.ndarray, val_labels: dict, val_struct: np.ndarray):
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
    from sklearn.metrics import accuracy_score, confusion_matrix

    kn2 = KN2Inference(model_path)
    if not kn2.is_ready:
        return {"pass": False, "acc": 0, "pred_dist": np.array([0,0,0])}

    preds = [];  nv = min(len(val_mf), 5000)
    for i in range(nv):
        mf_full = np.concatenate([val_mf[i, :68], val_struct[i, :30]])
        d = kn2.predict(mf_full, encode_position_state())
        preds.append(d["action"])

    preds = np.array(preds);  labels = val_labels["action"][:nv]
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    pd_dist = np.bincount(preds, minlength=3) / nv

    precs = []
    for c in (0, 1, 2):
        tp = cm[c, c] if cm.shape[0] > c else 0
        fp = cm[:, c].sum() - tp if cm.shape[1] > c else 0
        precs.append(tp / max(tp + fp, 1))

    passed = acc > 0.50 and min(precs) > 0.30 and all(p > 0.10 for p in pd_dist)

    log(f"  Accuracy: {acc*100:.1f}%")
    log(f"  Precision: hold={precs[0]*100:.1f}% long={precs[1]*100:.1f}% short={precs[2]*100:.1f}%")
    log(f"  Pred dist: hold={pd_dist[0]*100:.1f}% long={pd_dist[1]*100:.1f}% short={pd_dist[2]*100:.1f}%")
    log(f"  PASS: {passed}")
    return {"pass": passed, "acc": acc, "precs": precs, "pred_dist": pd_dist}


# ═════════════════════════════════════════════════════════════════
#  6. Main
# ═════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()
    sym = "XAUUSD" if args.symbol.upper() in ("XAUUSD",) else "USOIL"
    out_path = ROOT / "models" / f"kn2_regime_{sym.lower()}.pth"
    csv_path = Path(r"C:\Users\xiaomi\Desktop\XAUUSD5.csv")

    log("=" * 65)
    log(f"KN 2.0 REGIME-AWARE TRAINING | {sym} | {HIDDEN_DIM}d GRU x{NUM_LAYERS}")
    log(f"  Regime: SMA{REGIME_SMA} ±{REGIME_THRESH*100:.0f}%")
    log(f"  Train: {TRAIN_YEARS} | Val: {VAL_YEAR}")
    log("=" * 65)

    t0 = time.perf_counter()

    log(f"\n[1/6] Loading {csv_path.name}...")
    df = prepare_data(csv_path)
    tr_mask = df["year"].isin(TRAIN_YEARS).values
    vl_mask = (df["year"] == VAL_YEAR).values
    tr_df = df[tr_mask].copy();  vl_df = df[vl_mask].copy()
    log(f"  Train: {len(tr_df):,} | Val: {len(vl_df):,}")

    log("\n[2/6] V14 features...");  t1 = time.perf_counter()
    v14_tr = build_v14(tr_df);  v14_vl = build_v14(vl_df)
    log(f"  Time: {time.perf_counter() - t1:.0f}s")

    log("\n[3/6] Struct features...");  t2 = time.perf_counter()
    struct_tr = build_struct(tr_df, tr_df["atr"].values)
    struct_vl = build_struct(vl_df, vl_df["atr"].values)
    log(f"  struct[trend] mean: {struct_tr[:, 0].mean():+.4f}")

    tr_mf = np.concatenate([v14_tr[:, :68], struct_tr[:, :30]], 1).astype(np.float32)
    vl_mf = np.concatenate([v14_vl[:, :68], struct_vl[:, :30]], 1).astype(np.float32)
    log(f"  Market: train={tr_mf.shape} val={vl_mf.shape} (98-dim)")
    log(f"  Time: {time.perf_counter() - t2:.0f}s")

    log("\n[4/6] Regime-aware labels...");  t3 = time.perf_counter()
    tr_l = generate_regime_labels(tr_df)
    vl_l = generate_regime_labels(vl_df)
    log(f"  Time: {time.perf_counter() - t3:.1f}s")

    log(f"\n[5/6] Training ({args.attempts} attempts max)...")
    pos_tr = np.zeros((len(tr_mf), 6), dtype=np.float32)
    best_result = None

    for at in range(1, args.attempts + 1):
        lr = LR_BASE * (0.8 + 0.4 * np.random.random())
        log(f"\n  Attempt {at}/{args.attempts} | LR={lr:.5f}")
        result = train_kn2_fast(
            market_features=tr_mf, position_states=pos_tr,
            targets={
                "action": tr_l["action"], "position_size": tr_l["position_size"],
                "sl_atr_mult": tr_l["sl_atr_mult"], "tp_atr_mult": tr_l["tp_atr_mult"],
                "should_trade": tr_l["should_trade"],
            },
            val_ratio=0.1, epochs=EPOCHS, batch_size=BATCH_SEQ, lr=lr,
            patience=PATIENCE, class_weights=CLASS_WEIGHTS,
            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
            num_actions=3, out_path=out_path, device="cpu", sequence_length=SEQ_LEN,
        )
        log(f"  val_loss={result['val_loss']:.4f}")

        log("\n[6/6] Evaluation...")
        ev = evaluate(out_path, vl_mf, vl_l, struct_vl)
        ev["val_loss"] = result["val_loss"]

        if ev["pass"]:
            log("\n*** TRAINING PASSED! ***");  best_result = ev;  break
        if best_result is None or ev["acc"] > best_result["acc"]:
            best_result = ev

    total = time.perf_counter() - t0
    log(f"\n{'=' * 65}")
    log(f"FINAL: {'PASSED' if best_result and best_result['pass'] else 'BEST EFFORT'}")
    log(f"  Model: {out_path} | Time: {total:.0f}s ({total / 60:.0f}m)")
    if best_result:
        log(f"  Accuracy: {best_result['acc']*100:.1f}%")
        log(f"  Short pred: {best_result['pred_dist'][2]*100:.1f}%")
    log("=" * 65)
    return 0 if (best_result and best_result["pass"]) else 1

if __name__ == "__main__":
    sys.exit(main())
