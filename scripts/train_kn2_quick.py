#!/usr/bin/env python3
"""KN 2.0 Quick supervised training — small dataset, flush output."""

import sys, os, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_NPZ = ROOT / "data" / "kn2_training_data.npz"
OUT_PATH = ROOT / "models" / "kn2_trader.pth"

HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 256, 2, 64
EPOCHS, BATCH_SEQ, PATIENCE, SEQ_LEN = 80, 32, 15, 64
LR = 0.0005
# Train on all available years, val on 2025
TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
TP_ATR, SL_ATR, MAX_HOLD = 2.0, 1.5, 48
CHUNK = 5000  # labeling chunk size


def log(msg: str) -> None:
    print(msg, flush=True)


def generate_labels_vectorized(df: pd.DataFrame) -> dict:
    """Vectorized Triple Barrier."""
    n = len(df)
    H = MAX_HOLD
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    atr_vals = df["atr"].values.astype(np.float64)
    min_atr = close * 0.0005
    eff_atr = np.maximum(atr_vals, min_atr)

    upper_tp = close + TP_ATR * eff_atr
    lower_sl = close - SL_ATR * eff_atr
    upper_sl = close + SL_ATR * eff_atr
    lower_tp = close - TP_ATR * eff_atr

    actions = np.zeros(n, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.float32)
    sl_mults = np.full(n, SL_ATR, dtype=np.float32)
    tp_mults = np.full(n, TP_ATR, dtype=np.float32)
    should_trade = np.zeros(n, dtype=np.float32)

    log(f"  Labeling {n:,} bars with horizon={H}...")

    for start in range(0, n - H, CHUNK):
        end = min(start + CHUNK, n - H)
        m = end - start

        # Window indices
        h_idx = np.arange(start, start + m)[:, None] + np.arange(1, H + 1)[None, :]
        l_idx = np.arange(start, start + m)[:, None] + np.arange(1, H + 1)[None, :]
        h_idx = np.clip(h_idx, 0, n - 1)
        l_idx = np.clip(l_idx, 0, n - 1)

        h_win = high[h_idx]   # (m, H)
        l_win = low[l_idx]    # (m, H)

        ut = upper_tp[start:end, None]
        ls = lower_sl[start:end, None]
        us = upper_sl[start:end, None]
        lt = lower_tp[start:end, None]

        long_tp = h_win >= ut
        long_sl = l_win <= ls
        short_tp = l_win <= lt
        short_sl = h_win >= us

        lt_first = np.argmax(long_tp, axis=1)
        ls_first = np.argmax(long_sl, axis=1)
        st_first = np.argmax(short_tp, axis=1)
        ss_first = np.argmax(short_sl, axis=1)

        lt_any = long_tp.any(axis=1)
        ls_any = long_sl.any(axis=1)
        st_any = short_tp.any(axis=1)
        ss_any = short_sl.any(axis=1)

        for i in range(m):
            t = start + i
            # Long
            lh, sh = lt_any[i], ls_any[i]
            if lh and sh:
                if lt_first[i] < ls_first[i]:
                    actions[t] = 1; sizes[t] = 1.0; should_trade[t] = 1.0
            elif lh:
                actions[t] = 1; sizes[t] = 1.0; should_trade[t] = 1.0
            # Short
            st, ss = st_any[i], ss_any[i]
            if st and ss:
                if st_first[i] < ss_first[i] and actions[t] == 0:
                    actions[t] = 2; sizes[t] = 1.0; should_trade[t] = 1.0
            elif st:
                if actions[t] == 0:
                    actions[t] = 2; sizes[t] = 1.0; should_trade[t] = 1.0

        if (start // CHUNK) % 5 == 0:
            log(f"    ... {start:,}/{n-H:,}")

    log(f"  Done: action dist={np.bincount(actions, minlength=6)}  trade={should_trade.mean()*100:.1f}%")
    return {"action": actions, "position_size": sizes,
            "sl_atr_mult": sl_mults, "tp_atr_mult": tp_mults,
            "should_trade": should_trade}


def main() -> int:
    log("=" * 60)
    log("KN 2.0 Supervised Pre-Training (small dataset)")
    log(f"  Train: {TRAIN_YEARS}, Val: {VAL_YEAR}, Epochs: {EPOCHS}")
    log("=" * 60)

    # ---- 1. Load ----
    log("\n[1/4] Loading data...")
    t0 = time.perf_counter()
    raw = np.load(DATA_NPZ, allow_pickle=True)
    dd = {k: raw[k] for k in raw.files}
    market_feat = dd["market_feat"].astype(np.float32)
    times = pd.to_datetime(dd["time"])
    df = pd.DataFrame({
        "open": dd["open"], "high": dd["high"], "low": dd["low"],
        "close": dd["close"], "atr": dd["atr"], "year": times.year,
    })
    log(f"  Loaded {market_feat.shape[0]:,} bars in {time.perf_counter()-t0:.1f}s")

    train_mask = df["year"].isin(TRAIN_YEARS).values
    val_mask = (df["year"] == VAL_YEAR).values
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    train_market = market_feat[train_mask]
    val_market = market_feat[val_mask]
    log(f"  Train: {len(train_df):,}  Val: {len(val_df):,}")

    # ---- 2. Labels ----
    log("\n[2/4] Generating labels...")
    t2 = time.perf_counter()
    train_labels = generate_labels_vectorized(train_df)
    t3 = time.perf_counter()
    log(f"  Train labels: {t3-t2:.1f}s")
    val_labels = generate_labels_vectorized(val_df)
    log(f"  Val labels: {time.perf_counter()-t3:.1f}s")

    pos_train = np.zeros((len(train_market), 6), dtype=np.float32)

    # ---- 3. Train ----
    log("\n[3/4] Training GRU network...")
    t3 = time.perf_counter()
    from zhulong.agent.knowledge_net_kn2 import train_kn2_fast

    result = train_kn2_fast(
        market_features=train_market,
        position_states=pos_train,
        targets={
            "action": train_labels["action"],
            "position_size": train_labels["position_size"],
            "sl_atr_mult": train_labels["sl_atr_mult"],
            "tp_atr_mult": train_labels["tp_atr_mult"],
            "should_trade": train_labels["should_trade"],
        },
        val_ratio=0.1,
        epochs=EPOCHS, batch_size=BATCH_SEQ, lr=LR,
        patience=PATIENCE,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
        out_path=OUT_PATH, device="cpu", sequence_length=SEQ_LEN,
    )
    log(f"  Training: {time.perf_counter()-t3:.1f}s  val_loss={result['val_loss']:.4f}")

    # ---- 4. Validate ----
    log("\n[4/4] Validation...")
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
    kn2 = KN2Inference(OUT_PATH)
    if not kn2.is_ready:
        log("  FAILED: model not ready")
        return 1
    log(f"  Model OK: hidden={kn2.hidden_dim}")
    correct = 0
    n_test = min(10, len(val_market))
    for i in range(n_test):
        dec = kn2.predict(val_market[i], encode_position_state())
        label = int(val_labels["action"][i])
        correct += 1 if dec["action"] == label else 0
        log(f"    bar {i}: pred={dec['action_name']:>9} label={label} conf={dec['confidence']:.3f}")
    log(f"  Accuracy: {correct}/{n_test}")
    log(f"\n{'='*60}\nDONE in {time.perf_counter()-t0:.0f}s\n{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
