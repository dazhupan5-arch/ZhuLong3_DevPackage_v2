#!/usr/bin/env python3
"""KN 2.0 supervised pre-training — fast path with vectorized Triple Barrier."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_NPZ = ROOT / "data" / "kn2_training_data.npz"
OUT_PATH = ROOT / "models" / "kn2_trader.pth"

HIDDEN_DIM = 256
NUM_LAYERS = 2
EMBED_DIM = 64
EPOCHS = 150
LR = 0.0005
BATCH_SIZE = 128
PATIENCE = 25
SEQ_LEN = 64
TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.5
MAX_HOLD_BARS = 48


def print_header(msg: str) -> None:
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


def generate_labels_vectorized(df: pd.DataFrame) -> dict:
    """Vectorized Triple Barrier labeling using NumPy strides.

    Much faster than nested Python loops — handles 500k bars in seconds.
    """
    n = len(df)
    H = MAX_HOLD_BARS  # horizon
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    atr = df["atr"].values.astype(np.float64)

    actions = np.zeros(n, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.float32)
    sl_mults = np.full(n, SL_ATR_MULT, dtype=np.float32)
    tp_mults = np.full(n, TP_ATR_MULT, dtype=np.float32)
    should_trade = np.zeros(n, dtype=np.float32)

    # Pre-compute barriers for all bars
    upper_tp = close + TP_ATR_MULT * np.maximum(atr, close * 0.0005)
    lower_sl = close - SL_ATR_MULT * np.maximum(atr, close * 0.0005)
    upper_sl = close + SL_ATR_MULT * np.maximum(atr, close * 0.0005)
    lower_tp = close - TP_ATR_MULT * np.maximum(atr, close * 0.0005)

    # Use sliding window approach in chunks to avoid huge memory
    chunk = 10000
    for start in range(0, n - H, chunk):
        end = min(start + chunk, n - H)
        m = end - start

        # Build sliding window matrices for this chunk
        # high_window[t, f] = high[start+t+f]  for t in [0,m), f in [0,H)
        h_idx = np.arange(start, start + m)[:, None] + np.arange(1, H + 1)[None, :]
        l_idx = np.arange(start, start + m)[:, None] + np.arange(1, H + 1)[None, :]

        # Clip to valid range
        h_idx = np.clip(h_idx, 0, n - 1)
        l_idx = np.clip(l_idx, 0, n - 1)

        h_win = high[h_idx]  # (m, H)
        l_win = low[l_idx]   # (m, H)

        # Barriers for this chunk
        ut = upper_tp[start:end, None]   # (m, 1)
        ls = lower_sl[start:end, None]
        us = upper_sl[start:end, None]
        lt = lower_tp[start:end, None]

        # Long: TP first?  h_win >= ut
        long_tp_hit = h_win >= ut    # (m, H)
        long_sl_hit = l_win <= ls    # (m, H)

        # Short: TP first?  l_win <= lt
        short_tp_hit = l_win <= lt   # (m, H)
        short_sl_hit = h_win >= us   # (m, H)

        # Find first hit index for each (or H if no hit)
        long_tp_first = np.argmax(long_tp_hit, axis=1)  # (m,)
        long_sl_first = np.argmax(long_sl_hit, axis=1)
        short_tp_first = np.argmax(short_tp_hit, axis=1)
        short_sl_first = np.argmax(short_sl_hit, axis=1)

        # If no hit, argmax returns 0 — mask those out
        long_tp_any = long_tp_hit.any(axis=1)
        long_sl_any = long_sl_hit.any(axis=1)
        short_tp_any = short_tp_hit.any(axis=1)
        short_sl_any = short_sl_hit.any(axis=1)

        for i in range(m):
            t = start + i

            # Long direction
            lt_hit = long_tp_any[i]
            ls_hit = long_sl_any[i]
            if lt_hit and ls_hit:
                # Both hit — TP first?
                if long_tp_first[i] < long_sl_first[i]:
                    actions[t] = 1
                    sizes[t] = 1.0
                    should_trade[t] = 1.0
            elif lt_hit and not ls_hit:
                actions[t] = 1
                sizes[t] = 1.0
                should_trade[t] = 1.0

            # Short direction
            st_hit = short_tp_any[i]
            ss_hit = short_sl_any[i]
            if st_hit and ss_hit:
                if short_tp_first[i] < short_sl_first[i]:
                    if actions[t] == 0:
                        actions[t] = 2
                        sizes[t] = 1.0
                        should_trade[t] = 1.0
            elif st_hit and not ss_hit:
                if actions[t] == 0:
                    actions[t] = 2
                    sizes[t] = 1.0
                    should_trade[t] = 1.0

        if (start // chunk) % 10 == 0:
            pct = 100 * start / (n - H)
            print(f"  Labeling: {start:,}/{n-H:,} ({pct:.0f}%)")

    return {
        "action": actions,
        "position_size": sizes,
        "sl_atr_mult": sl_mults,
        "tp_atr_mult": tp_mults,
        "should_trade": should_trade,
    }


def main() -> int:
    print_header("KN 2.0 Supervised Pre-Training (vectorized)")
    print(f"  Output: {OUT_PATH}")
    print(f"  GRU: hidden={HIDDEN_DIM} layers={NUM_LAYERS} embed={EMBED_DIM}")
    print(f"  Epochs: {EPOCHS}, LR: {LR}")

    # ---- 1. Load ----
    print_header("[1/4] Loading data...")
    t0 = time.perf_counter()
    raw = np.load(DATA_NPZ, allow_pickle=True)
    dd = {k: raw[k] for k in raw.files}
    market_feat = dd["market_feat"].astype(np.float32)
    n_total = market_feat.shape[0]
    print(f"  Total: {n_total:,} bars, market_feat={market_feat.shape}")

    times = pd.to_datetime(dd["time"])
    df = pd.DataFrame({
        "open": dd["open"], "high": dd["high"], "low": dd["low"],
        "close": dd["close"], "atr": dd["atr"], "year": times.year,
    })

    train_mask = df["year"].isin(TRAIN_YEARS).values
    val_mask = (df["year"] == VAL_YEAR).values
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    train_market = market_feat[train_mask]
    val_market = market_feat[val_mask]
    print(f"  Train: {len(train_df):,} bars, Val: {len(val_df):,} bars")

    # ---- 2. Labels ----
    print_header("[2/4] Generating Triple Barrier labels (vectorized)...")
    t2 = time.perf_counter()
    train_labels = generate_labels_vectorized(train_df)
    dt = time.perf_counter() - t2
    print(f"  Train label dist: {np.bincount(train_labels['action'], minlength=6)}")
    print(f"  should_trade: {train_labels['should_trade'].mean()*100:.1f}%")
    print(f"  Time: {dt:.1f}s")
    val_labels = generate_labels_vectorized(val_df)
    print(f"  Val label dist:   {np.bincount(val_labels['action'], minlength=6)}")

    pos_train = np.zeros((len(train_market), 6), dtype=np.float32)

    # ---- 3. Train ----
    print_header("[3/4] Supervised GRU training...")
    t3 = time.perf_counter()

    from zhulong.agent.knowledge_net_kn2 import train_kn2_end_to_end

    result = train_kn2_end_to_end(
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
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        patience=PATIENCE,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
        out_path=OUT_PATH, device="cpu", sequence_length=SEQ_LEN,
    )
    t4 = time.perf_counter()
    print(f"  Train time: {t4-t3:.1f}s ({(t4-t3)/60:.1f}m)")
    print(f"  Best val loss: {result['val_loss']:.4f}")

    # ---- 4. Validate ----
    print_header("[4/4] Validation...")
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state

    kn2 = KN2Inference(OUT_PATH)
    if not kn2.is_ready:
        print("  ERROR: Model NOT ready!"); return 1

    print(f"  Model OK: hidden={kn2.hidden_dim}")
    correct = 0
    n_test = min(15, len(val_market))
    for i in range(n_test):
        dec = kn2.predict(val_market[i], encode_position_state())
        label = int(val_labels["action"][i])
        correct += 1 if dec["action"] == label else 0
        st = "OK" if dec["action"] == label else "  "
        print(f"    bar {i:4d}: pred={dec['action_name']:>9} label={label} "
              f"conf={dec['confidence']:.3f} {st}")
    print(f"  Accuracy: {correct}/{n_test} ({100*correct/n_test:.0f}%)")

    tt = time.perf_counter() - t0
    print_header(f"DONE in {tt:.0f}s ({tt/60:.1f}m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
