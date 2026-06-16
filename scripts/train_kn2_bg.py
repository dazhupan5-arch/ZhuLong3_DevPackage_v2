#!/usr/bin/env python3
"""KN 2.0 background continuous training — keeps improving the model."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_NPZ = ROOT / "data" / "kn2_training_data.npz"
OUT_PATH = ROOT / "models" / "kn2_trader.pth"

# Practical CPU-friendly settings
HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 128, 1, 32
EPOCHS, BATCH_SEQ, PATIENCE, SEQ_LEN = 100, 16, 20, 32
LR = 0.0005
TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
TP_ATR, SL_ATR, MAX_HOLD = 2.0, 1.5, 48
CHUNK = 5000


def log(msg): print(msg, flush=True)


def generate_labels(df: pd.DataFrame) -> dict:
    n = len(df); H = MAX_HOLD
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    atr_v = np.maximum(df["atr"].values.astype(np.float64), close * 0.0005)
    ut = close + TP_ATR * atr_v; ls = close - SL_ATR * atr_v
    us = close + SL_ATR * atr_v; lt = close - TP_ATR * atr_v

    actions = np.zeros(n, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.float32)
    should_trade = np.zeros(n, dtype=np.float32)

    log(f"  Labeling {n:,} bars...")
    for start in range(0, n - H, CHUNK):
        end = min(start + CHUNK, n - H); m = end - start
        hi = np.clip(np.arange(start, start+m)[:,None] + np.arange(1, H+1), 0, n-1)
        li = np.clip(np.arange(start, start+m)[:,None] + np.arange(1, H+1), 0, n-1)
        ltp = high[hi] >= ut[start:end,None]; lsl = low[li] <= ls[start:end,None]
        stp = low[li] <= lt[start:end,None]; ssl = high[hi] >= us[start:end,None]
        ltf = np.argmax(ltp, axis=1); lsf = np.argmax(lsl, axis=1)
        stf = np.argmax(stp, axis=1); ssf = np.argmax(ssl, axis=1)
        lta = ltp.any(1); lsa = lsl.any(1); sta = stp.any(1); ssa = ssl.any(1)
        for i in range(m):
            t = start + i
            if lta[i] and (not lsa[i] or ltf[i] < lsf[i]):
                actions[t] = 1; sizes[t] = 1.0; should_trade[t] = 1.0
            if actions[t] == 0 and sta[i] and (not ssa[i] or stf[i] < ssf[i]):
                actions[t] = 2; sizes[t] = 1.0; should_trade[t] = 1.0
        if (start // CHUNK) % 10 == 0: log(f"    {start:,}/{n-H:,}")
    return {"action": actions, "position_size": sizes,
            "sl_atr_mult": np.full(n, SL_ATR, dtype=np.float32),
            "tp_atr_mult": np.full(n, TP_ATR, dtype=np.float32),
            "should_trade": should_trade}


def main() -> int:
    log("=" * 60)
    log(f"KN 2.0 Background Training | hidden={HIDDEN_DIM} layers={NUM_LAYERS}")
    log(f"  Data: {TRAIN_YEARS} | Epochs: {EPOCHS} | Seq: {SEQ_LEN} Batch: {BATCH_SEQ}")
    log("=" * 60)

    t0 = time.perf_counter()
    raw = np.load(DATA_NPZ, allow_pickle=True); dd = {k: raw[k] for k in raw.files}
    mf = dd["market_feat"].astype(np.float32)
    df = pd.DataFrame({"close": dd["close"], "high": dd["high"], "low": dd["low"],
                       "atr": dd["atr"], "year": pd.to_datetime(dd["time"]).year})
    train_mask = df["year"].isin(TRAIN_YEARS).values
    val_mask = (df["year"] == VAL_YEAR).values
    log(f"  Train: {train_mask.sum():,}  Val: {val_mask.sum():,}")

    t2 = time.perf_counter()
    tr_l = generate_labels(df[train_mask])
    vl_l = generate_labels(df[val_mask])
    log(f"  Labels: {time.perf_counter()-t2:.1f}s")

    from zhulong.agent.knowledge_net_kn2 import train_kn2_fast
    result = train_kn2_fast(
        market_features=mf[train_mask], position_states=np.zeros((train_mask.sum(), 6), dtype=np.float32),
        targets={"action": tr_l["action"], "position_size": tr_l["position_size"],
                 "sl_atr_mult": tr_l["sl_atr_mult"], "tp_atr_mult": tr_l["tp_atr_mult"],
                 "should_trade": tr_l["should_trade"]},
        val_ratio=0.1, epochs=EPOCHS, batch_size=BATCH_SEQ, lr=LR, patience=PATIENCE,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
        out_path=OUT_PATH, device="cpu", sequence_length=SEQ_LEN,
    )

    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
    kn2 = KN2Inference(OUT_PATH)
    if kn2.is_ready:
        correct = 0; n_t = min(10, len(mf[val_mask]))
        for i in range(n_t):
            d = kn2.predict(mf[val_mask][i], encode_position_state())
            if d["action"] == vl_l["action"][i]: correct += 1
        log(f"  Val acc: {correct}/{n_t}  val_loss={result['val_loss']:.4f}")

    tt = time.perf_counter() - t0
    log(f"\nDONE in {tt:.0f}s ({tt/60:.0f}m)  Model: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
