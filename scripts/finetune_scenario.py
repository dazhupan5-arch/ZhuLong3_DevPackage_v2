#!/usr/bin/env python3
"""
KN 2.0 Scenario Head Fine-Tuning — freeze the existing GRU, train only scenario_head.
Fast path: precompute GRU hidden states, then train the head in isolation.
"""
import sys, time, argparse, json
from pathlib import Path

# CRITICAL: import torch BEFORE numpy/pandas to avoid WinError 1114
import torch
import torch.nn as nn

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zhulong.agent.knowledge_net_kn2 import (
    SCENARIO_HORIZONS, SCENARIO_PARAMS_PER, generate_scenario_labels,
    _build_trader_gru_class, _ensure_torch,
)

TRAIN_YEARS = [2022, 2023, 2024]
VAL_YEAR = 2025
TP_ATR, SL_ATR, MAX_HOLD = 2.0, 1.5, 48
HIDDEN_DIM, NUM_LAYERS, NUM_ACTIONS = 128, 2, 3
SEQ_LEN = 64


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
    df["atr"] = atr_v
    return df


def generate_labels(df):
    n = len(df); H = MAX_HOLD
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    av = df["atr"].values.astype(np.float64)
    av = np.maximum(av, c * 0.0005)
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
    return {"action": actions, "position_size": sizes,
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
        except Exception:
            pass
    np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0, copy=False)
    return feats


def precompute_hidden_states(model, mf, device):
    """Run the existing GRU over all data and collect hidden states.
    Returns: hidden_states (n, hidden_dim) as numpy array.
    """
    model.eval()
    n = len(mf)
    hidden_dim = model.hidden_dim
    hiddens = np.zeros((n, hidden_dim), dtype=np.float32)

    seq_len = SEQ_LEN
    step = seq_len // 2

    for start in range(0, n - seq_len + 1, step):
        end = min(start + seq_len, n)
        sl = end - start

        mf_t = torch.tensor(mf[start:end], dtype=torch.float32, device=device).unsqueeze(1)
        ps_t = torch.zeros(sl, 1, 6, dtype=torch.float32, device=device)

        m_enc = model.market_encoder(mf_t)
        p_enc = model.pos_encoder(ps_t)
        combined = torch.cat([m_enc, p_enc], dim=-1)

        with torch.no_grad():
            gru_out, _ = model.gru(combined)
            h_out = gru_out.squeeze(1).cpu().numpy()

        hiddens[start:end] = h_out[:sl]

        if start % 50000 == 0 and start > 0:
            log(f"    precomputed {start:,}/{n:,}")

    # Pad last incomplete sequence
    if n > len(hiddens):
        last = hiddens[-1] if len(hiddens) > 0 else np.zeros(hidden_dim, dtype=np.float32)
        for i in range(len(hiddens), n):
            hiddens[i] = last

    return hiddens


def fine_tune_scenario_head(model, train_h, train_scn, val_h, val_scn, epochs=30, lr=1e-3):
    """Train only the scenario_head on pre-computed hidden states."""
    device = next(model.parameters()).device

    for name, param in model.named_parameters():
        param.requires_grad = "scenario_head" in name

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    train_t = torch.tensor(train_h, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_scn, dtype=torch.float32, device=device)
    val_t = torch.tensor(val_h, dtype=torch.float32, device=device)
    val_y = torch.tensor(val_scn, dtype=torch.float32, device=device)

    batch_size = 4096
    best_val = float("inf")
    patience = 8
    stale = 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train_h))
        total_loss = 0.0
        n_batches = 0

        for b in range(0, len(train_h), batch_size):
            idx = perm[b:b + batch_size]
            h_batch = train_t[idx]
            y_batch = train_y[idx]

            pred = model.scenario_head(h_batch)
            loss = loss_fn(pred, y_batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_train = total_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model.scenario_head(val_t)
            val_loss = float(loss_fn(val_pred, val_y).item())

        with torch.no_grad():
            pred_dp = val_pred[:, 0]
            true_dp = val_y[:, 0]
            sign_acc = float(torch.mean((torch.sign(pred_dp) == torch.sign(true_dp)).float()))

        if ep % 5 == 0 or ep == epochs - 1:
            log(f"  Epoch {ep+1:3d}: train={avg_train:.6f} val={val_loss:.6f} sign_acc={sign_acc*100:.1f}%")

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log(f"  Early stop at epoch {ep+1}")
                break

    return best_val


def evaluate(model, val_h, val_scn):
    """Evaluate scenario_head predictions."""
    device = next(model.parameters()).device
    model.eval()
    h_t = torch.tensor(val_h, dtype=torch.float32, device=device)
    y_t = torch.tensor(val_scn, dtype=torch.float32, device=device)

    with torch.no_grad():
        preds = model.scenario_head(h_t).cpu().numpy()
        truths = y_t.cpu().numpy()

    results = {}
    for i, h in enumerate(SCENARIO_HORIZONS):
        off = i * SCENARIO_PARAMS_PER
        p = preds[:, off]
        t = truths[:, off]
        corr = np.corrcoef(p, t)[0, 1]
        sign_acc = np.mean(np.sign(p) == np.sign(t))
        mae = np.mean(np.abs(p - t))
        results[f"h{h}"] = {"corr": corr, "sign_acc": sign_acc, "mae": mae}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    model_path = ROOT / "models" / "kn2_trader.pth"
    out_path = ROOT / "models" / "kn2_scenario.pth"
    csv_path = Path(r"C:\Users\xiaomi\Desktop\XAUUSD5.csv")

    log("=" * 65)
    log("KN 2.0 SCENARIO HEAD FINE-TUNE")
    log(f"  Strategy: Freeze GRU, train scenario_head only")
    log(f"  Train years: {TRAIN_YEARS} | Val: {VAL_YEAR}")
    log(f"  Horizons: {SCENARIO_HORIZONS}")
    log("=" * 65)

    t0 = time.perf_counter()

    log(f"\n[1/5] Loading data...")
    df = prepare_data(csv_path)
    df["year"] = pd.to_datetime(df["datetime"]).dt.year
    tr_mask = df["year"].isin(TRAIN_YEARS).values
    vl_mask = (df["year"] == VAL_YEAR).values
    tr_df = df[tr_mask].copy(); vl_df = df[vl_mask].copy()
    log(f"  Train: {len(tr_df):,} | Val: {len(vl_df):,}")

    log(f"\n[2/5] V14 features...")
    v14_tr = build_v14(tr_df); v14_vl = build_v14(vl_df)
    tr_mf = np.concatenate([v14_tr[:, :68], np.zeros((len(v14_tr), 30), dtype=np.float32)], 1).astype(np.float32)
    vl_mf = np.concatenate([v14_vl[:, :68], np.zeros((len(v14_vl), 30), dtype=np.float32)], 1).astype(np.float32)
    log(f"  Train: {tr_mf.shape} | Val: {vl_mf.shape}")

    log(f"\n[3/5] Scenario labels..."); t1 = time.perf_counter()
    tr_scn = generate_scenario_labels(tr_df, tp_atr_mult=TP_ATR, sl_atr_mult=SL_ATR, max_hold_bars=MAX_HOLD)
    vl_scn = generate_scenario_labels(vl_df, tp_atr_mult=TP_ATR, sl_atr_mult=SL_ATR, max_hold_bars=MAX_HOLD)
    log(f"  Train: {tr_scn.shape} | Val: {vl_scn.shape} | Time: {time.perf_counter()-t1:.1f}s")

    log(f"\n[4/5] Loading model + precomputing hidden states...")
    device = torch.device("cpu")
    KnCls, _ = _build_trader_gru_class(
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=64, num_actions=NUM_ACTIONS
    )
    model = KnCls().to(device)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    log(f"  Model loaded from {model_path}")

    t2 = time.perf_counter()
    tr_h = precompute_hidden_states(model, tr_mf, device)
    vl_h = precompute_hidden_states(model, vl_mf, device)
    log(f"  Hidden states: train={tr_h.shape} val={vl_h.shape} | Time: {time.perf_counter()-t2:.1f}s")

    log(f"\n[5/5] Fine-tuning scenario_head ({args.epochs} epochs)...")
    t3 = time.perf_counter()
    val_loss = fine_tune_scenario_head(
        model, tr_h, tr_scn, vl_h, vl_scn, epochs=args.epochs
    )
    log(f"  Training time: {time.perf_counter()-t3:.1f}s")

    log(f"\nEvaluation by horizon:")
    ev = evaluate(model, vl_h, vl_scn)
    for k in sorted(ev.keys(), key=lambda x: int(x[1:])):
        v = ev[k]
        log(f"  {k:>4s}: corr={v['corr']:+.4f} sign_acc={v['sign_acc']*100:5.1f}% mae={v['mae']:.4f}")

    torch.save(model.state_dict(), out_path)
    meta = {
        "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
        "embed_dim": 64, "num_actions": NUM_ACTIONS, "market_dim": 98, "pos_dim": 6,
        "scenario_trained": True,
        "scenario_horizons": SCENARIO_HORIZONS,
        "val_loss": val_loss,
        "base_model": str(model_path.name),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    total = time.perf_counter() - t0
    log(f"\n{'='*65}")
    log(f"DONE | Model: {out_path} | Total time: {total:.0f}s ({total/60:.0f}m)")
    log(f"{'='*65}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
