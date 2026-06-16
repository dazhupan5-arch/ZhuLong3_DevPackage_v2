#!/usr/bin/env python3
"""
KN2 原油 V6：软标签（Soft Labels）训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心改进（基于 arXiv:2601.13435 软标签方法）:
1. 硬 Triple Barrier 标签 → 温度缩放 softmax 概率分布
   probs = softmax([hold_bias, fwd_ret/temp, -fwd_ret/temp])
   大收益 → 高方向概率，小收益 → 保持观望
2. Stage A: MSELoss 预测收益率（不变）
3. Stage B: KLDivLoss(log_softmax(pred), soft_targets)
4. Stage C: 同上，train_kn2_fast 已支持 action_probs
"""

import sys, time, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 200, 2, 64
NUM_ACTIONS = 3
SEQ_LEN = 64
BATCH_SEQ = 40
OUT_PATH = ROOT / "models" / "kn2_trader_oil.pth"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 黄金同款 Triple Barrier（仅用于评估，不作为硬标签训练）
TP_ATR, SL_ATR, MAX_HOLD = 4.0, 3.0, 128

# 软标签参数
SOFT_HOLD_BIAS, SOFT_TEMP = 4.0, 1.0

def log(msg): print(msg, flush=True)

# ========== 特征工程（与黄金同款） ==========
def build_features_rich(df):
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    feats = {}
    for lag in [1, 3, 5, 10, 20]:
        feats[f"ret_{lag}"] = pd.Series(c).pct_change(lag).fillna(0).values
    for lag in [1, 5, 20]:
        feats[f"logret_{lag}"] = pd.Series(np.log(np.maximum(c, 1e-8))).diff(lag).fillna(0).values
    for lag in [5, 10, 20]:
        feats[f"vol_{lag}"] = (pd.Series(c).rolling(lag).std() / np.maximum(c, 1e-8)).fillna(0).values
    feats["hl_ratio"] = (h - l) / np.maximum(c, 1e-8)
    feats["gap"] = pd.Series((o - np.roll(c, 1)) / np.maximum(np.roll(c, 1), 1e-8)).fillna(0).values
    diff_c = np.diff(c, prepend=c[0])
    gain = np.maximum(diff_c, 0); loss_ = np.maximum(-diff_c, 0)
    for w in [7, 14, 28]:
        ag = pd.Series(gain).rolling(w).mean().fillna(0).values
        al = pd.Series(loss_).rolling(w).mean().fillna(0).values
        feats[f"rsi_{w}"] = 100 - 100 / (1 + ag / np.maximum(al, 1e-8))
    for w in [20, 50]:
        ma = pd.Series(c).rolling(w).mean()
        std = pd.Series(c).rolling(w).std()
        feats[f"bb_{w}"] = ((c - ma) / np.maximum(std, 1e-8)).fillna(0).values
    e12 = pd.Series(c).ewm(span=26, adjust=False).mean()
    e26 = pd.Series(c).ewm(span=52, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=18, adjust=False).mean()
    feats["macd"] = (macd - sig).values
    feats["macd_hist"] = (macd - sig).diff().fillna(0).values
    tr_arr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)),
                                           np.abs(l - np.roll(c, 1))))
    for w in [5, 14, 21]:
        feats[f"atr_{w}"] = (pd.Series(tr_arr).rolling(w).mean() / np.maximum(c, 1e-8)).fillna(0).values
    for lag in [2, 4, 8]:
        feats[f"mom_{lag}"] = (pd.Series(c).diff(lag) / np.maximum(c, 1e-8)).fillna(0).values
    feats["pos_hl"] = (c - l) / np.maximum(h - l, 1e-8)
    e5 = pd.Series(c).ewm(span=10, adjust=False).mean()
    e40 = pd.Series(c).ewm(span=40, adjust=False).mean()
    feats["ema_cross"] = ((e5 - e40) / np.maximum(e40, 1e-8)).values
    feats["vratio"] = (v / np.maximum(pd.Series(v).rolling(10).mean(), 1e-8)).fillna(0).values
    ma60 = pd.Series(c).rolling(60).mean()
    feats["dma_60"] = ((c - ma60) / np.maximum(ma60, 1e-8)).fillna(0).values
    result = np.column_stack(list(feats.values())).astype(np.float32)
    np.nan_to_num(result, nan=0.0, copy=False)
    result = np.clip(result, -10, 10)
    return result, list(feats.keys())

# ========== 软标签：直接从 forward return 生成概率分布 ==========
def generate_soft_labels(df, H=128, hold_bias=4.0, temp=1.0):
    """
    从前向收益率直接生成 3 类概率分布:
      logits = [hold_bias, z/temp, -z/temp]  where z = fwd_ret / vol20
      probs = softmax(logits)
    
    不依赖 Triple Barrier，避免硬标签在均值回归资产上的噪声天花板。
    """
    c = df["close"].values.astype(np.float64)
    n = len(c)
    fwd_ret = np.zeros(n, dtype=np.float64)
    fwd_ret[:n-H] = (c[H:] - c[:-H]) / np.maximum(c[:-H], 1e-8)
    ret1 = pd.Series(c).pct_change(1).fillna(0)
    vol20 = ret1.rolling(20).std().fillna(ret1.std()).values
    z = fwd_ret[:n-H] / np.maximum(vol20[:n-H], 1e-8)
    
    logits = np.column_stack([
        np.full(n - H, hold_bias),
        z / temp,
        -z / temp,
    ])
    logits = np.clip(logits, -20, 20)
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exps / exps.sum(axis=1, keepdims=True)
    
    hard = np.argmax(probs, axis=1).astype(np.int32)
    dist = np.bincount(hard, minlength=3)
    avg = probs.mean(axis=0)
    log(f"  Soft labels (from fwd_ret, hb={hold_bias}, T={temp}):")
    log(f"    hard dist: hold={dist[0]} ({dist[0]/len(hard)*100:.0f}%) "
        f"long={dist[1]} ({dist[1]/len(hard)*100:.0f}%) "
        f"short={dist[2]} ({dist[2]/len(hard)*100:.0f}%)")
    log(f"    avg probs: hold={avg[0]:.3f} long={avg[1]:.3f} short={avg[2]:.3f}")
    return probs.astype(np.float32), hard

# ========== 硬标签（仅评估用） ==========
def generate_hard_labels(df, tp_atr, sl_atr, max_hold):
    n = len(df)
    c = df["close"].values.astype(np.float64)
    hi = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - np.roll(c, 1)),
                                         np.abs(lo - np.roll(c, 1))))
    atr_v = pd.Series(tr).rolling(14).mean().fillna(tr[14] if n > 14 else tr[-1]).values
    av = np.maximum(atr_v, c * 0.0005)
    ut = c + tp_atr * av
    ls = c - sl_atr * av
    us = c + sl_atr * av
    lt = c - tp_atr * av
    actions = np.zeros(n, dtype=np.int32)
    chunk = 5000
    H = max_hold
    for s in range(0, n - H, chunk):
        e = min(s + chunk, n - H); m = e - s
        hi_idx = np.clip(np.arange(s, s + m)[:, None] + np.arange(1, H + 1), 0, n - 1)
        lo_idx = np.clip(np.arange(s, s + m)[:, None] + np.arange(1, H + 1), 0, n - 1)
        ltp = hi[hi_idx] >= ut[s:e, None]
        lsl = lo[lo_idx] <= ls[s:e, None]
        ltf = np.argmax(ltp, axis=1); lsf = np.argmax(lsl, axis=1)
        lta = ltp.any(1); lsa = lsl.any(1)
        stp = lo[lo_idx] <= lt[s:e, None]
        ssl = hi[hi_idx] >= us[s:e, None]
        stf = np.argmax(stp, axis=1); ssf = np.argmax(ssl, axis=1)
        sta = stp.any(1); ssa = ssl.any(1)
        for i in range(m):
            t = s + i
            if lta[i] and (not lsa[i] or ltf[i] < lsf[i]): actions[t] = 1
            elif sta[i] and (not ssa[i] or stf[i] < ssf[i]): actions[t] = 2
            else: actions[t] = 0
    return actions

# ========== 阶段A：自监督预训练（不变） ==========
def train_stage_A(
    market_features, position_states, targets, *,
    val_ratio=0.1, epochs=50, batch_size=20, lr=0.0005, patience=25,
    hidden_dim=200, num_layers=2, embed_dim=64, device="cpu", sequence_length=64
):
    torch, nn = __import__('zhulong.agent.knowledge_net_kn2', fromlist=['_ensure_torch'])._ensure_torch()
    device_obj = torch.device("cpu")
    n_bars = len(market_features)
    seqs = []
    for start in range(0, n_bars - sequence_length, sequence_length // 2):
        end = min(start + sequence_length, n_bars)
        if end - start >= sequence_length // 2:
            seqs.append((start, end))
    n_seqs = len(seqs)
    train_seqs = seqs[:int(n_seqs * (1 - val_ratio))]
    val_seqs = seqs[int(n_seqs * (1 - val_ratio)):]
    log(f"  Stage A: {len(train_seqs)} train seqs / {len(val_seqs)} val seqs")

    from zhulong.agent.knowledge_net_kn2 import _build_trader_gru_class
    KnCls, _ = _build_trader_gru_class(hidden_dim=hidden_dim, num_layers=num_layers, embed_dim=embed_dim, num_actions=3)
    model = KnCls().to(device_obj)
    reg_loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_loss = float("inf"); stale = 0

    for ep in range(epochs):
        model.train()
        train_total = 0.0; n_batches = 0
        perm = torch.randperm(len(train_seqs))
        for b in range(0, len(train_seqs), batch_size):
            batch_seqs = [train_seqs[p.item()] for p in perm[b:b+batch_size]]
            B = len(batch_seqs); S = sequence_length
            mf_batch = torch.zeros(S, B, 98, device=device_obj, dtype=torch.float32)
            ps_batch = torch.zeros(S, B, 6, device=device_obj, dtype=torch.float32)
            ret_target = torch.zeros(B, device=device_obj, dtype=torch.float32)
            for i, (s, e) in enumerate(batch_seqs):
                sl = e - s
                mf_batch[:sl, i] = torch.tensor(market_features[s:e, :98], dtype=torch.float32)
                ps_batch[:sl, i] = torch.tensor(position_states[s:e], dtype=torch.float32)
                if "fwd_ret" in targets:
                    t = min(e, len(targets["fwd_ret"]) - 1)
                    ret_target[i] = float(targets["fwd_ret"][t])
            m_enc = model.market_encoder(mf_batch)
            p_enc = model.pos_encoder(ps_batch)
            gru_out, _ = model.gru(torch.cat([m_enc, p_enc], dim=-1))
            pred_ret = model.embed_head(gru_out[-1]).mean(dim=-1)
            loss = reg_loss_fn(pred_ret, ret_target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_total += loss.item(); n_batches += 1
        scheduler.step()
        avg_train = train_total / max(n_batches, 1)

        model.eval()
        val_total = 0.0; nv = 0
        with torch.no_grad():
            for b in range(0, len(val_seqs), batch_size):
                batch_seqs = val_seqs[b:b+batch_size]
                B = len(batch_seqs); S = sequence_length
                mf_batch = torch.zeros(S, B, 98, device=device_obj, dtype=torch.float32)
                ps_batch = torch.zeros(S, B, 6, device=device_obj, dtype=torch.float32)
                ret_target = torch.zeros(B, device=device_obj, dtype=torch.float32)
                for i, (s, e) in enumerate(batch_seqs):
                    sl = e - s
                    mf_batch[:sl, i] = torch.tensor(market_features[s:e, :98], dtype=torch.float32)
                    ps_batch[:sl, i] = torch.tensor(position_states[s:e], dtype=torch.float32)
                    if "fwd_ret" in targets:
                        t = min(e, len(targets["fwd_ret"]) - 1)
                        ret_target[i] = float(targets["fwd_ret"][t])
                m_enc = model.market_encoder(mf_batch)
                p_enc = model.pos_encoder(ps_batch)
                gru_out, _ = model.gru(torch.cat([m_enc, p_enc], dim=-1))
                pred_ret = model.embed_head(gru_out[-1]).mean(dim=-1)
                val_total += reg_loss_fn(pred_ret, ret_target).item(); nv += 1
        avg_val = val_total / max(nv, 1)

        if ep % 5 == 0 or ep == epochs - 1:
            log(f"  StageA E{ep+1:3d}: train={avg_train:.6f} val={avg_val:.6f}")

        if avg_val < best_loss - 1e-6:
            best_loss = avg_val; stale = 0
            torch.save(model.state_dict(), OUT_PATH)
            meta = {"hidden_dim": hidden_dim, "num_layers": num_layers, "embed_dim": embed_dim,
                    "num_actions": 3, "market_dim": 98, "pos_dim": 6, "val_loss": avg_val}
            OUT_PATH.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        else:
            stale += 1
            if stale >= patience:
                log(f"  Early stop at epoch {ep+1}")
                break
    return {"val_loss": best_loss, "model_path": str(OUT_PATH)}


# ========== 阶段B：软标签训练交易头 ==========
def train_stage_B_soft(
    market_features, position_states, targets, *,
    val_ratio=0.1, epochs=80, batch_size=20, lr=0.0005, patience=40,
    hidden_dim=200, num_layers=2, embed_dim=64, device="cpu", sequence_length=64
):
    """Stage B with soft labels: KLDivLoss(log_softmax(pred), soft_targets)"""
    torch, nn = __import__('zhulong.agent.knowledge_net_kn2', fromlist=['_ensure_torch'])._ensure_torch()
    device_obj = torch.device("cpu")

    from zhulong.agent.knowledge_net_kn2 import _build_trader_gru_class
    KnCls, _ = _build_trader_gru_class(hidden_dim=hidden_dim, num_layers=num_layers, embed_dim=embed_dim, num_actions=3)
    model = KnCls().to(device_obj)

    if OUT_PATH.exists():
        state = torch.load(OUT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        log("  Loaded Stage A weights")
    else:
        log("  WARNING: No pretrained weights found")

    for p in model.market_encoder.parameters(): p.requires_grad = False
    for p in model.pos_encoder.parameters(): p.requires_grad = False
    for p in model.gru.parameters(): p.requires_grad = False

    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    log_softmax = nn.LogSoftmax(dim=-1)

    opt = torch.optim.AdamW(
        list(model.action_head.parameters()) + list(model.size_head.parameters()) +
        list(model.sl_head.parameters()) + list(model.tp_head.parameters()) +
        list(model.trade_head.parameters()) + list(model.conf_head.parameters()),
        lr=lr * 2, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_bars = len(market_features)
    seqs = []
    for start in range(0, n_bars - sequence_length, sequence_length // 2):
        end = min(start + sequence_length, n_bars)
        if end - start >= sequence_length // 2:
            seqs.append((start, end))
    n_seqs = len(seqs)
    train_seqs = seqs[:int(n_seqs * (1 - val_ratio))]
    val_seqs = seqs[int(n_seqs * (1 - val_ratio)):]
    log(f"  Stage B (soft): {len(train_seqs)} train / {len(val_seqs)} val")
    best_loss = float("inf"); stale = 0

    for ep in range(epochs):
        model.train()
        total = 0.0; nb = 0
        perm = torch.randperm(len(train_seqs))
        for b in range(0, len(train_seqs), batch_size):
            batch_seqs = [train_seqs[p.item()] for p in perm[b:b+batch_size]]
            B = len(batch_seqs); S = sequence_length
            mf_batch = torch.zeros(S, B, 98, device=device_obj, dtype=torch.float32)
            ps_batch = torch.zeros(S, B, 6, device=device_obj, dtype=torch.float32)
            for i, (s, e) in enumerate(batch_seqs):
                sl = e - s
                mf_batch[:sl, i] = torch.tensor(market_features[s:e, :98], dtype=torch.float32)
                ps_batch[:sl, i] = torch.tensor(position_states[s:e], dtype=torch.float32)
            with torch.no_grad():
                m_enc = model.market_encoder(mf_batch)
                p_enc = model.pos_encoder(ps_batch)
                gru_out, _ = model.gru(torch.cat([m_enc, p_enc], dim=-1))
            act_logits = model.action_head(gru_out)  # (S, B, 3)
            loss = torch.tensor(0.0, device=device_obj)
            for i, (s, e) in enumerate(batch_seqs):
                sl = e - s
                soft_t = torch.tensor(targets["action_probs"][s:e], dtype=torch.float32, device=device_obj)
                log_p = log_softmax(act_logits[:sl, i])
                loss = loss + kl_loss_fn(log_p, soft_t)
            loss = loss / B
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        scheduler.step()
        avg_train = total / max(nb, 1)

        model.eval()
        vtotal = 0.0; nv = 0
        with torch.no_grad():
            for b in range(0, len(val_seqs), batch_size):
                batch_seqs = val_seqs[b:b+batch_size]
                B = len(batch_seqs); S = sequence_length
                mf_batch = torch.zeros(S, B, 98, device=device_obj, dtype=torch.float32)
                ps_batch = torch.zeros(S, B, 6, device=device_obj, dtype=torch.float32)
                for i, (s, e) in enumerate(batch_seqs):
                    sl = e - s
                    mf_batch[:sl, i] = torch.tensor(market_features[s:e, :98], dtype=torch.float32)
                    ps_batch[:sl, i] = torch.tensor(position_states[s:e], dtype=torch.float32)
                m_enc = model.market_encoder(mf_batch)
                p_enc = model.pos_encoder(ps_batch)
                gru_out, _ = model.gru(torch.cat([m_enc, p_enc], dim=-1))
                act_logits = model.action_head(gru_out)
                vloss = torch.tensor(0.0, device=device_obj)
                for i, (s, e) in enumerate(batch_seqs):
                    sl = e - s
                    soft_t = torch.tensor(targets["action_probs"][s:e], dtype=torch.float32, device=device_obj)
                    log_p = log_softmax(act_logits[:sl, i])
                    vloss = vloss + kl_loss_fn(log_p, soft_t)
                vtotal += (vloss/B).item(); nv += 1
        avg_val = vtotal / max(nv, 1)

        if ep % 5 == 0 or ep == epochs - 1:
            log(f"  StageB E{ep+1:3d}: train={avg_train:.4f} val={avg_val:.4f}")

        if avg_val < best_loss - 0.001:
            best_loss = avg_val; stale = 0
        else:
            stale += 1
            if stale >= patience:
                log(f"  Early stop at epoch {ep+1}")
                break

    torch.save(model.state_dict(), OUT_PATH)
    meta = {"hidden_dim": hidden_dim, "num_layers": num_layers, "embed_dim": embed_dim,
            "num_actions": 3, "market_dim": 98, "pos_dim": 6, "val_loss": best_loss}
    OUT_PATH.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return {"val_loss": best_loss}


# ========== 阶段C：联合微调（train_kn2_fast 已支持 action_probs） ==========
def train_stage_C(
    market_features, position_states, targets, *,
    val_ratio=0.1, epochs=60, batch_size=20, lr=0.00005, patience=30,
    hidden_dim=200, num_layers=2, embed_dim=64, device="cpu", sequence_length=64
):
    from zhulong.agent.knowledge_net_kn2 import train_kn2_fast
    return train_kn2_fast(
        market_features=market_features, position_states=position_states,
        targets=targets,
        val_ratio=val_ratio, epochs=epochs, batch_size=batch_size,
        lr=lr, patience=patience,
        class_weights=[1.0, 1.0, 1.0],
        num_actions=3, hidden_dim=hidden_dim, num_layers=num_layers,
        embed_dim=embed_dim, out_path=OUT_PATH, device=device,
        sequence_length=sequence_length,
    )


# ========== 评估（对硬标签） ==========
def evaluate_model(path, val_mf, val_labels):
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
    from sklearn.metrics import accuracy_score, confusion_matrix
    k = KN2Inference(path)
    if not k.is_ready: return {"pass": False, "acc": 0}
    p = np.array([k.predict(val_mf[i], encode_position_state())["action"] for i in range(len(val_mf))])
    p = np.clip(p, 0, 2)
    ll = val_labels[:len(p)]
    acc = accuracy_score(ll, p)
    cm = confusion_matrix(ll, p, labels=[0, 1, 2])
    pr = []; pd = np.bincount(p, minlength=3) / len(p)
    for c in range(3):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp
        pr.append(tp / max(tp + fp, 1))
    ok = acc > 0.40 and pr[1] > 0.35 and pr[2] > 0.35 and all(x > 0.05 for x in pd)
    log(f"  Acc: {acc*100:.2f}% | Prec: h={pr[0]*100:.0f}% l={pr[1]*100:.0f}% s={pr[2]*100:.0f}%")
    log(f"  Dist: h={pd[0]*100:.0f}% l={pd[1]*100:.0f}% s={pd[2]*100:.0f}% | PASS: {ok}")
    log(f"  CM:")
    for i, nm in enumerate(["hold", "long", "short"]):
        log(f"    {nm:5s} {cm[i,0]:5d} {cm[i,1]:5d} {cm[i,2]:5d}")
    return {"pass": ok, "acc": acc, "precs": pr, "cm": cm}


# ========== 主流程 ==========
def main():
    log("=" * 60)
    log("KN 2.0 原油 V6: 软标签训练")
    log(f"  核心: softmax([{SOFT_HOLD_BIAS}, ret/{SOFT_TEMP}, -ret/{SOFT_TEMP}])")
    log(f"  200d GRU x2 | H=128 | KLDivLoss")
    log("=" * 60)

    t0 = time.perf_counter()

    log("\n--- 数据准备 ---")
    log("加载 CSV...")
    df = pd.read_csv(r"C:\Users\xiaomi\Desktop\XTIUSD5.csv", header=None,
                     names=["date","time","open","high","low","close","volume"])
    df = df.dropna(subset=["open","high","low","close"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df = df.sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year
    log(f"  {len(df):,} bars loaded")

    log("计算特征...")
    t_feat = time.perf_counter()
    feats, _ = build_features_rich(df)
    feats_pad = np.pad(feats, ((0,0),(0,98-feats.shape[1])))[:,:98].astype(np.float32)
    feats_pad = (feats_pad - feats_pad.mean(axis=0)) / (feats_pad.std(axis=0) + 1e-8)
    feats_pad = np.clip(feats_pad, -5, 5)
    log(f"  特征: {feats_pad.shape} | 耗时: {time.perf_counter()-t_feat:.1f}s")

    log("生成软标签（前向收益率驱动，hb=4.0, T=1.0）...")
    t_lbl = time.perf_counter()
    soft_probs, soft_hard = generate_soft_labels(df, H=MAX_HOLD, hold_bias=SOFT_HOLD_BIAS, temp=SOFT_TEMP)
    # 硬标签（仅评估用）
    hard_labels = generate_hard_labels(df, TP_ATR, SL_ATR, MAX_HOLD)
    log(f"  标签耗时: {time.perf_counter()-t_lbl:.1f}s")

    H = MAX_HOLD
    tr_mask_full = df["year"].isin(TRAIN_YEARS).values
    vl_mask_full = (df["year"] == VAL_YEAR).values
    tr_mask = tr_mask_full[:-H]  # 对齐到 n-H 长度
    vl_mask = vl_mask_full[:-H]

    tr_mf = feats_pad[:-H][tr_mask]
    vl_mf = feats_pad[:-H][vl_mask]
    tr_soft = soft_probs[tr_mask]
    tr_hard = hard_labels[:-H][tr_mask]
    vl_hard = hard_labels[:-H][vl_mask]
    tr_soft_hard = soft_hard[tr_mask]
    pos_tr = np.zeros((len(tr_mf), 6), dtype=np.float32)

    c = df["close"].values.astype(np.float64)
    n = len(c)
    fwd_ret = np.zeros(n, dtype=np.float32)
    fwd_ret[:n-H] = (c[H:] - c[:-H]) / np.maximum(c[:-H], 1e-8)

    train_targets = {
        "action": tr_soft_hard,        # 软标签的 argmax 作为硬标签
        "action_probs": tr_soft,       # KLDivLoss 用的软目标
        "fwd_ret": fwd_ret[:n-H][tr_mask],
        "position_size": np.ones(len(tr_mf), dtype=np.float32),
        "sl_atr_mult": np.full(len(tr_mf), SL_ATR, dtype=np.float32),
        "tp_atr_mult": np.full(len(tr_mf), TP_ATR, dtype=np.float32),
        "should_trade": (tr_soft_hard > 0).astype(np.float32),
    }

    log(f"  Train: {len(tr_mf):,} bars | Val: {len(vl_mf):,} bars")
    log(f"  Soft mean: [{tr_soft[:,0].mean():.3f}, {tr_soft[:,1].mean():.3f}, {tr_soft[:,2].mean():.3f}]")

    # ========== 阶段A ==========
    log(f"\n--- Stage A: Pre-training (50 epochs) ---")
    stage_a = train_stage_A(tr_mf, pos_tr, train_targets,
                            val_ratio=0.1, epochs=50, batch_size=BATCH_SEQ, lr=0.0005, patience=25,
                            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
                            device="cpu", sequence_length=SEQ_LEN)
    log(f"  Stage A best val_loss: {stage_a['val_loss']:.6f}")

    # ========== 阶段B：软标签 ==========
    log(f"\n--- Stage B: Soft-label Head Training (80 epochs) ---")
    stage_b = train_stage_B_soft(tr_mf, pos_tr, train_targets,
                                 val_ratio=0.1, epochs=80, batch_size=BATCH_SEQ, lr=0.0005, patience=40,
                                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
                                 device="cpu", sequence_length=SEQ_LEN)

    # ========== 阶段C ==========
    log(f"\n--- Stage C: Joint Fine-tuning (60 epochs) ---")
    stage_c = train_stage_C(tr_mf, pos_tr, train_targets,
                            val_ratio=0.1, epochs=60, batch_size=BATCH_SEQ, lr=0.00005, patience=30,
                            hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
                            device="cpu", sequence_length=SEQ_LEN)

    # ========== 评估 ==========
    log(f"\n--- Final Evaluation (vs hard labels) ---")
    ev = evaluate_model(OUT_PATH, vl_mf, vl_hard)

    total_time = time.perf_counter() - t0
    status = "PASSED" if ev["pass"] else "NOT PASSED"
    log(f"\n{'='*60}")
    log(f"  {status}  Acc={ev['acc']*100:.2f}%  Time={total_time:.0f}s")
    log("=" * 60)
    return 0 if ev["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
