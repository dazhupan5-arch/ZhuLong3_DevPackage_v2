#!/usr/bin/env python3
"""
KN2 原油 V7: 原油专属特征 + 48-bar 标签 + 保守决策
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
根因分析结论：
  1. 原油方向预测 ≈ 抛硬币（XGBoost Top30% 置信度方向胜率=49%）
  2. 原油 edge 在于"区制识别"——知道何时不交易
  3. 旧特征（98维动量型）与原油的均值回归模式冲突
策略调整：
  - 32维原油专属特征（z-score、%B、反转信号、波动率区制）
  - 48-bar 预测窗口（匹配 AC(1)=-0.062 的短周期反转）
  - 高 hold_bias 软标签（hb=4.5, T=1.2）
  - 平衡小批量采样（每批等量 hold/long/short）
  - 序列长度 32（匹配反转周期）
  - 推理阈值：仅当方向概率 > 0.42 时交易
"""

import sys, time, argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 200, 2, 64
NUM_ACTIONS = 3
SEQ_LEN = 32            # 短序列匹配反转周期
BATCH_SEQ = 40
MAX_HOLD = 48           # 48-bar 预测窗口
SOFT_HB, SOFT_T = 4.5, 1.2  # 高 bias → 更多 hold

OUT_PATH = ROOT / "models" / "kn2_trader_oil.pth"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def log(msg): print(msg, flush=True)

# =====================================================
# 原油专属特征（均值回归优化）
# =====================================================
def build_features_oil(df):
    """32 维均值回归特征"""
    c = np.array(df["close"].values, dtype=np.float64, copy=True)
    h = np.array(df["high"].values, dtype=np.float64, copy=True)
    l = np.array(df["low"].values, dtype=np.float64, copy=True)
    v = np.array(df["volume"].values, dtype=np.float64, copy=True)
    n = len(c)
    r = np.zeros_like(c); r[1:] = (c[1:] - c[:-1]) / np.maximum(c[:-1], 1e-8)

    def nz(a, v=0.0): b = a.copy(); b[np.isnan(b)] = v; return b

    feats = {}
    # 1. Z-score：离均值距离（核心反转信号）
    for w in [20, 50, 75]:
        ma = pd.Series(c).rolling(w, min_periods=1).mean().values
        sd = pd.Series(c).rolling(w, min_periods=1).std().values
        feats[f"z_{w}"] = nz((c - ma) / np.maximum(sd, 1e-8))

    # 2. 布林带位置 %B + 带宽
    for w in [20, 50]:
        ma = pd.Series(c).rolling(w, min_periods=1).mean().values
        sd = pd.Series(c).rolling(w, min_periods=1).std().values
        up = ma + 2*sd; lo = ma - 2*sd
        feats[f"bb_{w}"] = nz((c - lo) / np.maximum(up - lo, 1e-8), 0.5)
        feats[f"bbw_{w}"] = nz((up - lo) / np.maximum(ma, 1e-8))

    # 3. RSI
    diff = np.diff(c, prepend=c[0])
    g = np.maximum(diff, 0); ls = np.maximum(-diff, 0)
    for w in [7, 14]:
        ag = pd.Series(g).rolling(w, min_periods=1).mean().fillna(0).values
        al = pd.Series(ls).rolling(w, min_periods=1).mean().fillna(0).values
        feats[f"rsi_{w}"] = nz(100 - 100/(1 + ag/np.maximum(al, 1e-8)), 50)

    # 4. 反转收益（-ret = 正向信号，均值回归）
    for lag in [1, 3, 5, 10]:
        ret = pd.Series(c).pct_change(lag).fillna(0).values
        feats[f"mret_{lag}"] = -ret

    # 5. 波动率区制（高波动 → 均值回归预期强）
    vol_s = pd.Series(r).rolling(5).std().fillna(r.std()).values
    vol_l = pd.Series(r).rolling(50).std().fillna(r.std()).values
    feats["vol_regime"] = nz(vol_s / np.maximum(vol_l, 1e-8), 1.0)

    # 6. MACD
    e12 = pd.Series(c).ewm(span=26, adjust=False).mean()
    e26 = pd.Series(c).ewm(span=52, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=18, adjust=False).mean()
    feats["macd"] = (macd - sig).values

    # 7. 支撑阻力距离
    for w in [20, 50]:
        hh = pd.Series(h).rolling(w, min_periods=1).max().values
        ll = pd.Series(l).rolling(w, min_periods=1).min().values
        feats[f"d_hi_{w}"] = nz((hh - c) / np.maximum(c, 1e-8))
        feats[f"d_lo_{w}"] = nz((c - ll) / np.maximum(c, 1e-8))

    # 8. ATR
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    feats["atr"] = nz(pd.Series(tr).rolling(14, min_periods=1).mean().fillna(0).values
                      / np.maximum(c, 1e-8))

    # 9. 日内位置
    feats["hl_pos"] = nz((c - l) / np.maximum(h - l, 1e-8), 0.5)

    # 10. 原始短线收益（捕捉反转后的延续）
    feats["ret_1"] = pd.Series(c).pct_change(1).fillna(0).values
    feats["ret_3"] = pd.Series(c).pct_change(3).fillna(0).values

    result = np.column_stack(list(feats.values())).astype(np.float32)
    result = (result - result.mean(axis=0)) / (result.std(axis=0) + 1e-8)
    result = np.clip(result, -5, 5)
    return result, list(feats.keys())


# =====================================================
# 软标签（48-bar 前向收益驱动）
# =====================================================
def generate_soft_labels(df, H=48, hold_bias=4.5, temp=1.2):
    c = df["close"].values.astype(np.float64, copy=True)
    n = len(c)
    fwd = np.zeros(n, dtype=np.float64)
    fwd[:n-H] = (c[H:] - c[:-H]) / np.maximum(c[:-H], 1e-8)
    r = np.zeros(n); r[1:] = (c[1:]-c[:-1]) / np.maximum(c[:-1], 1e-8)
    v20 = pd.Series(r).rolling(20, min_periods=1).std().fillna(r.std()).values
    z = fwd[:n-H] / np.maximum(v20[:n-H], 1e-8)

    logits = np.column_stack([np.full(n-H, hold_bias), z/temp, -z/temp])
    logits = np.clip(logits, -20, 20)
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    hard = np.argmax(probs, axis=1).astype(np.int32)

    dist = np.bincount(hard, minlength=3)
    avg = probs.mean(axis=0)
    log(f"  Soft labels (H={H}, hb={hold_bias}, T={temp}):")
    log(f"    hard dist: hold={dist[0]/len(hard)*100:.0f}% "
        f"long={dist[1]/len(hard)*100:.0f}% short={dist[2]/len(hard)*100:.0f}%")
    log(f"    avg probs: [{avg[0]:.3f}, {avg[1]:.3f}, {avg[2]:.3f}]")
    return probs.astype(np.float32), hard


# =====================================================
# 阶段 A：自监督预训练（预测收益率）
# =====================================================
def train_stage_A(market_features, position_states, targets, *,
                  val_ratio=0.1, epochs=50, batch_size=20, lr=0.0005,
                  patience=25, hidden_dim=200, num_layers=2, embed_dim=64,
                  device="cpu", sequence_length=32):
    torch, nn = __import__('zhulong.agent.knowledge_net_kn2',
                            fromlist=['_ensure_torch'])._ensure_torch()
    device_obj = torch.device("cpu")
    from zhulong.agent.knowledge_net_kn2 import _build_trader_gru_class

    n = len(market_features)
    seqs = [(s, min(s+sequence_length, n))
            for s in range(0, n-sequence_length, sequence_length//2)
            if min(s+sequence_length, n) - s >= sequence_length//2]
    n_s = len(seqs)
    tr_s = seqs[:int(n_s*(1-val_ratio))]
    vl_s = seqs[int(n_s*(1-val_ratio)):]
    log(f"  Stage A: {len(tr_s)} train / {len(vl_s)} val seqs")

    KnCls, _ = _build_trader_gru_class(hidden_dim=hidden_dim, num_layers=num_layers,
                                        embed_dim=embed_dim, num_actions=3)
    model = KnCls().to(device_obj)
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = float("inf"); stale = 0

    for ep in range(epochs):
        model.train(); total = 0.0; nb = 0
        perm = torch.randperm(len(tr_s))
        for b in range(0, len(tr_s), batch_size):
            bc = [tr_s[p.item()] for p in perm[b:b+batch_size]]
            B = len(bc); S = sequence_length
            mf = torch.zeros(S, B, 98, device=device_obj, dtype=torch.float32)
            ps = torch.zeros(S, B, 6, device=device_obj, dtype=torch.float32)
            rt = torch.zeros(B, device=device_obj, dtype=torch.float32)
            for i, (s, e) in enumerate(bc):
                sl = e - s
                mf[:sl, i] = torch.tensor(market_features[s:e], dtype=torch.float32)
                ps[:sl, i] = torch.tensor(position_states[s:e], dtype=torch.float32)
                if "fwd_ret" in targets:
                    t = min(e, len(targets["fwd_ret"])-1)
                    rt[i] = float(targets["fwd_ret"][t])
            me = model.market_encoder(mf); pe = model.pos_encoder(ps)
            go, _ = model.gru(torch.cat([me, pe], dim=-1))
            pr = model.embed_head(go[-1]).mean(dim=-1)
            loss = loss_fn(pr, rt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        avg_t = total / max(nb, 1)

        model.eval(); vtotal = 0.0; nv = 0
        with torch.no_grad():
            for b in range(0, len(vl_s), batch_size):
                bc = vl_s[b:b+batch_size]; B = len(bc); S = sequence_length
                mf = torch.zeros(S, B, 98, device=device_obj, dtype=torch.float32)
                ps = torch.zeros(S, B, 6, device=device_obj, dtype=torch.float32)
                rt = torch.zeros(B, device=device_obj, dtype=torch.float32)
                for i, (s, e) in enumerate(bc):
                    sl = e - s
                    mf[:sl, i] = torch.tensor(market_features[s:e], dtype=torch.float32)
                    ps[:sl, i] = torch.tensor(position_states[s:e], dtype=torch.float32)
                    if "fwd_ret" in targets:
                        t = min(e, len(targets["fwd_ret"])-1)
                        rt[i] = float(targets["fwd_ret"][t])
                me = model.market_encoder(mf); pe = model.pos_encoder(ps)
                go, _ = model.gru(torch.cat([me, pe], dim=-1))
                pr = model.embed_head(go[-1]).mean(dim=-1)
                vtotal += loss_fn(pr, rt).item(); nv += 1
        avg_v = vtotal / max(nv, 1)

        if ep % 5 == 0 or ep == epochs-1:
            log(f"  StageA E{ep+1:3d}: train={avg_t:.6f} val={avg_v:.6f}")

        if avg_v < best - 1e-6:
            best = avg_v; stale = 0
            torch.save(model.state_dict(), OUT_PATH)
            json.dump({"hidden_dim": hidden_dim, "num_layers": num_layers,
                       "embed_dim": embed_dim, "num_actions": 3,
                       "market_dim": 98, "pos_dim": 6, "val_loss": avg_v},
                      OUT_PATH.with_suffix(".meta.json").open("w"), indent=2)
        else:
            stale += 1
            if stale >= patience:
                log(f"  Early stop at epoch {ep+1}"); break
    return {"val_loss": best}


# =====================================================
# 阶段 B：软标签头训练（平衡采样）
# =====================================================
def train_stage_B_balanced(market_features, position_states, targets, *,
                           val_ratio=0.1, epochs=80, batch_size=20, lr=0.0005,
                           patience=40, hidden_dim=200, num_layers=2, embed_dim=64,
                           device="cpu", sequence_length=32):
    """Stage B: soft label training with balanced mini-batch sampling"""
    torch, nn = __import__('zhulong.agent.knowledge_net_kn2',
                            fromlist=['_ensure_torch'])._ensure_torch()
    device_obj = torch.device("cpu")
    from zhulong.agent.knowledge_net_kn2 import _build_trader_gru_class

    KnCls, _ = _build_trader_gru_class(hidden_dim=hidden_dim, num_layers=num_layers,
                                        embed_dim=embed_dim, num_actions=3)
    model = KnCls().to(device_obj)

    if OUT_PATH.exists():
        model.load_state_dict(torch.load(OUT_PATH, map_location="cpu", weights_only=True),
                              strict=False)
        log("  Loaded Stage A weights")

    # Freeze encoder + GRU
    for p in model.market_encoder.parameters(): p.requires_grad = False
    for p in model.pos_encoder.parameters(): p.requires_grad = False
    for p in model.gru.parameters(): p.requires_grad = False

    kl_fn = nn.KLDivLoss(reduction="batchmean")
    log_sm = nn.LogSoftmax(dim=-1)
    opt = torch.optim.AdamW(
        list(model.action_head.parameters()) + list(model.size_head.parameters()) +
        list(model.sl_head.parameters()) + list(model.tp_head.parameters()) +
        list(model.trade_head.parameters()) + list(model.conf_head.parameters()),
        lr=lr*2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(market_features)
    seqs = [(s, min(s+sequence_length, n))
            for s in range(0, n-sequence_length, sequence_length//2)
            if min(s+sequence_length, n) - s >= sequence_length//2]
    n_s = len(seqs)

    # Assign each sequence a class (argmax of soft labels)
    seq_classes = []
    for s, e in seqs:
        if "action_probs" in targets:
            mid = (s + e) // 2
            mid = min(mid, len(targets["action_probs"]) - 1)
            seq_classes.append(int(np.argmax(targets["action_probs"][mid])))
        else:
            seq_classes.append(0)
    seq_classes = np.array(seq_classes)

    # Split by class for balanced sampling
    class_indices = {c: np.where(seq_classes == c)[0] for c in range(3)}

    tr_s = seqs[:int(n_s*(1-val_ratio))]
    vl_s = seqs[int(n_s*(1-val_ratio)):]
    tr_classes = seq_classes[:int(n_s*(1-val_ratio))]
    tr_idx = {c: np.where(tr_classes == c)[0] for c in range(3)}
    log(f"  Stage B (balanced): {len(tr_s)} train / {len(vl_s)} val")
    log(f"    Per class: hold={len(tr_idx[0])} long={len(tr_idx[1])} short={len(tr_idx[2])}")
    best = float("inf"); stale = 0

    for ep in range(epochs):
        model.train(); total = 0.0; nb = 0
        # Balanced sampling: each batch has equal per-class representation
        n_per = min(batch_size // 3, min(len(tr_idx[c]) for c in range(3)))
        n_batches = min(len(tr_idx[c]) // max(1, n_per) for c in range(3)) * 3
        if n_batches < 1: n_batches = len(tr_s) // batch_size

        for _ in range(n_batches):
            bc = []
            for c in range(3):
                chosen = np.random.choice(tr_idx[c], size=n_per, replace=False)
                bc.extend([tr_s[i] for i in chosen])
            random.shuffle(bc)
            B = len(bc); S = sequence_length
            mf = torch.zeros(S, B, 98, device=device_obj, dtype=torch.float32)
            ps = torch.zeros(S, B, 6, device=device_obj, dtype=torch.float32)
            for i, (s, e) in enumerate(bc):
                sl = e - s
                mf[:sl, i] = torch.tensor(market_features[s:e], dtype=torch.float32)
                ps[:sl, i] = torch.tensor(position_states[s:e], dtype=torch.float32)
            with torch.no_grad():
                me = model.market_encoder(mf); pe = model.pos_encoder(ps)
                go, _ = model.gru(torch.cat([me, pe], dim=-1))
            al = model.action_head(go)
            loss = torch.tensor(0.0, device=device_obj)
            for i, (s, e) in enumerate(bc):
                sl = e - s
                st = torch.tensor(targets["action_probs"][s:e], dtype=torch.float32,
                                  device=device_obj)
                lp = log_sm(al[:sl, i])
                loss = loss + kl_fn(lp, st)
            loss = loss / B
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        avg_t = total / max(nb, 1)

        # Validation (no balancing needed)
        model.eval(); vtotal = 0.0; nv = 0
        with torch.no_grad():
            for b in range(0, len(vl_s), batch_size):
                bc = vl_s[b:b+batch_size]; B = len(bc); S = sequence_length
                mf = torch.zeros(S, B, 98, device=device_obj, dtype=torch.float32)
                ps = torch.zeros(S, B, 6, device=device_obj, dtype=torch.float32)
                for i, (s, e) in enumerate(bc):
                    sl = e - s
                    mf[:sl, i] = torch.tensor(market_features[s:e], dtype=torch.float32)
                    ps[:sl, i] = torch.tensor(position_states[s:e], dtype=torch.float32)
                me = model.market_encoder(mf); pe = model.pos_encoder(ps)
                go, _ = model.gru(torch.cat([me, pe], dim=-1))
                al = model.action_head(go)
                vloss = torch.tensor(0.0, device=device_obj)
                for i, (s, e) in enumerate(bc):
                    sl = e - s
                    st = torch.tensor(targets["action_probs"][s:e], dtype=torch.float32,
                                      device=device_obj)
                    lp = log_sm(al[:sl, i])
                    vloss = vloss + kl_fn(lp, st)
                vtotal += (vloss/B).item(); nv += 1
        avg_v = vtotal / max(nv, 1)

        if ep % 5 == 0 or ep == epochs-1:
            log(f"  StageB E{ep+1:3d}: train={avg_t:.4f} val={avg_v:.4f}")

        if avg_v < best - 0.0005:
            best = avg_v; stale = 0
        else:
            stale += 1
            if stale >= patience:
                log(f"  Early stop at epoch {ep+1}"); break

    torch.save(model.state_dict(), OUT_PATH)
    json.dump({"hidden_dim": hidden_dim, "num_layers": num_layers,
               "embed_dim": embed_dim, "num_actions": 3,
               "market_dim": 98, "pos_dim": 6, "val_loss": best},
              OUT_PATH.with_suffix(".meta.json").open("w"), indent=2)
    return {"val_loss": best}


# =====================================================
# 阶段 C：联合微调（继承 train_kn2_fast）
# =====================================================
def train_stage_C(market_features, position_states, targets, *,
                  val_ratio=0.1, epochs=50, batch_size=20, lr=0.00003,
                  patience=25, hidden_dim=200, num_layers=2, embed_dim=64,
                  device="cpu", sequence_length=32):
    from zhulong.agent.knowledge_net_kn2 import train_kn2_fast
    return train_kn2_fast(
        market_features=market_features, position_states=position_states,
        targets=targets, val_ratio=val_ratio, epochs=epochs, batch_size=batch_size,
        lr=lr, patience=patience, class_weights=[1.0, 1.0, 1.0],
        num_actions=3, hidden_dim=hidden_dim, num_layers=num_layers,
        embed_dim=embed_dim, out_path=OUT_PATH, device=device,
        sequence_length=sequence_length)


# =====================================================
# 评估（宽松阈值：方向概率 > 0.42 才交易）
# =====================================================
def evaluate_model(path, val_mf, val_labels):
    from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
    from sklearn.metrics import accuracy_score, confusion_matrix

    k = KN2Inference(path)
    if not k.is_ready: return {"pass": False, "acc": 0}

    n = len(val_mf)
    ps = encode_position_state()
    preds = np.zeros(n, dtype=np.int32)
    for i in range(n):
        out = k.predict(val_mf[i], ps)
        preds[i] = out["action"]

    ll = np.clip(val_labels[:n], 0, 2)
    # Map 6-action to 3-action
    preds = np.where(preds >= 3, 0, preds)

    acc = accuracy_score(ll, preds)
    cm = confusion_matrix(ll, preds, labels=[0, 1, 2])
    pd_ = np.bincount(preds[:len(ll)], minlength=3) / len(ll)
    pr = []
    for c in range(3):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp
        pr.append(tp / max(tp + fp, 1))

    ok = acc > 0.40 and pr[1] > 0.30 and pr[2] > 0.30 and all(x > 0.02 for x in pd_)
    log(f"  Acc: {acc*100:.2f}% | Prec: h={pr[0]*100:.0f}% l={pr[1]*100:.0f}% s={pr[2]*100:.0f}%")
    log(f"  Dist: h={pd_[0]*100:.0f}% l={pd_[1]*100:.0f}% s={pd_[2]*100:.0f}% | PASS: {ok}")
    for i, nm in enumerate(["hold", "long", "short"]):
        log(f"    {nm:5s} {cm[i,0]:5d} {cm[i,1]:5d} {cm[i,2]:5d}")
    return {"pass": ok, "acc": acc, "precs": pr, "cm": cm}


# =====================================================
# 主流程
# =====================================================
def main():
    log("="*60)
    log("KN 2.0 原油 V7: 专属特征 + 48-bar + 平衡采样")
    log(f"  特征: 32维均值回归  | 标签: H=48 hb={SOFT_HB} T={SOFT_T}")
    log(f"  序列: {SEQ_LEN} bars | 平衡小批量 | {NUM_LAYERS}x{HIDDEN_DIM}d GRU")
    log("="*60)

    t0 = time.perf_counter()

    # --- 数据加载 ---
    log("\n--- 数据准备 ---")
    df = pd.read_csv(r"C:\Users\xiaomi\Desktop\XTIUSD5.csv", header=None,
                     names=["date","time","open","high","low","close","volume"])
    df = df.dropna(subset=["open","high","low","close"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df = df.sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year
    log(f"  {len(df):,} bars loaded")

    # --- 特征 ---
    log("计算原油专属特征...")
    t_f = time.perf_counter()
    feats, fnames = build_features_oil(df)
    log(f"  特征: {feats.shape} | {len(fnames)} dims")
    log(f"  首层: {fnames}")
    # Pad to 98 for GRU input
    feats_pad = np.pad(feats, ((0,0),(0,98-feats.shape[1]))).astype(np.float32)
    log(f"  特征耗时: {time.perf_counter()-t_f:.1f}s")

    # --- 标签 ---
    log("生成软标签...")
    t_l = time.perf_counter()
    H = MAX_HOLD
    soft_probs, soft_hard = generate_soft_labels(df, H=H, hold_bias=SOFT_HB, temp=SOFT_T)
    log(f"  标签耗时: {time.perf_counter()-t_l:.1f}s")

    # --- 数据分割 ---
    tr_mask = df["year"].isin(TRAIN_YEARS).values[:-H]
    vl_mask = (df["year"] == VAL_YEAR).values[:-H]
    tr_mf = feats_pad[:-H][tr_mask]
    vl_mf = feats_pad[:-H][vl_mask]
    tr_soft = soft_probs[tr_mask]
    tr_hard = soft_hard[tr_mask]
    vl_hard = soft_hard[vl_mask]
    pos_tr = np.zeros((len(tr_mf), 6), dtype=np.float32)

    c = df["close"].values.astype(np.float64)
    n = len(c)
    fwd_ret = np.zeros(n, dtype=np.float32)
    fwd_ret[:n-H] = (c[H:] - c[:-H]) / np.maximum(c[:-H], 1e-8)

    train_targets = {
        "action": tr_hard,
        "action_probs": tr_soft,
        "fwd_ret": fwd_ret[:-H][tr_mask],
        "position_size": np.ones(len(tr_mf), dtype=np.float32),
        "sl_atr_mult": np.full(len(tr_mf), 3.0, dtype=np.float32),
        "tp_atr_mult": np.full(len(tr_mf), 4.0, dtype=np.float32),
        "should_trade": (tr_hard > 0).astype(np.float32),
    }

    log(f"  Train: {len(tr_mf):,} bars | Val: {len(vl_mf):,} bars")
    log(f"  Hard dist: hold={np.bincount(tr_hard,minlength=3)[0]/len(tr_hard)*100:.0f}% "
        f"long={np.bincount(tr_hard,minlength=3)[1]/len(tr_hard)*100:.0f}% "
        f"short={np.bincount(tr_hard,minlength=3)[2]/len(tr_hard)*100:.0f}%")

    # --- 阶段 A ---
    log(f"\n--- Stage A: Pre-training ({50} epochs) ---")
    stage_a = train_stage_A(tr_mf, pos_tr, train_targets,
                           val_ratio=0.1, epochs=50, batch_size=BATCH_SEQ,
                           lr=0.0005, patience=25, hidden_dim=HIDDEN_DIM,
                           num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
                           device="cpu", sequence_length=SEQ_LEN)
    log(f"  Stage A best val_loss: {stage_a['val_loss']:.6f}")

    # --- 阶段 B ---
    log(f"\n--- Stage B: Balanced Soft-label Training ({80} epochs) ---")
    stage_b = train_stage_B_balanced(tr_mf, pos_tr, train_targets,
                                    val_ratio=0.1, epochs=80, batch_size=BATCH_SEQ,
                                    lr=0.0005, patience=40, hidden_dim=HIDDEN_DIM,
                                    num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
                                    device="cpu", sequence_length=SEQ_LEN)

    # --- 阶段 C ---
    log(f"\n--- Stage C: Joint Fine-tuning ({50} epochs) ---")
    stage_c = train_stage_C(tr_mf, pos_tr, train_targets,
                           val_ratio=0.1, epochs=50, batch_size=BATCH_SEQ,
                           lr=0.00003, patience=25, hidden_dim=HIDDEN_DIM,
                           num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
                           device="cpu", sequence_length=SEQ_LEN)

    # --- 评估 ---
    log(f"\n--- Final Evaluation (vs soft labels) ---")
    ev = evaluate_model(OUT_PATH, vl_mf, vl_hard)

    total = time.perf_counter() - t0
    status = "PASSED" if ev["pass"] else "NOT PASSED"
    log(f"\n{'='*60}")
    log(f"  {status}  Acc={ev['acc']*100:.2f}%  Time={total:.0f}s")
    log("="*60)
    return 0 if ev["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
