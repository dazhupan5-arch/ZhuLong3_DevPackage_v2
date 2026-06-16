#!/usr/bin/env python3
"""
KN2 原油（XTIUSD）完整训练方案：课程学习 + 平衡标签 + 高质量特征
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阶段A：自监督预训练 — GRU学习市场动态（60 epoch）
阶段B：冻结GRU + 训练交易头 — 动作分类（80 epoch）
阶段C：联合微调 — 全模型优化（60 epoch）

核心思想：先让GRU学会"读懂市场"，再学"如何交易"
"""

import sys, time, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ===== 固定配置 =====
TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025
HIDDEN_DIM, NUM_LAYERS, EMBED_DIM = 200, 2, 64
NUM_ACTIONS = 3
SEQ_LEN = 64
BATCH_SEQ = 40
OUT_PATH = ROOT / "models" / "kn2_trader_oil.pth"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ===== 标签参数 — 黄金同款 =====
TP_ATR, SL_ATR, MAX_HOLD = 4.0, 3.0, 128

def log(msg): print(msg, flush=True)

# ========== 特征工程 ==========
def build_features_rich(df):
    """
    构建30维高质量特征：收益率/波动率/RSI/布林带/MACD/ATR/动量/量比
    全部用pandas向量化
    """
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    n = len(c)
    feats = {}

    # 1-5: 多周期收益率
    for lag in [1, 3, 5, 10, 20]:
        feats[f"ret_{lag}"] = pd.Series(c).pct_change(lag).fillna(0).values
    # 6-8: 对数收益率
    for lag in [1, 5, 20]:
        feats[f"logret_{lag}"] = pd.Series(np.log(np.maximum(c, 1e-8))).diff(lag).fillna(0).values

    # 9-11: 波动率
    for lag in [5, 10, 20]:
        feats[f"vol_{lag}"] = (pd.Series(c).rolling(lag).std() / np.maximum(c, 1e-8)).fillna(0).values

    # 12-13: HL ratio + gap
    feats["hl_ratio"] = (h - l) / np.maximum(c, 1e-8)
    feats["gap"] = pd.Series((o - np.roll(c, 1)) / np.maximum(np.roll(c, 1), 1e-8)).fillna(0).values

    # 14-16: RSI
    diff_c = np.diff(c, prepend=c[0])
    gain = np.maximum(diff_c, 0); loss_ = np.maximum(-diff_c, 0)
    for w in [7, 14, 28]:
        ag = pd.Series(gain).rolling(w).mean().fillna(0).values
        al = pd.Series(loss_).rolling(w).mean().fillna(0).values
        feats[f"rsi_{w}"] = 100 - 100 / (1 + ag / np.maximum(al, 1e-8))

    # 17-18: 布林带位置
    for w in [20, 50]:
        ma = pd.Series(c).rolling(w).mean()
        std = pd.Series(c).rolling(w).std()
        feats[f"bb_{w}"] = ((c - ma) / np.maximum(std, 1e-8)).fillna(0).values

    # 19-20: MACD
    e12 = pd.Series(c).ewm(span=26, adjust=False).mean()
    e26 = pd.Series(c).ewm(span=52, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=18, adjust=False).mean()
    feats["macd"] = (macd - sig).values
    feats["macd_hist"] = (macd - sig).diff().fillna(0).values

    # 21-23: ATR
    tr_arr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)),
                                           np.abs(l - np.roll(c, 1))))
    for w in [5, 14, 21]:
        feats[f"atr_{w}"] = (pd.Series(tr_arr).rolling(w).mean() / np.maximum(c, 1e-8)).fillna(0).values

    # 24-26: 动量
    for lag in [2, 4, 8]:
        feats[f"mom_{lag}"] = (pd.Series(c).diff(lag) / np.maximum(c, 1e-8)).fillna(0).values

    # 27-28: 价格位置 + ema交叉
    feats["pos_hl"] = (c - l) / np.maximum(h - l, 1e-8)
    e5 = pd.Series(c).ewm(span=10, adjust=False).mean()
    e40 = pd.Series(c).ewm(span=40, adjust=False).mean()
    feats["ema_cross"] = ((e5 - e40) / np.maximum(e40, 1e-8)).values

    # 29-30: 量比 + 距离MA
    feats["vratio"] = (v / np.maximum(pd.Series(v).rolling(10).mean(), 1e-8)).fillna(0).values
    ma60 = pd.Series(c).rolling(60).mean()
    feats["dma_60"] = ((c - ma60) / np.maximum(ma60, 1e-8)).fillna(0).values

    result = np.column_stack(list(feats.values())).astype(np.float32)
    np.nan_to_num(result, nan=0.0, copy=False)
    result = np.clip(result, -10, 10)
    return result, list(feats.keys())

# ========== 平衡 Triple Barrier 标签 ==========
def generate_balanced_labels(df, tp_atr, sl_atr, max_hold):
    """平衡版 Triple Barrier：确保 hold/long/short 三类均衡"""
    n = len(df)
    c = df["close"].values.astype(np.float64)
    hi = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)

    tr = np.maximum(hi - lo, np.maximum(
        np.abs(hi - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
    atr_v = pd.Series(tr).rolling(14).mean().fillna(tr[14] if n > 14 else tr[-1]).values
    av = np.maximum(atr_v, c * 0.0005)

    ut = c + tp_atr * av
    ls = c - sl_atr * av
    us = c + sl_atr * av
    lt = c - tp_atr * av

    actions = np.zeros(n, dtype=np.int32)
    chunk = 5000
    H = max_hold
    log(f"  Labeling {n:,} bars (H={H}, TP={tp_atr}ATR, SL={sl_atr}ATR)...")

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
            if lta[i] and (not lsa[i] or ltf[i] < lsf[i]):
                actions[t] = 1
            elif sta[i] and (not ssa[i] or stf[i] < ssf[i]):
                actions[t] = 2
            else:
                actions[t] = 0

    dist = np.bincount(actions[:n-H], minlength=3)
    log(f"  Done: hold={dist[0]} ({dist[0]/(n-H)*100:.0f}%) "
        f"long={dist[1]} ({dist[1]/(n-H)*100:.0f}%) "
        f"short={dist[2]} ({dist[2]/(n-H)*100:.0f}%)")
    return actions


# ========== 阶段A：自监督预训练 ==========
def train_stage_A(
    market_features, position_states, targets, *,
    val_ratio=0.1, epochs=60, batch_size=20, lr=0.001, patience=20,
    hidden_dim=128, num_layers=2, embed_dim=64, device="cpu", sequence_length=64
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
    KnCls, _ = _build_trader_gru_class(
        hidden_dim=hidden_dim, num_layers=num_layers,
        embed_dim=embed_dim, num_actions=3
    )
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
            last_hidden = gru_out[-1]
            pred_ret = model.embed_head(last_hidden).mean(dim=-1)
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
            meta = {"hidden_dim": hidden_dim, "num_layers": num_layers,
                    "embed_dim": embed_dim, "num_actions": 3,
                    "market_dim": 98, "pos_dim": 6, "val_loss": avg_val}
            OUT_PATH.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        else:
            stale += 1
            if stale >= patience:
                log(f"  Early stop at epoch {ep+1}")
                break
    return {"val_loss": best_loss, "model_path": str(OUT_PATH)}


# ========== 阶段B：冻结GRU + 训练交易头 ==========
def train_stage_B(
    market_features, position_states, targets, *,
    val_ratio=0.1, epochs=60, batch_size=20, lr=0.001, patience=20,
    hidden_dim=128, num_layers=2, embed_dim=64, device="cpu", sequence_length=64
):
    torch, nn = __import__('zhulong.agent.knowledge_net_kn2', fromlist=['_ensure_torch'])._ensure_torch()
    device_obj = torch.device("cpu")

    from zhulong.agent.knowledge_net_kn2 import _build_trader_gru_class
    KnCls, _ = _build_trader_gru_class(
        hidden_dim=hidden_dim, num_layers=num_layers,
        embed_dim=embed_dim, num_actions=3
    )
    model = KnCls().to(device_obj)

    if OUT_PATH.exists():
        state = torch.load(OUT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        log("  Loaded Stage A weights")
    else:
        log("  WARNING: No pretrained weights found, training from scratch")

    for p in model.market_encoder.parameters(): p.requires_grad = False
    for p in model.pos_encoder.parameters(): p.requires_grad = False
    for p in model.gru.parameters(): p.requires_grad = False

    action_loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        list(model.action_head.parameters()) +
        list(model.size_head.parameters()) +
        list(model.sl_head.parameters()) +
        list(model.tp_head.parameters()) +
        list(model.trade_head.parameters()) +
        list(model.conf_head.parameters()),
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
    log(f"  Stage B: {len(train_seqs)} train seqs / {len(val_seqs)} val seqs")

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
            act_logits = model.action_head(gru_out)
            loss = torch.tensor(0.0, device=device_obj)
            for i, (s, e) in enumerate(batch_seqs):
                sl = e - s
                ta = torch.tensor(targets["action"][s:e], dtype=torch.long, device=device_obj)
                loss = loss + action_loss_fn(act_logits[:sl, i], ta)
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
                    ta = torch.tensor(targets["action"][s:e], dtype=torch.long, device=device_obj)
                    vloss = vloss + action_loss_fn(act_logits[:sl, i], ta)
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
    meta = {"hidden_dim": hidden_dim, "num_layers": num_layers,
            "embed_dim": embed_dim, "num_actions": 3,
            "market_dim": 98, "pos_dim": 6, "val_loss": best_loss}
    OUT_PATH.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return {"val_loss": best_loss}


# ========== 阶段C：联合微调 ==========
def train_stage_C(
    market_features, position_states, targets, *,
    val_ratio=0.1, epochs=25, batch_size=20, lr=0.0001, patience=10,
    hidden_dim=128, num_layers=2, embed_dim=64, device="cpu", sequence_length=64
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


# ========== 评估 ==========
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=r"C:\Users\xiaomi\Desktop\XTIUSD5.csv")
    parser.add_argument("--tp", type=float, default=TP_ATR)
    parser.add_argument("--sl", type=float, default=SL_ATR)
    parser.add_argument("--hold", type=int, default=MAX_HOLD)
    args = parser.parse_args()

    log("=" * 60)
    log("KN 2.0 原油课程学习训练方案")
    log(f"  阶段A: 自监督预训练（预测未来收益）")
    log(f"  阶段B: 冻结GRU + 训练交易头")
    log(f"  阶段C: 联合微调")
    log(f"  特征: {HIDDEN_DIM}d GRU x{NUM_LAYERS} | 标签: TP={args.tp}ATR SL={args.sl}ATR H={args.hold}")
    log(f"  输出: {OUT_PATH}")
    log("=" * 60)

    t0 = time.perf_counter()

    # ========== 数据加载 ==========
    log("\n━━━ 数据准备 ━━━")
    log("加载 CSV...")
    # 原油 CSV 7列: date, time, open, high, low, close, volume
    df = pd.read_csv(args.csv, header=None,
                     names=["date", "time", "open", "high", "low", "close", "volume"])
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df = df.sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year
    log(f"  {len(df):,} bars loaded")

    log("计算特征...")
    t_feat = time.perf_counter()
    feats, feat_names = build_features_rich(df)
    feats_pad = np.pad(feats, ((0, 0), (0, max(0, 98 - feats.shape[1]))))[:, :98].astype(np.float32)
    feats_pad = (feats_pad - feats_pad.mean(axis=0)) / (feats_pad.std(axis=0) + 1e-8)
    feats_pad = np.clip(feats_pad, -5, 5)
    log(f"  特征: {feats_pad.shape} | 耗时: {time.perf_counter()-t_feat:.1f}s")
    log(f"  dims with std>0.01: {np.sum(np.std(feats_pad, axis=0) > 0.01)}/{feats_pad.shape[1]}")

    log("生成平衡标签...")
    t_lbl = time.perf_counter()
    labels = generate_balanced_labels(df, args.tp, args.sl, args.hold)
    log(f"  标签耗时: {time.perf_counter()-t_lbl:.1f}s")

    # 分割
    tr_mask = df["year"].isin(TRAIN_YEARS).values
    vl_mask = (df["year"] == VAL_YEAR).values
    H = args.hold

    tr_mf = feats_pad[tr_mask][:-H]
    vl_mf = feats_pad[vl_mask][:-H]
    tr_label = labels[tr_mask][:-H]
    vl_label = labels[vl_mask][:-H]
    pos_tr = np.zeros((len(tr_mf), 6), dtype=np.float32)

    c = df["close"].values.astype(np.float64)
    n = len(c)
    fwd_ret = np.zeros(n, dtype=np.float32)
    fwd_ret[:n-H] = (c[H:] - c[:-H]) / np.maximum(c[:-H], 1e-8)

    train_targets = {
        "action": tr_label,
        "fwd_ret": fwd_ret[tr_mask][:-H],
        "position_size": np.ones(len(tr_label), dtype=np.float32),
        "sl_atr_mult": np.full(len(tr_label), args.sl, dtype=np.float32),
        "tp_atr_mult": np.full(len(tr_label), args.tp, dtype=np.float32),
        "should_trade": (tr_label > 0).astype(np.float32),
    }

    tr_dist = np.bincount(tr_label, minlength=3)
    vl_dist = np.bincount(vl_label, minlength=3)
    log(f"  Train: {len(tr_mf):,} bars (h={tr_dist[0]} l={tr_dist[1]} s={tr_dist[2]})")
    log(f"  Val: {len(vl_mf):,} bars (h={vl_dist[0]} l={vl_dist[1]} s={vl_dist[2]})")

    # ========== 阶段A：自监督预训练 ==========
    log(f"\n━━━ 阶段A：自监督预训练（{60} epochs）━━━")
    stage_a = train_stage_A(
        market_features=tr_mf, position_states=pos_tr,
        targets=train_targets,
        val_ratio=0.1, epochs=60, batch_size=BATCH_SEQ, lr=0.0005, patience=35,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
        device="cpu", sequence_length=SEQ_LEN,
    )
    log(f"  Stage A best val_loss: {stage_a['val_loss']:.6f}")

    # ========== 阶段B：冻结GRU训练交易头 ==========
    log(f"\n━━━ 阶段B：冻结GRU + 训练交易头（{80} epochs）━━━")
    stage_b = train_stage_B(
        market_features=tr_mf, position_states=pos_tr,
        targets=train_targets,
        val_ratio=0.1, epochs=80, batch_size=BATCH_SEQ, lr=0.0005, patience=45,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
        device="cpu", sequence_length=SEQ_LEN,
    )

    # ========== 阶段C：联合微调 ==========
    log(f"\n━━━ 阶段C：联合微调（{60} epochs）━━━")
    stage_c = train_stage_C(
        market_features=tr_mf, position_states=pos_tr,
        targets=train_targets,
        val_ratio=0.1, epochs=60, batch_size=BATCH_SEQ, lr=0.00005, patience=35,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
        device="cpu", sequence_length=SEQ_LEN,
    )

    # ========== 最终评估 ==========
    log(f"\n━━━ 最终评估 ━━━")
    ev = evaluate_model(OUT_PATH, vl_mf, vl_label)

    total_time = time.perf_counter() - t0
    status = "PASSED" if ev["pass"] else "NOT PASSED"
    log(f"\n{'='*60}")
    log(f"  {status}")
    log(f"  Model: {OUT_PATH}")
    log(f"  Accuracy: {ev['acc']*100:.1f}%")
    log(f"  Total time: {total_time:.0f}s ({total_time/60:.0f}m)")
    log("=" * 60)

    return 0 if ev["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
