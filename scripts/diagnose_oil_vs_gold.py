"""诊断：为什么黄金KN2能学到，原油不行"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ========== 特征构建 ==========
def build_features(df):
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
    macd = e12 - e26; sig = macd.ewm(span=18, adjust=False).mean()
    feats["macd"] = (macd - sig).values
    feats["macd_hist"] = (macd - sig).diff().fillna(0).values
    tr_arr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
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
    np.nan_to_num(result, nan=0.0, copy=False); result = np.clip(result, -10, 10)
    return result, list(feats.keys())

# ========== 加载数据 ==========
def load_csv(path):
    df = pd.read_csv(path, header=None, names=["date","time","open","high","low","close","volume"])
    df = df.dropna(subset=["open","high","low","close"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
    df = df.sort_values("datetime").reset_index(drop=True)
    df["year"] = df["datetime"].dt.year
    return df

df_gold = load_csv(r"C:\Users\xiaomi\Desktop\XAUUSD5.csv")
df_oil  = load_csv(r"C:\Users\xiaomi\Desktop\XTIUSD5.csv")

print("=" * 70)
print("DIAGNOSTIC: Gold vs Oil — Why KN2 learns gold but not oil")
print("=" * 70)

# ===== 1. Statistical properties =====
print("\n--- 1. Statistical Properties ---")
for name, df in [("Gold", df_gold), ("Oil", df_oil)]:
    c = df["close"].values
    ret = np.diff(np.log(np.maximum(c, 1e-8)))
    print(f"\n  {name} ({len(df):,} bars):")
    print(f"    Mean log-ret: {ret.mean():.8f}")
    print(f"    Std log-ret:  {ret.std():.6f}")
    for lag in [1, 5, 20, 50]:
        ac = np.corrcoef(ret[lag:], ret[:-lag])[0,1]
        print(f"    AC({lag:2d}):      {ac:.4f}")
    print(f"    Kurtosis:     {pd.Series(ret).kurtosis():.2f}")
    vol20 = pd.Series(ret).rolling(20).std()
    print(f"    Vol(20) mean: {vol20.mean():.6f}")

# ===== 2. Forward return signal analysis =====
print("\n\n--- 2. Forward Return Signal Analysis ---")
for name, df in [("Gold", df_gold), ("Oil", df_oil)]:
    c = df["close"].values.astype(np.float64); n = len(c)
    H = 128
    fwd_ret = np.zeros(n, dtype=np.float64)
    fwd_ret[:n-H] = (c[H:] - c[:-H]) / np.maximum(c[:-H], 1e-8)
    ret1 = pd.Series(c).pct_change(1).fillna(0)
    vol20 = ret1.rolling(20).std().fillna(ret1.std()).values
    z = fwd_ret[:n-H] / np.maximum(vol20[:n-H], 1e-8)
    
    print(f"\n  {name} (H={H}):")
    print(f"    fwd_ret mean:     {fwd_ret[:n-H].mean():.6f}")
    print(f"    fwd_ret std:      {fwd_ret[:n-H].std():.6f}")
    print(f"    SNR:              {abs(fwd_ret[:n-H].mean()) / fwd_ret[:n-H].std():.4f}")
    print(f"    z-score mean:     {z.mean():.4f}")
    print(f"    z-score std:      {z.std():.4f}")
    print(f"    P(fwd_ret > 0):   {np.mean(fwd_ret[:n-H] > 0):.4f}")
    print(f"    P(fwd_ret > 1%):  {np.mean(fwd_ret[:n-H] > 0.01):.4f}")
    print(f"    P(fwd_ret < -1%): {np.mean(fwd_ret[:n-H] < -0.01):.4f}")

# ===== 3. Trend persistence vs mean reversion =====
print("\n\n--- 3. Trend Persistence vs Mean Reversion ---")
for name, df in [("Gold", df_gold), ("Oil", df_oil)]:
    c = df["close"].values.astype(np.float64); n = len(c)
    print(f"\n  {name}:")
    for H in [128]:
        fwd = np.zeros(n); fwd[:n-H] = (c[H:]-c[:-H])/np.maximum(c[:-H],1e-8)
        for L in [1, 5, 10, 20, 50]:
            past = np.zeros(n)
            past[L:] = (c[L:]-c[:-L])/np.maximum(c[:-L],1e-8)
            valid = np.where((np.arange(n) >= max(L,H)) & (np.arange(n) < n-H))[0]
            if len(valid) < 100: continue
            p_same = np.mean((past[valid] > 0) == (fwd[valid] > 0))
            # Conditional: P(fwd>0 | past>0)
            cond_up = fwd[valid][past[valid] > 0]
            cond_dn = fwd[valid][past[valid] < 0]
            p_up_given_up = np.mean(cond_up > 0) if len(cond_up) > 0 else 0
            p_down_given_down = np.mean(cond_dn < 0) if len(cond_dn) > 0 else 0
            print(f"    Lag={L:3d}: same_sign={p_same:.3f} P(up|up)={p_up_given_up:.3f} P(dn|dn)={p_down_given_down:.3f}")

# ===== 4. Triple Barrier label distribution =====
print("\n\n--- 4. Triple Barrier Labels (TP=4.0/3.0 ATR, H=128) ---")
for name, df in [("Gold", df_gold), ("Oil", df_oil)]:
    c = df["close"].values.astype(np.float64); n = len(c)
    hi = df["high"].values.astype(np.float64); lo = df["low"].values.astype(np.float64)
    tr = np.maximum(hi-lo, np.maximum(np.abs(hi-np.roll(c,1)), np.abs(lo-np.roll(c,1))))
    atr = pd.Series(tr).rolling(14).mean().fillna(tr[14] if n>14 else tr[-1]).values
    av = np.maximum(atr, c*0.0005)
    H = 128; tp, sl = 4.0, 3.0
    ut = c + tp*av; ls = c - sl*av; us = c + sl*av; lt = c - tp*av
    actions = np.zeros(n, dtype=np.int32)
    chunk = 5000
    for s in range(0, n-H, chunk):
        e = min(s+chunk, n-H); m = e-s
        idx = np.clip(np.arange(s,s+m)[:,None] + np.arange(1,H+1), 0, n-1)
        ltp = hi[idx] >= ut[s:e,None]; lsl = lo[idx] <= ls[s:e,None]
        ltf = np.argmax(ltp,1); lsf = np.argmax(lsl,1)
        lta = ltp.any(1); lsa = lsl.any(1)
        stp = lo[idx] <= lt[s:e,None]; ssl = hi[idx] >= us[s:e,None]
        stf = np.argmax(stp,1); ssf = np.argmax(ssl,1)
        sta = stp.any(1); ssa = ssl.any(1)
        for i in range(m):
            t=s+i
            if lta[i] and (not lsa[i] or ltf[i] < lsf[i]): actions[t]=1
            elif sta[i] and (not ssa[i] or stf[i] < ssf[i]): actions[t]=2
    dist = np.bincount(actions[:n-H], minlength=3)
    s = dist.sum()
    print(f"  {name}: hold={dist[0]/s*100:.0f}% long={dist[1]/s*100:.0f}% short={dist[2]/s*100:.0f}%")

# ===== 5. Auto-correlation of features (oil-specific) =====
print("\n\n--- 5. Feature Auto-Correlation (Oil only, critical for GRU) ---")
c = df_oil["close"].values.astype(np.float64); n = len(c)
ret = np.diff(np.log(np.maximum(c, 1e-8)))
for lag in [1, 2, 3, 5, 10, 20, 50]:
    ac = np.corrcoef(ret[lag:], ret[:-lag])[0,1]
    bar = "+" * int(abs(ac)*200) if abs(ac) > 0.005 else ""
    print(f"  Oil log-ret AC({lag:2d}) = {ac:+.4f} {bar}")

# ===== 6. Feature auto-correlation comparison =====
print("\n\n--- 6. Feature cross-correlation with forward return ---")
feat_names = ["ret_1", "ret_5", "ret_20", "rsi_14", "macd", "bb_20", "mom_2", "mom_8"]
for name, df in [("Gold", df_gold), ("Oil", df_oil)]:
    feats, fnames = build_features(df)
    c = df["close"].values.astype(np.float64); n = len(c)
    H = 128
    fwd = np.zeros(n); fwd[:n-H] = (c[H:]-c[:-H])/np.maximum(c[:-H],1e-8)
    print(f"\n  {name}:")
    for fn in feat_names:
        if fn in fnames:
            idx = fnames.index(fn)
            corr = np.corrcoef(feats[:n-H, idx], fwd[:n-H])[0,1]
            print(f"    corr({fn:10s}, fwd_{H}) = {corr:+.4f}")

# ===== 7. XGBoost quick test =====
print("\n\n--- 7. XGBoost Baseline ---")
try:
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.decomposition import PCA
    
    H = 128
    for name, df in [("Gold", df_gold), ("Oil", df_oil)]:
        feats, _ = build_features(df)
        c = df["close"].values.astype(np.float64); n = len(c)
        
        # PCA to 30 dims
        pca = PCA(n_components=min(30, feats.shape[1]))
        X_all = pca.fit_transform((feats - feats.mean(0)) / (feats.std(0) + 1e-8))
        
        # Labels: sigmoid soft label argmax
        fwd_ret = np.zeros(n); fwd_ret[:n-H] = (c[H:]-c[:-H])/np.maximum(c[:-H],1e-8)
        ret1 = pd.Series(c).pct_change(1).fillna(0)
        vol20 = ret1.rolling(20).std().fillna(ret1.std()).values
        z = fwd_ret[:n-H] / np.maximum(vol20[:n-H], 1e-8)
        logits = np.column_stack([np.full(n-H,4.0), z, -z])
        probs = np.exp(logits - logits.max(1,keepdims=True))
        probs /= probs.sum(1,keepdims=True)
        y = np.argmax(probs, 1)
        
        train_mask = df["year"].isin([2020,2021,2022,2023,2024]).values[:n-H]
        test_mask = (df["year"] == 2025).values[:n-H]
        X_tr = X_all[:n-H][train_mask]
        X_te = X_all[:n-H][test_mask]
        y_tr = y[train_mask]
        y_te = y[test_mask]
        
        clf = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, verbosity=0, n_jobs=4)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        acc = accuracy_score(y_te, pred)
        cm = confusion_matrix(y_te, pred, labels=[0,1,2])
        pr = [cm[i,i]/max(cm[:,i].sum(),1) for i in range(3)]
        pd = np.bincount(pred, minlength=3) / len(pred)
        print(f"\n  {name}: Acc={acc*100:.2f}% | Prec=[{pr[0]*100:.0f}/{pr[1]*100:.0f}/{pr[2]*100:.0f}]% "
              f"| Dist=[{pd[0]*100:.0f}/{pd[1]*100:.0f}/{pd[2]*100:.0f}]%")
except ImportError:
    print("  XGBoost not available")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
