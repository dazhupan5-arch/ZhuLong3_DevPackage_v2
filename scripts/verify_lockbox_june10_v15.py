#!/usr/bin/env python3
"""Lockbox：6/10 V15 vs ground truth（不参与训练）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INSTALL = Path(r"C:\Program Files\ZhuLong")
ROOT = Path(__file__).resolve().parent.parent
CSV = Path(__file__).resolve().parent / "_june_multi_bars.csv"
sys.path.insert(0, str(ROOT))

from zhulong.training.lgb.features_v15 import FEATURE_COLUMNS_V15, compute_features_v15
from zhulong.training.lgb.labels_v15 import V15_HORIZON
from zhulong.v14_live import load_v15_bundle

HORIZON = V15_HORIZON
GAIN = 0.002
WINDOW = 2000

df = pd.read_csv(CSV)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
day_idx = df[df["datetime"].dt.strftime("%Y-%m-%d") == "2026-06-10"].index.tolist()

bundle = load_v15_bundle("XAUUSD", root=ROOT)
model = bundle["model"]
cols = bundle["columns"]
lthr = bundle["long_thr"]
sthr = bundle["short_thr"]

close_all = df["close"].values.astype(np.float64)
gt_all = np.zeros(len(close_all), dtype=np.int8)
for i in range(len(close_all) - HORIZON):
    ret = (close_all[i + HORIZON] - close_all[i]) / max(close_all[i], 1e-9)
    if ret > GAIN:
        gt_all[i] = 1
    elif ret < -GAIN:
        gt_all[i] = -1

rows = []
for idx in day_idx:
    if idx < WINDOW or idx + HORIZON >= len(df):
        continue
    seg = df.iloc[max(0, idx - WINDOW) : idx]
    dt = seg["datetime"].iloc[-1]
    idx_utc = pd.DatetimeIndex(seg["datetime"], tz="UTC")
    m5 = pd.DataFrame(
        {
            "open": seg["open"].values,
            "high": seg["high"].values,
            "low": seg["low"].values,
            "close": seg["close"].values,
            "volume": seg["volume"].fillna(0).values,
        },
        index=idx_utc,
    )
    feat = compute_features_v15(m5)
    if feat.empty:
        continue
    x = feat.iloc[-1][cols].values.astype(np.float32).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    vf, vl, vs = map(float, proba)
    argmax = ["flat", "long", "short"][int(np.argmax(proba))]
    if vl >= lthr and vl >= vs and vl > vf:
        thr = "long"
    elif vs >= sthr and vs >= vl and vs > vf:
        thr = "short"
    else:
        thr = "flat"
    gt = {1: "long", -1: "short", 0: "flat"}[int(gt_all[idx])]
    rows.append({"t": dt.strftime("%H:%M"), "gt": gt, "argmax": argmax, "thr": thr, "vs": vs, "vl": vl})

rdf = pd.DataFrame(rows)
gs = rdf[rdf["gt"] == "short"]
print("=" * 80)
print(f"  V15 lockbox 2026-06-10  thr long={lthr} short={sthr}  bars={len(rdf)}")
print(f"  argmax: {rdf['argmax'].value_counts().to_dict()}")
print(f"  thr:    {rdf['thr'].value_counts().to_dict()}")
if len(gs):
    print(f"  GT=short argmax hit: {(gs['argmax']=='short').mean():.1%}")
    print(f"  GT=short thr=short:  {(gs['thr']=='short').mean():.1%}")
    print(f"  mean vs={gs['vs'].mean():.3f} vl={gs['vl'].mean():.3f}")
print("=" * 80)
