#!/usr/bin/env python3
"""验证 6/10：V14 对「未来12根跌0.2%」是否预测正确（仅 XGBoost，无 ONNX）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

INSTALL = Path(r"C:\Program Files\ZhuLong")
CSV = Path(__file__).resolve().parent / "_june_multi_bars.csv"
TARGET = "2026-06-10"
HORIZON = 12
GAIN = 0.002
WINDOW = 256
LONG_THR = SHORT_THR = 0.70

sys.path.insert(0, str(INSTALL))
from zhulong.v14_live import build_live_v14_features, load_v14_bundle

df = pd.read_csv(CSV)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

close_all = df["close"].values.astype(np.float64)
gt_all = np.zeros(len(close_all), dtype=np.int8)
for i in range(len(close_all) - HORIZON):
    ret = (close_all[i + HORIZON] - close_all[i]) / max(close_all[i], 1e-9)
    if ret > GAIN:
        gt_all[i] = 1
    elif ret < -GAIN:
        gt_all[i] = -1

day_idx = df[df["datetime"].dt.strftime("%Y-%m-%d") == TARGET].index.tolist()
v14 = load_v14_bundle("XAUUSD", root=INSTALL)
model = v14["model"]
cols = v14["columns"]

rows = []
for idx in day_idx:
    if idx < WINDOW or idx + HORIZON >= len(df):
        continue
    seg = df.iloc[idx - WINDOW : idx].copy()
    dt = seg["datetime"].iloc[-1]
    cp = float(seg["close"].iloc[-1])
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

    gt = int(gt_all[idx])
    gt_name = {1: "long", -1: "short", 0: "flat"}[gt]
    fut_ret = (close_all[idx + HORIZON] - close_all[idx]) / close_all[idx]

    feat_row, _, _, _ = build_live_v14_features(m5=m5)
    proba = model.predict_proba(
        np.asarray(feat_row[: len(cols)], dtype=np.float32).reshape(1, -1)
    )[0]
    vf, vl, vs = map(float, proba)
    argmax = ["flat", "long", "short"][int(np.argmax(proba))]
    if vl >= LONG_THR and vl >= vs and vl > vf:
        thr = "long"
    elif vs >= SHORT_THR and vs >= vl and vs > vf:
        thr = "short"
    else:
        thr = "flat"

    rows.append({
        "t": dt.strftime("%H:%M"),
        "close": cp,
        "fut12": fut_ret * 100,
        "gt": gt_name,
        "vf": vf, "vl": vl, "vs": vs,
        "argmax": argmax, "thr": thr,
    })

rdf = pd.DataFrame(rows)
n = len(rdf)
gs = rdf[rdf["gt"] == "short"]
gl = rdf[rdf["gt"] == "long"]
gf = rdf[rdf["gt"] == "flat"]

print("=" * 92)
print(f"  V14 vs ground truth — {TARGET}  h={HORIZON} gain={GAIN*100:.2f}%  thr={LONG_THR}")
print(f"  Bars: {n}")
print("=" * 92)
print(f"\nGround truth: short={len(gs)} ({100*len(gs)/n:.0f}%)  flat={len(gf)}  long={len(gl)}")
print(f"V14 argmax:   {rdf['argmax'].value_counts().to_dict()}")
print(f"V14 thr0.70:  {rdf['thr'].value_counts().to_dict()}")

if len(gs):
    print(f"\nWhen GT=SHORT ({len(gs)} bars) — should predict short:")
    print(f"  argmax=short:  {(gs['argmax']=='short').mean():.1%}")
    print(f"  thr=short:     {(gs['thr']=='short').mean():.1%}")
    print(f"  short_p>long_p: {(gs['vs']>gs['vl']).mean():.1%}")
    print(f"  mean proba: flat={gs['vf'].mean():.3f} long={gs['vl'].mean():.3f} short={gs['vs'].mean():.3f}")
    print(f"  max short_p:   {gs['vs'].max():.3f}  (thr needs {SHORT_THR})")

trade = rdf[rdf["gt"] != "flat"]
print(f"\nNon-flat GT accuracy (argmax): {(trade['argmax']==trade['gt']).mean():.1%} on {len(trade)} bars")
print(f"Overall argmax accuracy:       {(rdf['argmax']==rdf['gt']).mean():.1%}")

print("\nSamples every 2h:")
for _, r in rdf.iloc[::24].iterrows():
    print(f"  {r['t']} C={r['close']:.0f} fut12={r['fut12']:+.2f}% GT={r['gt']:5s} "
          f"p=[{r['vf']:.2f},{r['vl']:.2f},{r['vs']:.2f}] → {r['argmax']}/{r['thr']}")
print("=" * 92)
