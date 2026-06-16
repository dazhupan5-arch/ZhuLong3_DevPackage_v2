#!/usr/bin/env python3
"""验证 6/10：V14/KN1 对「未来12根跌0.2%」标签的预测是否生效。"""
from __future__ import annotations

import json
import os
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

os.environ.setdefault("ZHULONG_IMF_CSV_ONLY", "1")
os.environ.setdefault("ZHULONG_INSTALL_DIR", str(INSTALL))
for p in (INSTALL, INSTALL / "ZhuLong.PythonEngine"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from zhulong.utils.win_dll import configure_native_dll_paths

configure_native_dll_paths()
import onnxruntime  # noqa: F401

from zhulong.agent.knowledge_net import KnowledgeNetInference, resolve_knowledge_paths
from zhulong.agent.trading_agent import TradingAgent
from zhulong.v14_live import (
    _proba_to_direction_v14,
    build_live_v14_features,
    load_v14_bundle,
    predict_v14,
)

CFG = Path.home() / "AppData/Roaming/ZhuLong/config_agent.json"
cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))

df = pd.read_csv(CSV)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# 全量 close 用于 ground truth
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
kn_path, kn_scaler = resolve_knowledge_paths("XAUUSD", cfg, INSTALL)
kn = KnowledgeNetInference(kn_path, scaler_path=kn_scaler)
agent = TradingAgent(config=cfg, root=str(INSTALL))

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

    feat_row, _, _, feats_df = build_live_v14_features(m5=m5)
    proba_v14 = v14["model"].predict_proba(
        np.asarray(feat_row[: len(v14["columns"])], dtype=np.float32).reshape(1, -1)
    )[0]
    # V14 cols: 0=flat, 1=long, 2=short
    v14_flat, v14_long, v14_short = map(float, proba_v14)
    v14_argmax = ["flat", "long", "short"][int(np.argmax(proba_v14))]
    v14_dir_int = _proba_to_direction_v14(proba_v14.reshape(1, -1), LONG_THR, SHORT_THR)[0]
    v14_side = {1: "long", -1: "short", 0: "flat"}[v14_dir_int]

    feat68 = agent._build_v14_features(m5)
    kn_probs, _ = kn.predict(feat68.reshape(1, -1))
    kn_s, kn_f, kn_l = map(float, kn_probs[0])
    kn_argmax = ["short", "flat", "long"][int(kn_probs[0].argmax())]

    rows.append({
        "t": dt.strftime("%H:%M"),
        "close": cp,
        "fut12_ret_pct": fut_ret * 100,
        "gt": gt_name,
        "v14_f": v14_flat,
        "v14_l": v14_long,
        "v14_s": v14_short,
        "v14_argmax": v14_argmax,
        "v14_thr": v14_side,
        "kn_argmax": kn_argmax,
        "kn_s": kn_s,
        "kn_l": kn_l,
    })

rdf = pd.DataFrame(rows)
n = len(rdf)
gt_short = rdf[rdf["gt"] == "short"]
gt_long = rdf[rdf["gt"] == "long"]
gt_flat = rdf[rdf["gt"] == "flat"]

print("=" * 92)
print(f"  V14 label verification — {TARGET}  (horizon={HORIZON}, gain={GAIN*100:.2f}%)")
print(f"  Evaluable bars: {n}  (need {WINDOW} history + {HORIZON} future)")
print("=" * 92)

print("\n--- Ground truth (next 12 bars) ---")
for name, sub in [("short", gt_short), ("flat", gt_flat), ("long", gt_long)]:
    print(f"  {name:5s}: {len(sub):3d}  ({100*len(sub)/n:.1f}%)")

print("\n--- V14 raw argmax (no threshold) ---")
print(rdf["v14_argmax"].value_counts().to_string())
print("\n--- V14 with thr=0.70 ---")
print(rdf["v14_thr"].value_counts().to_string())
print("\n--- KN1 argmax ---")
print(rdf["kn_argmax"].value_counts().to_string())

# When GT=short, does model predict short?
if len(gt_short):
    v14_arg_short = (gt_short["v14_argmax"] == "short").mean()
    v14_thr_short = (gt_short["v14_thr"] == "short").mean()
    kn_short = (gt_short["kn_argmax"] == "short").mean()
    v14_s_gt_l = (gt_short["v14_s"] > gt_short["v14_l"]).mean()
    kn_s_gt_l = (gt_short["kn_s"] > gt_short["kn_l"]).mean()
    print(f"\n--- When ground truth = SHORT ({len(gt_short)} bars) ---")
    print(f"  V14 argmax=short:     {v14_arg_short:.1%}")
    print(f"  V14 thr signal=short: {v14_thr_short:.1%}")
    print(f"  V14 short_p > long_p: {v14_s_gt_l:.1%}  (mean short_p={gt_short['v14_s'].mean():.3f} long_p={gt_short['v14_l'].mean():.3f})")
    print(f"  KN  argmax=short:     {kn_short:.1%}")
    print(f"  KN  short_p > long_p: {kn_s_gt_l:.1%}  (mean short_p={gt_short['kn_s'].mean():.3f} long_p={gt_short['kn_l'].mean():.3f})")

if len(gt_long):
    print(f"\n--- When ground truth = LONG ({len(gt_long)} bars) ---")
    print(f"  V14 argmax=long: {(gt_long['v14_argmax']=='long').mean():.1%}")
    print(f"  KN  argmax=long: {(gt_long['kn_argmax']=='long').mean():.1%}")

# Overall alignment with GT
v14_arg_acc = (rdf["v14_argmax"].map({"short":"short","long":"long","flat":"flat"}) == rdf["gt"]).mean()
kn_acc = (rdf["kn_argmax"] == rdf["gt"]).mean()
print(f"\n--- Overall argmax accuracy vs GT ---")
print(f"  V14 argmax: {v14_arg_acc:.1%}")
print(f"  KN1 argmax: {kn_acc:.1%}")

# Trade-only: exclude flat GT
trade = rdf[rdf["gt"] != "flat"]
if len(trade):
    v14_trade = (trade.apply(lambda r: r["v14_argmax"] == r["gt"], axis=1)).mean()
    kn_trade = (trade["kn_argmax"] == trade["gt"]).mean()
    print(f"  V14 argmax (non-flat GT only): {v14_trade:.1%}  ({len(trade)} bars)")
    print(f"  KN1 argmax (non-flat GT only): {kn_trade:.1%}")

print("\n--- Hourly samples (GT vs V14 vs KN) ---")
for _, r in rdf.iloc[::24].iterrows():
    print(
        f"  {r['t']} C={r['close']:.0f} fut12={r['fut12_ret_pct']:+.2f}% GT={r['gt']:5s} "
        f"V14=[{r['v14_f']:.2f},{r['v14_l']:.2f},{r['v14_s']:.2f}]→{r['v14_argmax']}/{r['v14_thr']} "
        f"KN→{r['kn_argmax']}"
    )
print("=" * 92)
