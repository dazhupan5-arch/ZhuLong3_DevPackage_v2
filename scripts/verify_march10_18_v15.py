#!/usr/bin/env python3
"""V15 信号回放：2026-03-10 ~ 03-18（lockbox，不参与训练）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
from zhulong.v14_live import load_v15_bundle

HORIZON = 12
GAIN = 0.002
WINDOW = 2000
START = "2026-03-10"
END = "2026-03-18"


def main() -> int:
    m5_path = ROOT / "data" / "training" / "lgb" / "XAUUSD" / "XAUUSD_M5.csv"
    m5_all = load_vendor_csv(m5_path)
    pad_start = m5_all.index[m5_all.index < pd.Timestamp(START)][-WINDOW:]
    m5 = pd.concat([m5_all.loc[pad_start], m5_all.loc[START:END]]).sort_index()
    m5 = m5[~m5.index.duplicated(keep="last")]

    bundle = load_v15_bundle("XAUUSD", root=ROOT)
    model, cols = bundle["model"], bundle["columns"]
    lthr, sthr = bundle["long_thr"], bundle["short_thr"]

    feats = compute_features(m5, include_mtf=True, include_reversal=True)
    feats = feats.reindex(m5.index).dropna(how="any")
    common = m5.index.intersection(feats.index)
    m5, feats = m5.loc[common], feats.loc[common]

    close = m5["close"].values.astype(np.float64)
    gt = np.zeros(len(close), dtype=np.int8)
    for i in range(len(close) - HORIZON):
        ret = (close[i + HORIZON] - close[i]) / max(close[i], 1e-9)
        if ret > GAIN:
            gt[i] = 1
        elif ret < -GAIN:
            gt[i] = -1

    target_days = pd.date_range(START, END, freq="D")
    print("=" * 95)
    print(f"  V15 replay {START} ~ {END}  thr long={lthr:.2f} short={sthr:.2f}")
    print("=" * 95)
    print(
        f"{'Date':>10s}  {'Chg%':>6s}  {'GT':>18s}  {'argmax':>22s}  {'thr':>22s}  "
        f"{'S_hit':>5s}  {'S_thr':>5s}"
    )
    print("-" * 95)

    all_rows: list[dict] = []
    for day in target_days:
        day_str = day.strftime("%Y-%m-%d")
        mask = m5.index.normalize() == day.normalize()
        if not mask.any():
            continue
        ix = np.where(mask)[0]
        ix = ix[(ix >= WINDOW) & (ix + HORIZON < len(m5))]
        if len(ix) == 0:
            continue

        o = float(m5.iloc[ix[0]]["open"])
        c = float(m5.iloc[ix[-1]]["close"])
        chg = (c - o) / o * 100

        day_thr: list[str] = []
        day_argmax: list[str] = []
        day_gt: list[str] = []
        for idx in ix:
            x = feats.iloc[idx][cols].values.astype(np.float32).reshape(1, -1)
            vf, vl, vs = model.predict_proba(x)[0]
            am = ["flat", "long", "short"][int(np.argmax([vf, vl, vs]))]
            if vl >= lthr and vl >= vs and vl > vf:
                th = "long"
            elif vs >= sthr and vs >= vl and vs > vf:
                th = "short"
            else:
                th = "flat"
            g = {1: "long", -1: "short", 0: "flat"}[int(gt[idx])]
            day_argmax.append(am)
            day_thr.append(th)
            day_gt.append(g)
            all_rows.append({"date": day_str, "gt": g, "argmax": am, "thr": th, "vs": vs, "vl": vl})

        gt_c = pd.Series(day_gt).value_counts().to_dict()
        am_c = pd.Series(day_argmax).value_counts().to_dict()
        th_c = pd.Series(day_thr).value_counts().to_dict()
        gs = [g for g, a, t in zip(day_gt, day_argmax, day_thr) if g == "short"]
        s_hit = sum(1 for g, a in zip(day_gt, day_argmax) if g == "short" and a == "short")
        s_thr = sum(1 for g, t in zip(day_gt, day_thr) if g == "short" and t == "short")
        n_gs = max(sum(1 for g in day_gt if g == "short"), 1)

        def _fmt(d: dict) -> str:
            parts = [f"{k[0]}:{d.get(k, 0)}" for k in ("short", "flat", "long")]
            return " ".join(parts)

        print(
            f"{day_str:>10s}  {chg:+5.2f}  {_fmt({k: gt_c.get(k, 0) for k in ('short','flat','long')}):>18s}  "
            f"{_fmt({k: am_c.get(k, 0) for k in ('short','flat','long')}):>22s}  "
            f"{_fmt({k: th_c.get(k, 0) for k in ('short','flat','long')}):>22s}  "
            f"{100*s_hit/n_gs:4.0f}%  {100*s_thr/n_gs:4.0f}%"
        )

    rdf = pd.DataFrame(all_rows)
    print("-" * 95)
    print(f"  合计 bars={len(rdf)}")
    print(f"  GT:     {rdf['gt'].value_counts().to_dict()}")
    print(f"  argmax: {rdf['argmax'].value_counts().to_dict()}")
    print(f"  thr:    {rdf['thr'].value_counts().to_dict()}")
    gs = rdf[rdf["gt"] == "short"]
    if len(gs):
        print(f"  GT=short → argmax hit: {(gs['argmax']=='short').mean():.1%}")
        print(f"  GT=short → thr=short:  {(gs['thr']=='short').mean():.1%}")
        print(f"  GT=short mean proba: vs={gs['vs'].mean():.3f} vl={gs['vl'].mean():.3f}")

    short_thr_bars = rdf[rdf["thr"] == "short"]
    if len(short_thr_bars):
        print("\n  thr=short 时段（前 15 条）:")
        for _, row in short_thr_bars.head(15).iterrows():
            print(f"    {row['date']}  gt={row['gt']}  vs={row['vs']:.3f} vl={row['vl']:.3f}")
    else:
        print("\n  该区间 thr=short 信号: 0")
    print("=" * 95)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
