#!/usr/bin/env python3
"""P0：KN2 V16 结构位置标签 — 生成 NPZ + 分布报告（GPU 训练前验收）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from zhulong.agent.kn2_location_labels import (
    LocationLabelConfig,
    generate_location_labels,
    replay_bar_diagnosis,
    summarize_location_labels,
)
from zhulong.agent.trading_env_kn2 import generate_kn2_training_labels
from zhulong.agent.training_utils import load_npz
from zhulong.strategies.indicators import atr_series


def _load_frames(npz_path: Path, year_max: int | None = None) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    data = load_npz(npz_path)
    struct = np.asarray(data["struct"], dtype=np.float32)
    n = len(struct)
    times = pd.to_datetime(data["time"], utc=True) if "time" in data else pd.date_range("2020-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": data.get("open", np.zeros(n)),
            "high": data.get("high", np.zeros(n)),
            "low": data.get("low", np.zeros(n)),
            "close": data.get("close", np.zeros(n)),
            "volume": data.get("volume", np.zeros(n)),
        },
        index=times[:n],
    )
    if "atr" in data:
        df["atr"] = np.asarray(data["atr"], dtype=np.float32)
    else:
        df["atr"] = atr_series(df).bfill().fillna(df["close"] * 0.001).values.astype(np.float32)

    if year_max is not None:
        mask = df.index.year <= year_max
        df = df.loc[mask].copy()
        struct = struct[np.asarray(mask)]
        times = df.index
    return df, struct, times


def _synthetic_replay_cases() -> list[dict]:
    """模拟 4339 震荡高位追多 vs 理想低位做多。"""
    cases = []
    bad = np.zeros(30, dtype=np.float32)
    bad[0] = 0.05
    bad[3] = 1.2
    bad[4] = 0.25
    bad[5] = 0.2
    bad[6] = 0.5
    cases.append(
        {
            "name": "replay_ranging_top_chase_long",
            "note": "类似 2026-06-16 4339 震荡区间上沿追多",
            **replay_bar_diagnosis(bad, pos_in_range=0.72, regime_name="ranging"),
        }
    )
    good = np.zeros(30, dtype=np.float32)
    good[0] = 0.08
    good[3] = 0.35
    good[4] = 1.8
    good[5] = 0.55
    good[6] = 0.3
    cases.append(
        {
            "name": "replay_ranging_bottom_long",
            "note": "震荡区间下沿靠近支撑做多",
            **replay_bar_diagnosis(good, pos_in_range=0.28, regime_name="ranging"),
        }
    )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="KN2 V16 结构位置标签 P0")
    parser.add_argument("--npz", default="data/clean/kn2_training_v16.npz")
    parser.add_argument("--out", default="data/clean/kn2_training_v16_location.npz")
    parser.add_argument("--report", default="data/training/reports/kn2_v16/location_label_report.json")
    parser.add_argument("--year-max", type=int, default=0, help="仅处理到该年（0=全部，快速试跑可用 2024）")
    parser.add_argument("--compare-legacy", action="store_true", help="与固定 ATR barrier 标签对比")
    parser.add_argument("--skip-save", action="store_true", help="只出报告不写 NPZ")
    args = parser.parse_args()

    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}，请先 prepare_kn2_v16_data.py")
        return 1

    year_max = args.year_max if args.year_max > 0 else None
    print(f"Loading {npz_path} ...")
    df, struct, times = _load_frames(npz_path, year_max=year_max)
    print(f"bars={len(df):,}  range={times.min()} .. {times.max()}")

    cfg = LocationLabelConfig()
    t0 = time.perf_counter()
    print("Generating location-aware labels ...")
    labels = generate_location_labels(df, struct, cfg=cfg)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.1f}s | should_trade={labels['should_trade'].mean()*100:.2f}%")

    legacy = None
    if args.compare_legacy:
        print("Generating legacy fixed-barrier labels for comparison ...")
        market_feat = np.zeros((len(df), 1), dtype=np.float32)
        legacy = generate_kn2_training_labels(df.reset_index(drop=True), market_feat)

    report = summarize_location_labels(labels, struct, times=times, legacy_labels=legacy, cfg=cfg)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["source_npz"] = str(npz_path)
    report["elapsed_sec"] = round(elapsed, 2)
    report["replay_cases"] = _synthetic_replay_cases()

    report_path = _ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport → {report_path}")

    print("\n=== P0 Summary ===")
    print(f"  should_trade: {report['should_trade_pct']}%")
    print(f"  actions: {report['action_counts']}")
    if "long_labeled" in report:
        ll = report["long_labeled"]
        print(f"  long: n={ll['count']} support_dist={ll['mean_support_dist']} pos_in_range={ll['mean_pos_in_range']}")
    if "vs_legacy_fixed_barrier" in report:
        v = report["vs_legacy_fixed_barrier"]
        print(f"  legacy should_trade: {v['legacy_should_trade_pct']}%")
        print(f"  filtered_out legacy longs: {v['legacy_only_long']}")
    for case in report["replay_cases"]:
        print(f"  replay [{case['name']}]: {case['verdict']} long_cand={case['long_candidate']}")

    if not args.skip_save:
        raw = load_npz(npz_path)
        n = len(labels["action"])
        if year_max is not None:
            full_times = pd.to_datetime(raw["time"], utc=True)
            sel = full_times.year <= year_max
        else:
            sel = slice(None)

        out_dict = {}
        for k in raw.keys():
            arr = raw[k]
            if hasattr(arr, "__len__") and len(arr) >= n and k not in ("symbol", "feature_layout", "loc_label_version", "loc_config_json"):
                out_dict[k] = arr[sel][:n] if year_max is not None else arr[:n]
            elif k == "symbol":
                out_dict[k] = arr
            else:
                out_dict[k] = arr
        for k, v in labels.items():
            out_dict[f"loc_{k}"] = v
        out_dict["loc_label_version"] = np.array(["location_v1"])
        out_dict["loc_config_json"] = np.array([json.dumps(report["config"])])

        out_path = _ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **out_dict)
        mb = out_path.stat().st_size / 1024**2
        print(f"\nSaved {out_path} ({mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
