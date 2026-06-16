#!/usr/bin/env python3
"""P4：Horizon V16 位置感知方向标签 — 生成 NPZ + 分布报告（GPU 训练前验收）。"""

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

from zhulong.agent.horizon_location_labels import (
    apply_location_gate_to_horizon_labels,
    summarize_horizon_location_labels,
)
from zhulong.agent.kn2_location_labels import LocationLabelConfig, replay_bar_diagnosis
from zhulong.agent.training_utils import load_npz


def _synthetic_replay_cases() -> list[dict]:
    bad = np.zeros(30, dtype=np.float32)
    bad[0], bad[3], bad[4], bad[5], bad[6] = 0.05, 1.2, 0.25, 0.2, 0.5
    good = np.zeros(30, dtype=np.float32)
    good[0], good[3], good[4], good[5], good[6] = 0.08, 0.35, 1.8, 0.55, 0.3
    return [
        {
            "name": "replay_ranging_top_chase_long",
            "note": "震荡高位 legacy long 应被门控为 flat",
            **replay_bar_diagnosis(bad, pos_in_range=0.72, regime_name="ranging"),
        },
        {
            "name": "replay_ranging_bottom_long",
            "note": "震荡低位 long 候选保留",
            **replay_bar_diagnosis(good, pos_in_range=0.28, regime_name="ranging"),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Horizon V16 位置感知标签 P4")
    parser.add_argument("--npz", default="data/clean/training_horizon_v16.npz")
    parser.add_argument("--out", default="data/clean/training_horizon_v16_location.npz")
    parser.add_argument("--report", default="data/training/reports/horizon_v16/location_label_report.json")
    parser.add_argument("--year-max", type=int, default=0, help="仅处理到该年（0=全部）")
    parser.add_argument("--skip-save", action="store_true")
    args = parser.parse_args()

    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}，请先 clean_training_data_v16.py 或 prepare_horizon_v16_data.py")
        return 1

    print(f"Loading {npz_path} ...")
    raw = load_npz(npz_path)
    struct = np.asarray(raw["struct"], dtype=np.float32)
    labels_legacy = np.asarray(raw["labels"], dtype=np.int8)
    close = np.asarray(raw["close"], dtype=np.float64)
    times = pd.to_datetime(raw["time"], utc=True) if "time" in raw else None

    if args.year_max > 0 and times is not None:
        mask = times.year <= args.year_max
        struct = struct[mask]
        labels_legacy = labels_legacy[mask]
        close = close[mask]
        times = times[mask]
        print(f"year_max={args.year_max} → {len(close):,} bars")

    cfg = LocationLabelConfig()
    t0 = time.perf_counter()
    print("Applying location gate to horizon direction labels ...")
    gated = apply_location_gate_to_horizon_labels(struct, labels_legacy, close, cfg=cfg)
    elapsed = time.perf_counter() - t0

    report = summarize_horizon_location_labels(gated, cfg=cfg)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["source_npz"] = str(npz_path)
    report["elapsed_sec"] = round(elapsed, 2)
    report["replay_cases"] = _synthetic_replay_cases()
    if times is not None:
        for year in sorted(set(times.year)):
            ym = times.year == year
            leg = gated["labels_legacy"][ym]
            lab = gated["labels"][ym]
            report[f"year_{year}"] = {
                "bars": int(ym.sum()),
                "legacy_long": int((leg == 1).sum()),
                "gated_long": int((lab == 1).sum()),
                "legacy_short": int((leg == -1).sum()),
                "gated_short": int((lab == -1).sum()),
            }

    report_path = _ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report → {report_path}")

    print("\n=== P4 Horizon Location Summary ===")
    print(f"  legacy: {report['legacy_counts']} ({report['legacy_pct']})")
    print(f"  gated:  {report['gated_counts']} ({report['gated_pct']})")
    print(f"  filtered long={report['filtered_long']} short={report['filtered_short']}")
    for case in report["replay_cases"]:
        print(f"  replay [{case['name']}]: {case['verdict']}")

    if args.skip_save:
        return 0

    n = len(gated["labels"])
    out_dict: dict[str, np.ndarray] = {}
    for k in raw.keys():
        arr = raw[k]
        if hasattr(arr, "__len__") and len(arr) >= n and k not in (
            "symbol",
            "loc_horizon_version",
            "loc_config_json",
            "labels",
            "labels_legacy",
        ):
            if args.year_max > 0 and times is not None and len(arr) > n:
                full_times = pd.to_datetime(raw["time"], utc=True)
                sel = full_times.year <= args.year_max
                out_dict[k] = arr[sel][:n]
            else:
                out_dict[k] = arr[:n]
        elif k == "symbol":
            out_dict[k] = arr

    out_dict["labels"] = gated["labels"]
    out_dict["labels_legacy"] = gated["labels_legacy"]
    for k in ("long_candidate", "short_candidate", "pos_in_range", "regime_code", "gate_mask"):
        out_dict[f"loc_{k}"] = gated[k]
    out_dict["loc_horizon_version"] = np.array(["horizon_location_v1"])
    out_dict["loc_config_json"] = np.array([json.dumps(report["config"])])

    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out_dict)
    mb = out_path.stat().st_size / 1024**2
    print(f"\nSaved {out_path} ({mb:.1f} MB)")
    print("GPU: git pull && git lfs pull && scripts/train_horizon_v16_remote.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
