#!/usr/bin/env python3
"""生成 V17 LocationGate 二分类标签 NPZ。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from zhulong.agent.kn2_location_labels import compute_pos_in_range
from zhulong.agent.training_utils import load_npz
from zhulong.agent.v17.labels import (
    atr_percentile_series,
    build_location_feature_matrix,
    direction_series_from_scores,
    make_direction_regression_labels,
    make_location_binary_labels,
    summarize_labels,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/clean/training_v17_direction.npz")
    parser.add_argument("--fallback-npz", default="data/clean/training_horizon_v16.npz")
    parser.add_argument("--out", default="data/clean/training_v17_location.npz")
    parser.add_argument("--direction-threshold", type=float, default=0.35)
    args = parser.parse_args()

    src = _ROOT / args.npz
    if not src.is_file():
        src = _ROOT / args.fallback_npz
    if not src.is_file():
        print("缺少 direction NPZ，请先 prepare_direction_regression_labels.py")
        return 1

    raw = load_npz(src)
    close = np.asarray(raw["close"], dtype=np.float64)
    high = np.asarray(raw["high"], dtype=np.float64)
    low = np.asarray(raw["low"], dtype=np.float64)
    atr = np.asarray(raw["atr"], dtype=np.float64)
    struct = np.asarray(raw["struct"], dtype=np.float32)

    if "direction_score" in raw:
        direction_score = np.asarray(raw["direction_score"], dtype=np.float32)
    else:
        direction_score = make_direction_regression_labels(close, atr)

    direction_series = direction_series_from_scores(direction_score, args.direction_threshold)
    location_labels = make_location_binary_labels(
        close, high, low, atr, direction_series
    )

    times = pd.to_datetime(raw["time"])
    pos_in_range = compute_pos_in_range(close, window=48)
    atr_pct = atr_percentile_series(atr)
    loc_features = build_location_feature_matrix(
        struct, pos_in_range, direction_score, atr_pct
    )

    summary = summarize_labels(direction_score, location_labels, direction_series)
    summary["location_features_shape"] = list(loc_features.shape)

    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: raw[k] for k in raw}
    payload["location_label"] = location_labels
    payload["location_features"] = loc_features
    payload["pos_in_range"] = pos_in_range
    payload["direction_series"] = direction_series
    np.savez_compressed(out_path, **payload)

    report = _ROOT / "data/training/reports/v17/location_label_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"保存 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
