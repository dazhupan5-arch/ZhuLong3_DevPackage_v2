#!/usr/bin/env python3
"""验证 V17 标签分布。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from zhulong.agent.training_utils import load_npz
from zhulong.agent.v17.labels import summarize_labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-npz", default="data/clean/training_v17_direction.npz")
    parser.add_argument("--location-npz", default="data/clean/training_v17_location.npz")
    args = parser.parse_args()

    failures: list[str] = []
    dir_path = _ROOT / args.direction_npz
    loc_path = _ROOT / args.location_npz

    if dir_path.is_file():
        raw = load_npz(dir_path)
        ds = np.asarray(raw["direction_score"], dtype=np.float32)
        summary = summarize_labels(ds, np.zeros(len(ds), dtype=np.int8))
        print("Direction labels:", json.dumps(summary, indent=2))
        if summary["direction_score_std"] < 0.05:
            failures.append("direction_score_std_too_low")
        if abs(summary["direction_score_mean"]) > 0.15:
            failures.append("direction_score_mean_off_center")
    else:
        failures.append("missing_direction_npz")

    if loc_path.is_file():
        raw = load_npz(loc_path)
        ds = np.asarray(raw["direction_score"], dtype=np.float32)
        ll = np.asarray(raw["location_label"], dtype=np.int8)
        dirs = np.asarray(raw.get("direction_series", np.zeros(len(ds), dtype=np.int8)))
        summary = summarize_labels(ds, ll, dirs)
        print("Location labels:", json.dumps(summary, indent=2))
        rate = summary["location_label_positive_rate"]
        if rate < 0.35 or rate > 0.70:
            failures.append(f"location_positive_rate_out_of_range:{rate:.3f}")
    else:
        failures.append("missing_location_npz")

    if failures:
        print("FAIL:", failures)
        return 1
    print("PASS: v17 label validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
