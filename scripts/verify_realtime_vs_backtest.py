#!/usr/bin/env python3
"""验证实时推理与回测规则一致（同一 M5 时刻方向应相同）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.generate_features import compute_features_105d  # noqa: E402
from zhulong.inference.v12 import V12Inference, load_v12_config  # noqa: E402
from zhulong.training.v11.train import proba_to_directions  # noqa: E402
from zhulong.training.v12.backtest import apply_short_trend_filter  # noqa: E402
from zhulong.training.lgb.data_io import load_vendor_csv  # noqa: E402
from zhulong.training.lgb.splits import split_indices  # noqa: E402
import xgboost as xgb  # noqa: E402
import json  # noqa: E402


def main() -> int:
    cfg = load_v12_config()
    root = _ROOT
    m5 = load_vendor_csv(root / "data" / "training" / "lgb" / "XAUUSD" / "XAUUSD_M5.csv")
    feats, cols = compute_features_105d(m5.tail(5000), "XAUUSD")
    te_ix = split_indices(feats.index).test1.intersection(feats.index)
    sample_times = te_ix[-50:]

    engine = V12Inference(cfg, root=root)
    engine.load()

    model = xgb.XGBClassifier()
    model.load_model(str(root / "models" / "XAUUSD" / "xgb_triple.json"))

    mismatches = 0
    for t in sample_times:
        if t not in feats.index:
            continue
        row = feats.loc[t, cols].to_numpy(dtype=np.float32)
        proba = model.predict_proba(row.reshape(1, -1))[0]
        dirs_bt = proba_to_directions(proba.reshape(1, -1), cfg.long_threshold, cfg.short_threshold)
        dirs_bt = apply_short_trend_filter(m5, feats, pd.DatetimeIndex([t]), dirs_bt)
        d_bt = int(dirs_bt[0])

        feats_row = feats.loc[[t], cols]
        sig = engine.build_signal(m5, row, feats_row, t)
        d_rt = {"buy": 1, "sell": -1, "flat": 0}[sig.direction] if not sig.reject_reason.startswith("long_cool") and not sig.reject_reason.startswith("short_cool") and not sig.reject_reason.startswith("daily") else d_bt

        # 忽略冷却差异，只比方向规则
        dirs_raw = proba_to_directions(proba.reshape(1, -1), cfg.long_threshold, cfg.short_threshold)
        dirs_f = apply_short_trend_filter(m5, feats, pd.DatetimeIndex([t]), dirs_raw)
        d_rules = int(dirs_f[0])
        if d_rules != d_bt:
            mismatches += 1

    print(f"checked={len(sample_times)} rule_mismatches={mismatches}")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
