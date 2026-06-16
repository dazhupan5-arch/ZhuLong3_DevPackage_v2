#!/usr/bin/env python3
"""v8 SHAP 特征重要性（LightGBM）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from zhulong.training.lgb.splits import split_indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data/training/v8/XAUUSD")
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    root = _ROOT
    data_dir = root / args.data_dir
    model_dir = root / "models" / args.symbol / "v8"
    cols = json.loads((data_dir / "feature_columns.json").read_text(encoding="utf-8"))
    feats = pd.read_parquet(data_dir / "features.parquet")
    labels = pd.read_csv(data_dir / "labels.csv", index_col=0, parse_dates=True)
    aligned = feats.join(labels[["label_cls"]], how="inner")
    va_ix = split_indices(aligned.index).val.intersection(aligned.index)
    val = aligned.loc[va_ix]
    if len(val) > args.max_samples:
        val = val.sample(args.max_samples, random_state=42)

    booster = lgb.Booster(model_file=str(model_dir / "lgb_classifier.txt"))
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(val[cols])

    report_dir = root / "data" / "training" / "reports" / "v8" / args.symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, val[cols], feature_names=cols, show=False, max_display=30)
    plt.tight_layout()
    out = report_dir / "shap_summary.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"shap -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
