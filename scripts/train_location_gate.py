#!/usr/bin/env python3
"""训练 V17 LocationGate (XGBoost 二分类)。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from zhulong.agent.training_utils import load_npz, temporal_train_val_masks, TRAIN_END_DEFAULT
from zhulong.agent.v17.features import LOCATION_FEATURE_NAMES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/clean/training_v17_location.npz")
    parser.add_argument("--out", default="models/location_gate")
    parser.add_argument("--train-end", default=TRAIN_END_DEFAULT)
    args = parser.parse_args()

    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}")
        return 1

    raw = load_npz(npz_path)
    X = np.asarray(raw["location_features"], dtype=np.float32)
    y = np.asarray(raw["location_label"], dtype=np.int8)
    times = np.asarray(raw["time"])

    train_mask, val_mask = temporal_train_val_masks(times, train_end=args.train_end)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    train_has_dir = X_train[:, 11] != 0
    val_has_dir = X_val[:, 11] != 0
    print(f"train dir samples={train_has_dir.sum():,} pos rate={y_train[train_has_dir].mean():.1%}")
    print(f"val dir samples={val_has_dir.sum():,} pos rate={y_val[val_has_dir].mean():.1%}")

    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=30,
        tree_method="hist",
        random_state=42,
        verbosity=1,
    )
    model.fit(
        X_train[train_has_dir],
        y_train[train_has_dir],
        eval_set=[(X_val[val_has_dir], y_val[val_has_dir])],
        verbose=50,
    )

    y_prob = model.predict_proba(X_val[val_has_dir])[:, 1]
    threshold_report = {}
    for threshold in [0.50, 0.55, 0.60, 0.65]:
        mask = y_prob >= threshold
        if mask.sum() < 50:
            continue
        prec = float(y_val[val_has_dir][mask].mean())
        cov = float(mask.mean())
        threshold_report[str(threshold)] = {
            "precision": round(prec, 4),
            "coverage": round(cov, 4),
            "samples": int(mask.sum()),
        }
        print(f"  quality>={threshold:.2f}: prec={prec:.1%} cov={cov:.1%}")

    out_dir = _ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "location_gate.xgb"
    model.save_model(str(model_path))
    meta = {
        "model_type": "xgb_binary",
        "feature_names": list(LOCATION_FEATURE_NAMES),
        "architecture": "v17_location_gate",
        "val_auc": float(getattr(model, "best_score", 0) or 0),
        "threshold_report": threshold_report,
        "passed": False,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"保存 → {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
