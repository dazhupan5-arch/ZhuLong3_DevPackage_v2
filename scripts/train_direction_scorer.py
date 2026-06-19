#!/usr/bin/env python3
"""训练 V17 DirectionScorer (LightGBM 回归)。"""

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


def _load_keep_features(path: Path | None) -> list[int] | None:
    if path is None or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    keep = data.get("keep_features")
    return list(keep) if keep else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/clean/training_v17_direction.npz")
    parser.add_argument("--out", default="models/direction_scorer")
    parser.add_argument("--keep-features", default="logs/shap_feature_rank.json")
    parser.add_argument("--train-end", default=TRAIN_END_DEFAULT)
    args = parser.parse_args()

    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}")
        return 1

    raw = load_npz(npz_path)
    X = np.asarray(raw["struct"], dtype=np.float32)
    y = np.asarray(raw["direction_score"], dtype=np.float32)
    times = np.asarray(raw["time"])
    keep = _load_keep_features(_ROOT / args.keep_features if args.keep_features else None)
    if keep:
        X = X[:, keep]

    train_mask, val_mask = temporal_train_val_masks(times, train_end=args.train_end)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"train={X_train.shape} val={X_val.shape}")

    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 100,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    y_pred = model.predict(X_val)
    threshold_report = {}
    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40]:
        mask = np.abs(y_pred) >= threshold
        if mask.sum() < 100:
            continue
        acc = float((np.sign(y_pred[mask]) == np.sign(y_val[mask])).mean())
        cov = float(mask.mean())
        threshold_report[str(threshold)] = {
            "direction_accuracy": round(acc, 4),
            "coverage": round(cov, 4),
            "samples": int(mask.sum()),
        }
        print(f"  threshold={threshold:.2f}: acc={acc:.1%} cov={cov:.1%} n={mask.sum():,}")

    out_dir = _ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "direction_scorer.lgb"
    model.save_model(str(model_path))
    meta = {
        "model_type": "lgb_regression",
        "feature_dim": int(X.shape[1]),
        "keep_features": keep,
        "train_end": args.train_end,
        "best_iteration": int(model.best_iteration),
        "val_rmse": float(model.best_score["valid_0"]["rmse"]),
        "architecture": "v17_direction_scorer",
        "threshold_report": threshold_report,
        "passed": False,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"保存 → {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
