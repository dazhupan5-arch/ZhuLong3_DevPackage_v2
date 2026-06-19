#!/usr/bin/env python3
"""V17 SHAP/LGB 特征重要性审计。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from zhulong.agent.structure_analyzer import FEATURE_NAMES
from zhulong.agent.training_utils import load_npz, signed_to_class, temporal_train_val_masks, TRAIN_END_DEFAULT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/clean/training_horizon_v16.npz")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", default="logs/shap_feature_rank.json")
    parser.add_argument("--train-end", default=TRAIN_END_DEFAULT)
    args = parser.parse_args()

    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}")
        return 1

    raw = load_npz(npz_path)
    X = np.asarray(raw["struct"], dtype=np.float32)
    y = signed_to_class(np.asarray(raw["labels"], dtype=np.int8))
    times = np.asarray(raw["time"])
    train_mask, _ = temporal_train_val_masks(times, train_end=args.train_end)
    X_train, y_train = X[train_mask], y[train_mask]

    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    mean_shap = model.feature_importances_.astype(np.float64)
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[~train_mask][:5000])
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(0)
    except Exception as ex:
        print(f"SHAP 不可用，使用 LGB feature_importances: {ex}")

    ranked = sorted(
        zip(FEATURE_NAMES, mean_shap.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    keep_idx = sorted(FEATURE_NAMES.index(name) for name, _ in ranked[: args.top_k])

    print("特征重要性排名：")
    for i, (name, val) in enumerate(ranked):
        mark = "OK" if i < args.top_k else "--"
        print(f"  [{mark}] [{i+1:2d}] {name:30s} {val:.4f}")

    out = {
        "source_npz": str(npz_path),
        "top_k": args.top_k,
        "keep_features": keep_idx,
        "ranking": [{"name": n, "score": round(v, 6)} for n, v in ranked],
    }
    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"保存 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
