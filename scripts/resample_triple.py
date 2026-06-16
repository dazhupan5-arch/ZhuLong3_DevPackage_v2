#!/usr/bin/env python3
"""训练集三分类 1:1:1 下采样；验证/测试保持原分布。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.splits import split_indices


def balance_triple(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    ix_long = np.where(y == 1)[0]
    ix_short = np.where(y == 2)[0]
    ix_flat = np.where(y == 0)[0]
    n = min(len(ix_long), len(ix_short), len(ix_flat))
    if n == 0:
        return X.iloc[:0], y[:0]
    pick = np.concatenate([
        rng.choice(ix_long, n, replace=False),
        rng.choice(ix_short, n, replace=False),
        rng.choice(ix_flat, n, replace=False),
    ])
    rng.shuffle(pick)
    return X.iloc[pick], y[pick]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training/XAUUSD_labeled_triple.csv")
    parser.add_argument("--output", default="data/train_balanced_triple.csv")
    parser.add_argument("--features", default="data/training/v8/XAUUSD/features.parquet")
    parser.add_argument("--feature-cols", default="data/training/v8/XAUUSD/feature_columns.json")
    parser.add_argument("--strategy", default="undersample", choices=["undersample"])
    args = parser.parse_args()

    root = _ROOT
    import json

    cols = json.loads((root / args.feature_cols).read_text(encoding="utf-8"))
    lab = pd.read_csv(root / args.input, index_col=0, parse_dates=True)
    feats = pd.read_parquet(root / args.features)
    aligned = feats.join(lab[["label"]], how="inner")
    splits = split_indices(aligned.index)

    tr_ix = splits.train.intersection(aligned.index)
    X_tr = aligned.loc[tr_ix, cols]
    y_tr = aligned.loc[tr_ix, "label"].values.astype(int)
    X_bal, y_bal = balance_triple(X_tr, y_tr)

    out_df = X_bal.copy()
    out_df["label"] = y_bal
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path)
    print(f"train_balanced: n={len(y_bal)} per_class={int((y_bal==1).sum())} -> {out_path}")

    for name, ix in (("val", splits.val), ("test1", splits.test1)):
        sub_ix = ix.intersection(aligned.index)
        sub = aligned.loc[sub_ix, cols + ["label"]]
        p = root / f"data/{name}_triple.csv"
        sub.to_csv(p)
        y = sub["label"].values
        print(f"{name}: n={len(sub)} long={100*(y==1).mean():.1f}% short={100*(y==2).mean():.1f}% -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
