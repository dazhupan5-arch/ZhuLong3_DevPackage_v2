#!/usr/bin/env python3
"""训练集观望样本下采样至 (做多+做空) 数量的 1.5 倍，目标比例约 1:1:3。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.splits import split_indices


def undersample_flat_1_1_3(
    X: pd.DataFrame,
    y: np.ndarray,
    flat_mult: float = 2.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """下采样至做多:做空:观望 ≈ 1:1:flat_mult（默认 1:1:2）。"""
    rng = np.random.default_rng(seed)
    ix_long = np.where(y == 1)[0]
    ix_short = np.where(y == 2)[0]
    ix_flat = np.where(y == 0)[0]
    n_base = min(len(ix_long), len(ix_short))
    if n_base == 0:
        return X.iloc[:0], y[:0]
    pick_long = rng.choice(ix_long, n_base, replace=False)
    pick_short = rng.choice(ix_short, n_base, replace=False)
    n_flat = min(int(flat_mult * n_base), len(ix_flat))
    pick_flat = rng.choice(ix_flat, n_flat, replace=False)
    pick = np.concatenate([pick_long, pick_short, pick_flat])
    rng.shuffle(pick)
    return X.iloc[pick], y[pick]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training/XAUUSD_triple_labels_v2.csv")
    parser.add_argument("--output", default="data/training/train_balanced_v13_triple.csv")
    parser.add_argument("--features", default="data/training/v13/XAUUSD/features.parquet")
    parser.add_argument("--feature-cols", default="models/XAUUSD/v13/feature_columns.json")
    parser.add_argument("--flat-mult", type=float, default=None)
    parser.add_argument(
        "--pos-neg-ratio",
        type=float,
        default=0.5,
        help="正例:观望 = 1:pos_neg_ratio 的倒数，0.5 表示观望=正例×2",
    )
    args = parser.parse_args()
    flat_mult = args.flat_mult if args.flat_mult is not None else (1.0 / args.pos_neg_ratio)

    root = _ROOT
    cols_path = root / args.feature_cols
    if cols_path.is_file():
        cols = json.loads(cols_path.read_text(encoding="utf-8"))
    else:
        from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13
        cols = list(FEATURE_COLUMNS_LGB_V13)

    lab = pd.read_csv(root / args.input, index_col=0, parse_dates=True)
    feats = pd.read_parquet(root / args.features)
    aligned = feats.join(lab[["label"]], how="inner").dropna(subset=["label"])
    aligned["label"] = aligned["label"].astype(int)
    splits = split_indices(aligned.index)

    tr_ix = splits.train.intersection(aligned.index)
    X_tr = aligned.loc[tr_ix, cols]
    y_tr = aligned.loc[tr_ix, "label"].values.astype(int)
    X_bal, y_bal = undersample_flat_1_1_3(X_tr, y_tr, flat_mult=flat_mult)

    out_df = X_bal.copy()
    out_df["label"] = y_bal
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path)
    n = max(len(y_bal), 1)
    print(
        f"train_balanced: n={len(y_bal)} long={100*(y_bal==1).mean():.1f}% "
        f"short={100*(y_bal==2).mean():.1f}% flat={100*(y_bal==0).mean():.1f}% -> {out_path}"
    )

    for name, ix in (("val", splits.val), ("test1", splits.test1)):
        sub_ix = ix.intersection(aligned.index)
        sub = aligned.loc[sub_ix, cols + ["label"]]
        p = root / f"data/training/{name}_v13_triple.csv"
        sub.to_csv(p)
        y = sub["label"].values
        print(f"{name}: n={len(sub)} long={100*(y==1).mean():.1f}% short={100*(y==2).mean():.1f}% -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
