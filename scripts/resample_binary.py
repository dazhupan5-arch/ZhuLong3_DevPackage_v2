#!/usr/bin/env python3
"""训练集下采样（正:负=1:5），验证/测试保持原分布。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB
from zhulong.training.lgb.splits import split_indices
from zhulong.training.lgb.train_binary import _downsample, to_binary_long


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="含 label 或 label_binary 的 CSV")
    parser.add_argument("--output", default="data/train_balanced.csv")
    parser.add_argument("--features", default="data/training/lgb/XAUUSD/XAUUSD_features.parquet")
    parser.add_argument("--pos-neg-ratio", type=int, default=5, help="负例数 = 正例数 × ratio")
    parser.add_argument("--symbol", default="XAUUSD")
    args = parser.parse_args()

    root = _ROOT
    lab = pd.read_csv(root / args.input if not Path(args.input).is_absolute() else args.input,
                      index_col=0, parse_dates=True)
    if "label_binary" not in lab.columns:
        lab["label_binary"] = (lab["label"] == 1).astype(int)

    feats = pd.read_parquet(root / args.features if not Path(args.features).is_absolute() else args.features)
    aligned = feats.join(lab[["label_binary"]], how="inner")
    splits = split_indices(aligned.index)
    tr_ix = splits.train.intersection(aligned.index)

    X = aligned.loc[tr_ix, FEATURE_COLUMNS_LGB]
    y = aligned.loc[tr_ix, "label_binary"].values.astype(int)
    X_bal, y_bal = _downsample(X, y, neg_ratio=args.pos_neg_ratio)

    out_df = X_bal.copy()
    out_df["label_binary"] = y_bal
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path)
    print(f"wrote {out_path} rows={len(out_df)} pos={y_bal.sum()} neg={(y_bal==0).sum()} pos_rate={y_bal.mean():.2%}")

    # 验证集/测试集（未下采样）
    for name, ix in (("val", splits.val), ("test1", splits.test1)):
        sub = aligned.loc[ix.intersection(aligned.index)]
        p = root / f"data/{name}.csv"
        sub[FEATURE_COLUMNS_LGB + ["label_binary"]].to_csv(p)
        y_sub = sub["label_binary"].values
        print(f"{name}: rows={len(sub)} pos_rate={y_sub.mean():.2%} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
