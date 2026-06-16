#!/usr/bin/env python3
"""构建 V13 特征缓存（含增强 + 关键位置特征）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.v13.train_pipeline import compute_features_v13


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--enhanced", action="store_true")
    parser.add_argument("--key-levels", action="store_true", default=True)
    args = parser.parse_args()

    m5 = load_vendor_csv(_ROOT / "data" / "training" / "lgb" / args.symbol / f"{args.symbol}_M5.csv")
    feats = compute_features_v13(
        m5,
        include_enhanced=args.enhanced or True,
        include_key_levels=args.key_levels,
        root=_ROOT,
    )
    sub = "v13_quality" if args.key_levels else ("v13_enhanced" if args.enhanced else "v13")
    out = _ROOT / "data" / "training" / sub / args.symbol / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out)
    print(f"features: {feats.shape[0]} x {feats.shape[1]} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
