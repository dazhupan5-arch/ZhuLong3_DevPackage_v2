#!/usr/bin/env python3
"""生成三重屏障三分类标签（0 观望 / 1 多 / 2 空）。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.splits import split_indices
from zhulong.training.lgb.triple_barrier import generate_triple_barrier_labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training/lgb/XAUUSD/XAUUSD_M5.csv")
    parser.add_argument("--output", default="data/training/XAUUSD_triple_labels_v2.csv")
    parser.add_argument("--sl", type=float, default=1.2)
    parser.add_argument("--tp", type=float, default=1.8)
    parser.add_argument("--hold", type=int, default=12)
    parser.add_argument("--min-atr", type=float, default=0.001)
    parser.add_argument("--trend-filter", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = _ROOT
    m5 = load_vendor_csv(root / args.input)
    lab = generate_triple_barrier_labels(
        m5,
        sl_mult=args.sl,
        tp_mult=args.tp,
        max_hold=args.hold,
        min_atr_pct=args.min_atr,
        trend_filter=args.trend_filter,
    )
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(out)
    print(f"wrote {out}")

    splits = split_indices(lab.index)
    for name, ix in (("all", lab.index), ("train", splits.train), ("val", splits.val), ("test1", splits.test1)):
        sub = lab.loc[ix].dropna(subset=["label"])
        y = sub["label"].astype(int)
        n = max(len(y), 1)
        print(
            f"{name}: long={100*(y==1).mean():.1f}% short={100*(y==2).mean():.1f}% "
            f"flat={100*(y==0).mean():.1f}% n={len(y)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
