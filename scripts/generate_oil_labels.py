#!/usr/bin/env python3
"""生成 USOIL v1 三分类标签（动态 ATR 阈值）。"""

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
from zhulong.training.oil_v1.labels import generate_oil_triple_labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training/lgb/USOIL/USOIL_M5.csv")
    parser.add_argument("--output", default="data/training/USOIL_labeled_triple.csv")
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--gain-fixed", type=float, default=0.003)
    parser.add_argument("--no-dynamic", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = _ROOT
    m5 = load_vendor_csv(root / args.input)
    lab = generate_oil_triple_labels(
        m5, args.horizon, args.gain_fixed, use_dynamic=not args.no_dynamic
    )
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    lab[["label"]].to_csv(out)
    print(f"wrote {out}")

    splits = split_indices(lab.index)
    for name, ix in (("all", lab.index), ("train", splits.train), ("val", splits.val), ("test1", splits.test1)):
        sub = lab.loc[ix, "label"]
        n = max(len(sub), 1)
        print(
            f"{name}: long={100*(sub==1).mean():.1f}% short={100*(sub==2).mean():.1f}% "
            f"flat={100*(sub==0).mean():.1f}% n={len(sub)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
