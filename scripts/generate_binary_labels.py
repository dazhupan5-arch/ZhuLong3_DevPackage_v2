#!/usr/bin/env python3
"""从 v4.2 三分类标签生成 v5 二分类标签（做多=1，其余=0）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inp = Path(args.input)
    df = pd.read_csv(inp, index_col=0, parse_dates=True) if inp.suffix == ".csv" else pd.read_parquet(inp)
    if "label" not in df.columns:
        raise SystemExit("input must contain 'label' column")

    out = df.copy()
    out["label_binary"] = (out["label"] == 1).astype(int)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)
    n = len(out)
    pos = int(out["label_binary"].sum())
    print(f"wrote {out_path}")
    print(f"正例: {pos} ({100*pos/n:.2f}%)  负例: {n-pos} ({100*(n-pos)/n:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
