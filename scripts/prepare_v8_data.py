#!/usr/bin/env python3
"""v8 数据准备：VMD 分解 + 特征 + 标签。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.v8.decompose import decompose_h4_to_m5, save_decomposition
from zhulong.training.v8.features import build_v8_features
from zhulong.training.v8.labels import generate_v8_labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--input", default="data/training/lgb/XAUUSD/XAUUSD_M5.csv")
    parser.add_argument("--output-dir", default="data/training/v8/XAUUSD")
    parser.add_argument("--vmd-k", type=int, default=6)
    parser.add_argument("--skip-decompose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = _ROOT
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    m5 = load_vendor_csv(root / args.input)
    imf_path = out / "imf_vmd.parquet"
    if args.skip_decompose and imf_path.is_file():
        import pandas as pd
        imf = pd.read_parquet(imf_path)
    else:
        imf = decompose_h4_to_m5(m5, K=args.vmd_k)
        save_decomposition(imf, imf_path)

    labels = generate_v8_labels(m5)
    labels.to_csv(out / "labels.csv")
    feats, cols = build_v8_features(m5, imf, macro_dir=root / "data" / "macro")
    feats.to_parquet(out / "features.parquet")
    (out / "feature_columns.json").write_text(
        __import__("json").dumps(cols, indent=2), encoding="utf-8"
    )
    print(f"features {feats.shape} -> {out / 'features.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
