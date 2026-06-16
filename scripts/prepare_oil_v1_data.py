#!/usr/bin/env python3
"""USOIL v1 数据准备：VMD 分解 + 原油特征 + 宏观/库存。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.v8.decompose import decompose_h4_to_m5, save_decomposition
from zhulong.training.oil_v1.features import build_oil_v1_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="USOIL")
    parser.add_argument("--input", default="data/training/lgb/USOIL/USOIL_M5.csv")
    parser.add_argument("--output-dir", default="data/training/oil_v1/USOIL")
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

    feats, cols = build_oil_v1_features(m5, imf, macro_dir=root / "data" / "macro")
    feats.to_parquet(out / "features.parquet")
    (out / "feature_columns.json").write_text(json.dumps(cols, indent=2), encoding="utf-8")
    print(f"oil v1 features {feats.shape} -> {out / 'features.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
