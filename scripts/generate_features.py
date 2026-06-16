#!/usr/bin/env python3
"""v8 特征计算（105 维，实时可调用，无未来数据泄漏）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.live_v8_features import build_live_v8_row, m5_from_mt5  # noqa: E402
from zhulong.training.v8.features import build_v8_features  # noqa: E402
from zhulong.training.v8.decompose import decompose_h4_to_m5  # noqa: E402
from zhulong.utils.paths import install_dir, model_dir_for_symbol  # noqa: E402


def compute_features_105d(
    m5: pd.DataFrame,
    symbol: str = "XAUUSD",
    imf_cache: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    计算全量 v8 特征表（每行对应当前时刻，仅使用历史窗口）。
    模型推理取最后一行即可（单样本 105 维，已聚合序列统计）。
    """
    root = install_dir()
    imf_path = imf_cache or model_dir_for_symbol(symbol) / "imf_vmd.parquet"
    if imf_path.is_file():
        imf = pd.read_parquet(imf_path)
        if imf.index.tz is not None:
            imf.index = imf.index.tz_localize(None)
        imf = imf.reindex(m5.index, method="ffill")
        if imf.isna().all(axis=1).any():
            tail = decompose_h4_to_m5(m5.tail(min(len(m5), 8000)))
            imf = imf.combine_first(tail)
    else:
        imf = decompose_h4_to_m5(m5)
    macro_dir = root / "data" / "macro"
    return build_v8_features(m5, imf, macro_dir=macro_dir)


def latest_feature_vector(
    symbol: str = "XAUUSD",
    m5_bars: int = 2000,
    m5: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame, pd.DataFrame]:
    """返回 (row, columns, m5, feats_df_last_row)。"""
    if m5 is None:
        m5 = m5_from_mt5(symbol, bars=m5_bars)
    feats, feat_cols = compute_features_105d(m5, symbol)
    t = feats.index[-1]
    row = feats.loc[t, feat_cols].to_numpy(dtype=np.float32)
    feats_row = feats.loc[[t], feat_cols]
    return row, feat_cols, m5, feats_row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    m5 = m5_from_mt5(args.symbol, bars=args.bars)
    feats, cols = compute_features_105d(m5, args.symbol)
    print(f"features: {len(feats)} rows x {len(cols)} cols")
    print(f"last bar: {feats.index[-1]}")
    if args.output:
        out = Path(args.output)
        feats.to_parquet(out)
        print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
