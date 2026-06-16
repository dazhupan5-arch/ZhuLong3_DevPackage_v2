"""v4 标签：未来 horizon 根 M5 收益率符号（±gain_threshold）。"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HORIZON = 12
DEFAULT_GAIN_THRESHOLD = 0.0025  # v4.2: 0.25% @ 60min
V13_GAIN_THRESHOLD = 0.0015  # v13: 0.15% @ 60min


@dataclass
class LabelConfig:
    horizon: int = DEFAULT_HORIZON
    gain_threshold: float = DEFAULT_GAIN_THRESHOLD

    def as_dict(self) -> dict:
        return {"horizon": self.horizon, "gain_threshold": self.gain_threshold}


def generate_labels(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    gain_threshold: float = DEFAULT_GAIN_THRESHOLD,
    config: LabelConfig | None = None,
) -> pd.DataFrame:
    """
    未来 horizon 根 M5 收盘价收益率：
      > gain_threshold  -> 1 (做多)
      < -gain_threshold -> -1 (做空)
      其余              -> 0 (观望)
    """
    cfg = config or LabelConfig(horizon=horizon, gain_threshold=gain_threshold)
    close = df["close"]
    future_returns = (close.shift(-cfg.horizon) - close) / close.replace(0, np.nan)

    labels = np.zeros(len(df), dtype=np.int8)
    labels[future_returns > cfg.gain_threshold] = 1
    labels[future_returns < -cfg.gain_threshold] = -1

    out = pd.DataFrame({"label": labels}, index=df.index)
    n = max(len(df), 1)
    n_long = int((labels == 1).sum())
    n_short = int((labels == -1).sum())
    n_flat = int((labels == 0).sum())
    logger.info(
        "labels v4 h=%s gain=%.4f (%.2f%%) long=%s (%.2f%%) short=%s (%.2f%%) flat=%s (%.2f%%)",
        cfg.horizon,
        cfg.gain_threshold,
        cfg.gain_threshold * 100,
        n_long,
        100.0 * n_long / n,
        n_short,
        100.0 * n_short / n,
        n_flat,
        100.0 * n_flat / n,
    )
    return out


generate_strict_labels = generate_labels


def generate_direction_labels(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    gain_threshold: float = V13_GAIN_THRESHOLD,
) -> np.ndarray:
    """
    v13 方向标签：未来 horizon 根 M5 收益率符号。
    返回 labels 数组 (1: 做多, -1: 做空, 0: 观望)。
    """
    future_ret = (df["close"].shift(-horizon) - df["close"]) / df["close"].replace(0, np.nan)
    labels = np.zeros(len(df), dtype=np.int8)
    labels[future_ret > gain_threshold] = 1
    labels[future_ret < -gain_threshold] = -1
    n = max(len(df), 1)
    logger.info(
        "direction labels h=%s gain=%.4f (%.2f%%) long=%s (%.2f%%) short=%s (%.2f%%) flat=%s (%.2f%%)",
        horizon,
        gain_threshold,
        gain_threshold * 100,
        int((labels == 1).sum()),
        100.0 * (labels == 1).sum() / n,
        int((labels == -1).sum()),
        100.0 * (labels == -1).sum() / n,
        int((labels == 0).sum()),
        100.0 * (labels == 0).sum() / n,
    )
    return labels


def _label_stats(labels: pd.Series, name: str = "all") -> dict[str, float]:
    n = max(len(labels), 1)
    return {
        f"{name}_long_pct": 100.0 * (labels == 1).sum() / n,
        f"{name}_short_pct": 100.0 * (labels == -1).sum() / n,
        f"{name}_flat_pct": 100.0 * (labels == 0).sum() / n,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成方向标签（v4 / v13 direction）")
    parser.add_argument("--method", choices=["v4", "direction"], default="v4")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--input", default="", help="M5 CSV 输入（同 --m5）")
    parser.add_argument("--output", default="", help="标签 CSV 输出路径")
    parser.add_argument(
        "--m5",
        default="",
        help="M5 CSV 路径，默认 data/training/lgb/{symbol}/{symbol}_M5.csv",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = Path(__file__).resolve().parents[3]
    m5_arg = args.input or args.m5
    m5_path = Path(m5_arg) if m5_arg else root / "data" / "training" / "lgb" / args.symbol / f"{args.symbol}_M5.csv"
    if not m5_path.is_file():
        logger.error("M5 文件不存在: %s", m5_path)
        return 1

    sys.path.insert(0, str(root))
    from zhulong.training.lgb.data_io import load_vendor_csv
    from zhulong.training.lgb.splits import split_indices

    m5 = load_vendor_csv(m5_path)
    if args.method == "direction":
        gain = args.gain if args.gain is not None else V13_GAIN_THRESHOLD
        labels = generate_direction_labels(m5, horizon=args.horizon, gain_threshold=gain)
        lab = pd.DataFrame({"label": labels}, index=m5.index)
        default_csv = root / "data" / "training" / f"{args.symbol}_labeled_v13.csv"
        parquet_suffix = "labels_v13"
    else:
        gain = args.gain if args.gain is not None else DEFAULT_GAIN_THRESHOLD
        cfg = LabelConfig(horizon=args.horizon, gain_threshold=gain)
        lab = generate_labels(m5, config=cfg)
        default_csv = root / "data" / "training" / f"{args.symbol}_labeled_v4_2.csv"
        parquet_suffix = "labels"

    out_dir = m5_path.parent
    parquet_path = out_dir / f"{args.symbol}_{parquet_suffix}.parquet"
    lab.to_parquet(parquet_path)
    logger.info("wrote %s", parquet_path)

    if args.output:
        csv_path = Path(args.output)
    else:
        csv_path = default_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(csv_path)
    logger.info("wrote %s", csv_path)

    splits = split_indices(lab.index)
    for split_name, ix in (("train", splits.train), ("val", splits.val), ("test1", splits.test1)):
        sub = lab.loc[ix, "label"]
        stats = _label_stats(sub, split_name)
        logger.info(
            "%s: long=%.1f%% short=%.1f%% flat=%.1f%%",
            split_name,
            stats[f"{split_name}_long_pct"],
            stats[f"{split_name}_short_pct"],
            stats[f"{split_name}_flat_pct"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
