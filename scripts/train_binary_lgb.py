#!/usr/bin/env python3
"""v5/v5.1/v6 二分类 LightGBM 训练主入口。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from zhulong.training.lgb.backtest import DEFAULT_COOLDOWN_BARS
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.labels_profit import DEFAULT_MAX_HOLD_BARS
from zhulong.training.lgb.train_binary import run_binary_training, save_binary_model
from zhulong.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

_STAGE_CFG = {
    "v5": ("v5", "lgb_binary_meta.pkl", "config_v5.json", "manifest_v5.json", "acceptance_report_v5.json", "binary_long"),
    "v51": ("v5.1", "lgb_binary_v5_1_meta.pkl", "config_v5_1.json", "manifest_v5_1.json", "acceptance_report_v5_1.json", "binary_long"),
    "v6": ("v6", "lgb_profit_meta.pkl", "config_v6.json", "manifest_v6.json", "acceptance_report_v6.json", "profit_long"),
    "v61": ("v6.1", "lgb_profit_24_meta.pkl", "config_v61.json", "manifest_v61.json", "acceptance_report_v61.json", "profit_long"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="二分类 LightGBM 训练")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--labels", default="data/training/XAUUSD_labeled_profit.csv")
    parser.add_argument("--output", default="models/XAUUSD/lgb/lgb_profit.txt")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--gain", type=float, default=0.0, help="v6 盈亏标签忽略此参数")
    parser.add_argument("--acceptance-stage", default="v6", choices=list(_STAGE_CFG))
    parser.add_argument("--target-precision", type=float, default=0.55)
    parser.add_argument("--target-recall", type=float, default=0.20)
    parser.add_argument("--max-signals-per-day", type=float, default=8.0)
    parser.add_argument("--cooldown-bars", type=int, default=DEFAULT_COOLDOWN_BARS)
    parser.add_argument("--max-hold-bars", type=int, default=DEFAULT_MAX_HOLD_BARS)
    parser.add_argument("--no-downsample", action="store_true")
    args = parser.parse_args()
    setup_logging()

    root = _ROOT
    data_dir = root / "data" / "training" / "lgb" / args.symbol
    m5 = load_vendor_csv(data_dir / f"{args.symbol}_M5.csv")
    feats = pd.read_parquet(data_dir / f"{args.symbol}_features.parquet")

    lab_path = root / args.labels
    lab_df = pd.read_csv(lab_path, index_col=0, parse_dates=True)
    if "label" not in lab_df.columns:
        raise SystemExit(f"{lab_path} must contain 'label' column")
    labels = lab_df["label"]

    logger.info("labels from %s positive=%.2f%%", lab_path, 100 * (labels == 1).mean())

    ver, meta_name, config_name, manifest_name, report_name, label_mode = _STAGE_CFG[args.acceptance_stage]
    result = run_binary_training(
        args.symbol, m5, feats, labels,
        quick=args.quick,
        acceptance_stage=args.acceptance_stage,
        label_horizon=args.horizon,
        gain_threshold=args.gain,
        target_precision=args.target_precision,
        target_recall=args.target_recall,
        max_signals_per_day=args.max_signals_per_day,
        cooldown_bars=args.cooldown_bars,
        max_hold_bars=args.max_hold_bars,
        use_downsample=False if args.no_downsample or args.acceptance_stage in ("v6", "v61") else None,
    )

    out_dir = root / "models" / args.symbol / "lgb"
    save_binary_model(
        result, out_dir,
        model_name=(root / args.output).name,
        meta_name=meta_name,
        config_name=config_name,
        manifest_name=manifest_name,
        version=ver,
        cooldown_bars=args.cooldown_bars,
        max_hold_bars=args.max_hold_bars,
        label_mode=label_mode,
    )

    report_dir = root / "data" / "training" / "reports" / "lgb" / args.symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / report_name
    report_path.write_text(json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("%s passed=%s failures=%s", ver, result.report.passed, result.report.failures)
    logger.info("val=%s", result.report.metrics.get("validation"))
    logger.info("test1=%s", result.report.metrics.get("test1"))
    logger.info("threshold=%.3f", result.threshold.threshold)
    return 0 if result.report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
