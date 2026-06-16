#!/usr/bin/env python3
"""v9：XGB 二分类 + 复用 v8 LGB，双分类软投票 + v9 回测。"""

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

from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.v9.ensemble import run_v9_pipeline, save_v9_models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data/training/v8/XAUUSD")
    parser.add_argument("--lgb-model", default="models/XAUUSD/v8/lgb_classifier.txt")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    root = _ROOT
    data_dir = root / args.data_dir
    lgb_path = root / args.lgb_model

    m5 = load_vendor_csv(root / "data" / "training" / "lgb" / args.symbol / f"{args.symbol}_M5.csv")
    feats = pd.read_parquet(data_dir / "features.parquet")
    labels = pd.read_csv(data_dir / "labels.csv", index_col=0, parse_dates=True)
    cols = json.loads((data_dir / "feature_columns.json").read_text(encoding="utf-8"))

    result = run_v9_pipeline(feats, labels, cols, m5, lgb_path, quick=args.quick)
    out_dir = root / "models" / args.symbol / "v9"
    save_v9_models(result, out_dir, lgb_path)

    report_dir = root / "data" / "training" / "reports" / "v9" / args.symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "acceptance_report_v9.json").write_text(
        json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("v9 passed=%s failures=%s", result.report.passed, result.report.failures)
    logger.info("val=%s", result.val_metrics)
    logger.info("test1=%s", result.report.metrics.get("test1"))
    logger.info("threshold=%.3f", result.threshold)
    return 0 if result.report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
