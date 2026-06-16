#!/usr/bin/env python3
"""v11 三分类 XGBoost 训练。"""

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
from zhulong.training.v11.train import run_v11_training, save_v11_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    root = _ROOT
    data_v8 = root / "data" / "training" / "v8" / args.symbol
    cols = json.loads((data_v8 / "feature_columns.json").read_text(encoding="utf-8"))
    feats = pd.read_parquet(data_v8 / "features.parquet")
    labels = pd.read_csv(root / "data" / "training" / f"{args.symbol}_labeled_triple.csv", index_col=0, parse_dates=True)["label"]
    train_bal = pd.read_csv(root / "data" / "train_balanced_triple.csv")
    m5 = load_vendor_csv(root / "data" / "training" / "lgb" / args.symbol / f"{args.symbol}_M5.csv")

    result = run_v11_training(feats, labels, cols, m5, train_bal, quick=args.quick)
    out_dir = root / "models" / args.symbol / "v11"
    save_v11_model(result, out_dir)

    report_dir = root / "data" / "training" / "reports" / "v11" / args.symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "acceptance_report_v11.json").write_text(
        json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "classification_report.txt").write_text(result.clf_report, encoding="utf-8")

    logger.info("v11 passed=%s failures=%s", result.report.passed, result.report.failures)
    logger.info("test1=%s", result.report.metrics.get("test1"))
    return 0 if result.report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
