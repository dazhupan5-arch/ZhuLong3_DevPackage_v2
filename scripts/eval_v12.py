#!/usr/bin/env python3
"""v12 评估：v11 模型 + 不对称后处理（不重训）。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import pandas as pd
import xgboost as xgb

from zhulong.training.lgb.acceptance import evaluate_lgb_acceptance
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.splits import split_indices
from zhulong.training.v11.train import proba_to_directions
from zhulong.training.v12.backtest import (
    V12_LONG_THR,
    V12_SHORT_THR,
    backtest_v12,
    val_weighted_precision,
)

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--split", choices=["val", "test1"], default="test1")
    parser.add_argument("--model-version", choices=["v11", "v12"], default="v11")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = _ROOT
    meta = joblib.load(root / "models" / args.symbol / args.model_version / f"{args.model_version}_meta.pkl")
    cols = meta["feature_columns"]

    m5 = load_vendor_csv(root / "data" / "training" / "lgb" / args.symbol / f"{args.symbol}_M5.csv")
    feats = pd.read_parquet(root / "data" / "training" / "v8" / args.symbol / "features.parquet")
    lab = pd.read_csv(root / "data" / "training" / f"{args.symbol}_labeled_triple.csv", index_col=0, parse_dates=True)
    splits = split_indices(feats.index)
    va_ix = splits.val.intersection(feats.index)
    te_ix = splits.test1.intersection(feats.index)

    model = xgb.XGBClassifier()
    model.load_model(str(root / "models" / args.symbol / args.model_version / "xgb_triple.json"))

    proba_va = model.predict_proba(feats.loc[va_ix, cols])
    y_va = lab.loc[va_ix, "label"].values.astype(int)
    val_m = val_weighted_precision(proba_va, y_va, va_ix, m5, feats, V12_LONG_THR, V12_SHORT_THR)

    ix = va_ix if args.split == "val" else te_ix
    proba = model.predict_proba(feats.loc[ix, cols])
    dirs = proba_to_directions(proba, V12_LONG_THR, V12_SHORT_THR)
    bt = backtest_v12(m5, feats, ix, dirs)

    report_obj = evaluate_lgb_acceptance(val_m, bt, {}, stage="v12")
    # v12 额外检查做空胜率
    failures = list(report_obj.failures)
    if bt.get("short_win_rate", 0) < 0.50 and bt.get("n_short", 0) > 0:
        failures.append(f"test1_short_win_rate={bt.get('short_win_rate', 0):.3f}<0.50")
    report_obj.failures = failures
    report_obj.passed = len(failures) == 0
    report_obj.metrics["validation"] = val_m
    report_obj.metrics["test1"] = bt
    report_obj.metrics["params"] = {
        "model_version": args.model_version,
        "long_threshold": V12_LONG_THR,
        "short_threshold": V12_SHORT_THR,
        "long_sl": 1.2,
        "short_sl": 1.0,
        "long_cooldown_bars": 18,
        "short_cooldown_bars": 24,
        "use_breakeven": False,
    }

    out_dir = root / "data" / "training" / "reports" / "v12" / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.model_version}" if args.model_version != "v11" else ""
    (out_dir / f"acceptance_report_v12{suffix}.json").write_text(
        json.dumps(report_obj.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"backtest_{args.split}_v12{suffix}.json").write_text(
        json.dumps({"backtest": bt, "val": val_m}, indent=2), encoding="utf-8"
    )

    print(f"=== v12 {args.split} model={args.model_version} long_thr={V12_LONG_THR} short_thr={V12_SHORT_THR} ===")
    print(f"val weighted_precision={val_m['precision']:.3f} sig/day={val_m['signals_per_day']:.1f}")
    for k, v in bt.items():
        print(f"  {k}: {v}")
    print(f"passed={report_obj.passed} failures={failures}")
    return 0 if report_obj.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
