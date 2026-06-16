#!/usr/bin/env python3
"""V15 XGBoost：V14 稳定管线 + stress 跌日门槛；KN1 另用 76 维特征蒸馏。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, recall_score

from zhulong.training.lgb.acceptance import evaluate_lgb_acceptance
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
from zhulong.training.lgb.labels import generate_direction_labels
from zhulong.training.v10.backtest import backtest_both

from train_v14 import (
    MIN_ATR_PCT_V14,
    V14_COOLDOWN,
    V14_GAIN,
    V14_MAX_DAILY,
    V14_MAX_HOLD,
    V14_XGB_PARAMS,
    _detect_gpu_params,
    atr_filter,
    class_sample_weights,
    proba_to_directions_v14,
    to_multiclass_v14,
    tune_thresholds_v14,
)

logger = logging.getLogger(__name__)

TRAIN_END = pd.Timestamp("2022-12-31 23:59:59")
VAL2023_START = pd.Timestamp("2023-01-01")
VAL2023_END = pd.Timestamp("2023-12-31 23:59:59")
VAL2024_START = pd.Timestamp("2024-01-01")
VAL2024_END = pd.Timestamp("2024-12-31 23:59:59")
TEST2025_START = pd.Timestamp("2025-01-01")
TEST2025_END = pd.Timestamp("2025-12-31 23:59:59")

V15_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 400,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_lambda": 8.0,
    "reg_alpha": 2.0,
    "min_child_weight": 7,
    "gamma": 0.2,
    "random_state": 42,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 50,
}


def class_sample_weights_v15(y: np.ndarray, stress: np.ndarray) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float64)
    for label in (0, 1, 2):
        cnt = max(int((y == label).sum()), 1)
        w = len(y) / (3.0 * cnt)
        if label in (1, 2):
            w *= 1.5
        weights[y == label] = w
    weights[stress & (y == 2)] *= 1.5
    return weights

def eval_stress_days(
    m5: pd.DataFrame,
    model: xgb.XGBClassifier,
    cols: list[str],
    long_thr: float,
    short_thr: float,
    *,
    aligned_labels: pd.DataFrame | None = None,
    top_n: int = 30,
    year_end: str = "2025-12-31",
) -> dict:
    """2018-2025 最大跌幅日集合（lockbox 2026 不在此）。"""
    sub = m5.loc[:year_end].copy()
    sub["day"] = sub.index.normalize()
    daily = sub.groupby("day")["close"].agg(["first", "last"])
    daily["ret"] = (daily["last"] - daily["first"]) / daily["first"]
    worst = daily.nsmallest(top_n, "ret").index

    if aligned_labels is None:
        feats = compute_features_v15(sub)
        raw, _ = generate_labels_v15(sub.loc[feats.index])
        aligned = feats.join(
            pd.Series(to_multiclass_v15(raw), index=feats.index, name="label"),
            how="inner",
        )
    else:
        aligned = aligned_labels
    hit_short = []
    hit_long = []
    for day in worst:
        ix = aligned.index[aligned.index.normalize() == day]
        if len(ix) < 20:
            continue
        x = aligned.loc[ix, cols]
        y = aligned.loc[ix, "label"].values.astype(int)
        proba = model.predict_proba(x.values.astype(np.float32))
        argmax = np.argmax(proba, axis=1)
        gt_short = y == 2
        if gt_short.any():
            hit_short.append(float((argmax[gt_short] == 2).mean()))
        gt_long = y == 1
        if gt_long.any():
            hit_long.append(float((argmax[gt_long] == 1).mean()))

    return {
        "stress_days": len(worst),
        "evaluated_days": len(hit_short),
        "gt_short_argmax_hit_mean": float(np.mean(hit_short)) if hit_short else 0.0,
        "gt_long_argmax_hit_mean": float(np.mean(hit_long)) if hit_long else 0.0,
    }


@dataclass
class V15TrainResult:
    model: xgb.XGBClassifier
    feature_columns: list[str]
    long_threshold: float
    short_threshold: float
    report: object
    clf_report: str


def run_v15_training(symbol: str = "XAUUSD", quick: bool = False) -> V15TrainResult:
    """V15 XGB = V14 稳定管线 + 2025 截断 + stress 跌日门槛（KN 另用 76 维）。"""
    m5_path = _ROOT / "data" / "training" / "lgb" / symbol / f"{symbol}_M5.csv"
    m5 = load_vendor_csv(m5_path).loc[:TEST2025_END]
    logger.info("M5 loaded: %d bars, %s ~ %s", len(m5), m5.index[0], m5.index[-1])

    feat_cache = _ROOT / "data" / "training" / "v14" / symbol / "features.parquet"
    if feat_cache.is_file() and not quick:
        feats = pd.read_parquet(feat_cache).reindex(m5.index).dropna(how="any")
    else:
        feats = compute_features(m5, include_mtf=True, include_reversal=True)
        feat_cache.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(feat_cache)
    logger.info("features: %d rows x %d cols", len(feats), feats.shape[1])

    raw_labels = generate_direction_labels(m5.loc[feats.index], horizon=12, gain_threshold=V14_GAIN)
    labels = pd.Series(to_multiclass_v14(raw_labels), index=feats.index, name="label")
    aligned = feats.join(labels, how="inner").dropna(subset=["label"])
    cols = list(FEATURE_COLUMNS_LGB_V13)

    train_ix = aligned.index[aligned.index <= TRAIN_END]
    val23_ix = aligned.index[(aligned.index >= VAL2023_START) & (aligned.index <= VAL2023_END)]
    val24_ix = aligned.index[(aligned.index >= VAL2024_START) & (aligned.index <= VAL2024_END)]
    test25_ix = aligned.index[(aligned.index >= TEST2025_START) & (aligned.index <= TEST2025_END)]

    X_tr, y_tr = aligned.loc[train_ix, cols], aligned.loc[train_ix, "label"].values.astype(int)
    X_v23, y_v23 = aligned.loc[val23_ix, cols], aligned.loc[val23_ix, "label"].values.astype(int)
    X_v24, y_v24 = aligned.loc[val24_ix, cols], aligned.loc[val24_ix, "label"].values.astype(int)

    gpu = _detect_gpu_params()
    n_est = 150 if quick else V14_XGB_PARAMS["n_estimators"]
    model = xgb.XGBClassifier(**{**V14_XGB_PARAMS, **gpu, "n_estimators": n_est})
    model.fit(
        X_tr, y_tr,
        sample_weight=class_sample_weights(y_tr),
        eval_set=[(X_v23, y_v23)],
        verbose=50 if not quick else False,
    )

    clf_rep = classification_report(
        y_v23, model.predict(X_v23), target_names=["flat", "long", "short"], zero_division=0
    )
    logger.info("Val2023:\n%s", clf_rep)

    proba_v24 = model.predict_proba(X_v24)
    best_thr, sweep = tune_thresholds_v14(
        proba_v24, y_v24, val24_ix, m5,
        lo=0.40, hi=0.72, step=0.02,
        target_precision=0.50,
        max_signals_per_day=V14_MAX_DAILY,
        min_signals_per_day=2.0,
        min_signals=50,
    )
    long_thr, short_thr = best_thr["long_threshold"], best_thr["short_threshold"]
    logger.info(
        "V15 thresholds long=%.2f short=%.2f wprec=%.3f n_long=%d n_short=%d",
        long_thr, short_thr, best_thr["weighted_precision"],
        best_thr.get("n_long", 0), best_thr.get("n_short", 0),
    )

    dirs_v24 = atr_filter(m5, val24_ix, proba_to_directions_v14(proba_v24, long_thr, short_thr))
    pred_cls_v24 = np.zeros(len(dirs_v24), dtype=int)
    pred_cls_v24[dirs_v24 == 1], pred_cls_v24[dirs_v24 == -1] = 1, 2
    sig24 = pred_cls_v24 > 0
    val_metrics = {
        "precision": best_thr["weighted_precision"],
        "recall": float(
            recall_score(y_v24[sig24], pred_cls_v24[sig24], labels=[1, 2], average="macro", zero_division=0)
        ) if sig24.any() else 0.0,
        "long_precision": best_thr["long_precision"],
        "short_precision": best_thr["short_precision"],
        "n_signals": int(sig24.sum()),
    }

    proba_t25 = model.predict_proba(aligned.loc[test25_ix, cols])
    dirs_t25 = atr_filter(m5, test25_ix, proba_to_directions_v14(proba_t25, long_thr, short_thr))
    oos_bt = backtest_both(
        m5, test25_ix, dirs_t25,
        max_hold=V14_MAX_HOLD, cooldown_bars=V14_COOLDOWN, max_daily_signals=V14_MAX_DAILY,
    )
    logger.info("OOS 2025: %s", json.dumps(oos_bt, indent=2, default=str))

    stress = eval_stress_days(m5, model, cols, long_thr, short_thr, aligned_labels=aligned)
    logger.info("Stress pool: %s", stress)

    report = evaluate_lgb_acceptance(val_metrics, oos_bt, {}, stage="v15")
    report.metrics.update({"thresholds": best_thr, "stress_pool": stress, "model_version": "v15"})

    failures = list(report.failures)
    if int(oos_bt.get("n_trades", 0)) < 80:
        failures.append(f"n_trades={oos_bt.get('n_trades', 0)}<80")
    if best_thr.get("short_precision", 0) < 0.38:
        failures.append(f"short_precision={best_thr.get('short_precision', 0):.3f}<0.38")
    if stress.get("gt_short_argmax_hit_mean", 0) < 0.32:
        failures.append(f"stress_short_hit={stress.get('gt_short_argmax_hit_mean', 0):.3f}<0.32")
    report.failures = failures
    report.passed = len(failures) == 0

    return V15TrainResult(model, cols, long_thr, short_thr, report, clf_rep)


def save_v15_model(result: V15TrainResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_model(str(out_dir / "xgb_v15.json"))
    meta = {
        "params": V14_XGB_PARAMS,
        "feature_columns": result.feature_columns,
        "long_threshold": result.long_threshold,
        "short_threshold": result.short_threshold,
        "column_mapping": {"flat": 0, "long": 1, "short": 2},
        "model_version": "v15",
    }
    joblib.dump(meta, out_dir / "v15_meta.pkl")
    (out_dir / "feature_columns.json").write_text(
        json.dumps(result.feature_columns, indent=2),
        encoding="utf-8",
    )
    (out_dir / "config_v15.json").write_text(
        json.dumps(
            {
                "long_threshold": result.long_threshold,
                "short_threshold": result.short_threshold,
                "passed": result.report.passed,
                "n_features": len(result.feature_columns),
                "model_version": "v15",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="V15 XGBoost 训练")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = run_v15_training(symbol=args.symbol, quick=args.quick)
    out_dir = _ROOT / "models" / args.symbol / "v15"
    save_v15_model(result, out_dir)

    report_dir = _ROOT / "data" / "training" / "reports" / "v15" / args.symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "train_report_v15.json").write_text(
        json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / "classification_report_v15.txt").write_text(result.clf_report, encoding="utf-8")

    logger.info("V15 passed=%s failures=%s", result.report.passed, result.report.failures)
    return 0 if result.report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
