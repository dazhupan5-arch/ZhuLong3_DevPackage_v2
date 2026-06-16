#!/usr/bin/env python3
"""v13：简化方向标签 (0.15%) + 反转特征 XGBoost 三分类。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score, recall_score

from zhulong.training.lgb.acceptance import evaluate_lgb_acceptance
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
from zhulong.training.lgb.labels import V13_GAIN_THRESHOLD, generate_direction_labels
from zhulong.training.lgb.train import to_multiclass
from zhulong.training.lgb.backtest import _atr_series
from zhulong.training.v10.backtest import MIN_ATR_PCT, backtest_both
from zhulong.training.v11.train import (
    V11_MAX_DAILY,
    V11_MAX_HOLD,
    V11_COOLDOWN,
    TripleThresholds,
    proba_to_directions,
)

logger = logging.getLogger(__name__)

TRAIN_END = pd.Timestamp("2024-12-31 23:59:59")
VAL_START = pd.Timestamp("2025-01-01 00:00:00")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")

V13_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.5,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 50,
}


@dataclass
class V13TrainResult:
    model: xgb.XGBClassifier
    feature_columns: list[str]
    long_threshold: float
    short_threshold: float
    report: object
    clf_report: str


def _split_indices(index: pd.DatetimeIndex) -> tuple[pd.Index, pd.Index]:
    train_ix = index[index <= TRAIN_END]
    val_ix = index[(index >= VAL_START) & (index <= VAL_END)]
    return train_ix, val_ix


def _atr_ok(m5: pd.DataFrame, times: pd.DatetimeIndex, dirs: np.ndarray) -> np.ndarray:
    """与回测一致的 ATR 过滤，剔除低波动假信号。"""
    atr = _atr_series(m5)
    close = m5["close"]
    out = dirs.copy()
    for i, t in enumerate(times):
        if out[i] == 0 or t not in m5.index:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            out[i] = 0
            continue
        a = float(atr.iloc[idx])
        c = float(close.iloc[idx])
        if a <= 0 or (a / c) < MIN_ATR_PCT:
            out[i] = 0
    return out


def tune_v13_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    times: pd.DatetimeIndex,
    m5: pd.DataFrame,
    lo: float = 0.40,
    hi: float = 0.58,
    step: float = 0.02,
    max_signals_per_day: float = 8.0,
    min_signals: int = 50,
    target_precision: float = 0.60,
) -> tuple[TripleThresholds, list[dict]]:
    """在 ATR 可交易约束下选取阈值。"""
    days = max((times.max() - times.min()).total_seconds() / 86400.0, 1.0)
    rows: list[dict] = []
    best: TripleThresholds | None = None
    best_score = -1.0

    for thr in np.arange(lo, hi + step * 0.5, step):
        dirs = _atr_ok(m5, times, proba_to_directions(proba, thr, thr))
        pred_cls = np.zeros(len(dirs), dtype=int)
        pred_cls[dirs == 1] = 1
        pred_cls[dirs == -1] = 2
        mask = pred_cls > 0
        n_sig = int(mask.sum())
        if n_sig < min_signals:
            continue
        spd = n_sig / days
        if spd > max_signals_per_day:
            continue
        long_m = dirs == 1
        short_m = dirs == -1
        lp = float((y_true[long_m] == 1).mean()) if long_m.any() else 0.0
        sp = float((y_true[short_m] == 2).mean()) if short_m.any() else 0.0
        wprec = (lp * long_m.sum() + sp * short_m.sum()) / max(n_sig, 1)
        row = {
            "threshold": float(round(thr, 3)),
            "weighted_precision": wprec,
            "long_precision": lp,
            "short_precision": sp,
            "n_signals": n_sig,
            "signals_per_day": spd,
        }
        rows.append(row)
        if wprec >= target_precision:
            score = wprec + min(n_sig / 500.0, 1.0) * 0.02
            if score > best_score:
                best_score = score
                best = TripleThresholds(thr, thr, lp, sp, wprec, spd)

    if best is None and rows:
        r = max(rows, key=lambda x: (x["weighted_precision"], x["n_signals"]))
        best = TripleThresholds(
            r["threshold"], r["threshold"], r["long_precision"], r["short_precision"],
            r["weighted_precision"], r["signals_per_day"],
        )
    if best is None:
        best = TripleThresholds(0.50, 0.50, 0.0, 0.0, 0.0, 0.0)
    return best, rows


def _class_sample_weights(y: np.ndarray) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float64)
    for label in (0, 1, 2):
        cnt = max(int((y == label).sum()), 1)
        w = len(y) / (3.0 * cnt)
        weights[y == label] = w
    return weights


def run_v13_training(
    symbol: str = "XAUUSD",
    horizon: int = 12,
    gain: float = V13_GAIN_THRESHOLD,
    quick: bool = False,
) -> V13TrainResult:
    root = _ROOT
    m5_path = root / "data" / "training" / "lgb" / symbol / f"{symbol}_M5.csv"
    if not m5_path.is_file():
        m5_path = root / "data" / "training" / f"{symbol}_M5.csv"
    m5 = load_vendor_csv(m5_path)
    feat_cache = root / "data" / "training" / "v13" / symbol / "features.parquet"
    if feat_cache.is_file():
        feats = pd.read_parquet(feat_cache)
        logger.info("loaded cached features: %s", feat_cache)
    else:
        feats = compute_features(m5, include_reversal=True)
        feat_cache.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(feat_cache)
        logger.info("cached features: %s", feat_cache)
    raw_labels = generate_direction_labels(m5.loc[feats.index], horizon=horizon, gain_threshold=gain)
    labels = pd.Series(to_multiclass(raw_labels), index=feats.index, name="label")

    aligned = feats.join(labels, how="inner")
    cols = list(FEATURE_COLUMNS_LGB_V13)
    train_ix, val_ix = _split_indices(aligned.index)
    logger.info("split train=%d val=%d", len(train_ix), len(val_ix))

    X_tr = aligned.loc[train_ix, cols]
    y_tr = aligned.loc[train_ix, "label"].values.astype(int)
    X_va = aligned.loc[val_ix, cols]
    y_va = aligned.loc[val_ix, "label"].values.astype(int)
    sample_weights = _class_sample_weights(y_tr)

    n_est = 100 if quick else V13_XGB_PARAMS["n_estimators"]
    model = xgb.XGBClassifier(**{**V13_XGB_PARAMS, "n_estimators": n_est})
    model.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weights,
        eval_set=[(X_va, y_va)],
        verbose=50 if not quick else False,
    )

    y_pred = model.predict(X_va)
    clf_rep = classification_report(
        y_va, y_pred, target_names=["flat", "long", "short"], zero_division=0
    )
    logger.info("val classification:\n%s", clf_rep)

    proba_va = model.predict_proba(X_va)
    thr, sweep = tune_v13_thresholds(
        proba_va,
        y_va,
        val_ix,
        m5,
        lo=0.40,
        hi=0.58,
        step=0.02,
        min_signals=50,
        target_precision=0.60,
    )
    logger.info(
        "thresholds long=%.2f short=%.2f wprec=%.3f sig/day=%.1f",
        thr.long_thr,
        thr.short_thr,
        thr.weighted_precision,
        thr.signals_per_day,
    )

    dirs_va = proba_to_directions(proba_va, thr.long_thr, thr.short_thr)
    pred_cls = np.zeros(len(dirs_va), dtype=int)
    pred_cls[dirs_va == 1] = 1
    pred_cls[dirs_va == -1] = 2
    sig_mask = pred_cls > 0
    macro_recall = float(
        recall_score(y_va[sig_mask], pred_cls[sig_mask], average="macro", zero_division=0)
    ) if sig_mask.any() else 0.0

    val_metrics = {
        "precision": thr.weighted_precision,
        "recall": macro_recall,
        "long_precision": thr.long_precision,
        "short_precision": thr.short_precision,
        "n_signals": int(sig_mask.sum()),
    }

    dirs_va_bt = _atr_ok(
        m5, val_ix, proba_to_directions(proba_va, thr.long_thr, thr.short_thr)
    )
    oos_bt = backtest_both(
        m5,
        val_ix,
        dirs_va_bt,
        max_hold=V11_MAX_HOLD,
        cooldown_bars=V11_COOLDOWN,
        max_daily_signals=V11_MAX_DAILY,
    )

    report = evaluate_lgb_acceptance(val_metrics, oos_bt, {}, stage="v13")
    report.metrics["thresholds"] = thr.__dict__
    report.metrics["threshold_sweep"] = sweep
    report.metrics["horizon"] = horizon
    report.metrics["gain"] = gain

    if thr.long_precision < 0.55:
        report.failures.append(f"long_precision {thr.long_precision:.3f} < 0.55")
        report.passed = False
    if thr.short_precision < 0.55:
        report.failures.append(f"short_precision {thr.short_precision:.3f} < 0.55")
        report.passed = False
    if macro_recall < 0.40:
        report.failures.append(f"recall {macro_recall:.3f} < 0.40")
        report.passed = False

    return V13TrainResult(
        model=model,
        feature_columns=cols,
        long_threshold=thr.long_thr,
        short_threshold=thr.short_thr,
        report=report,
        clf_report=clf_rep,
    )


def save_v13_model(result: V13TrainResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_model(str(out_dir / "xgb_direction.json"))
    joblib.dump(
        {
            "params": V13_XGB_PARAMS,
            "feature_columns": result.feature_columns,
            "long_threshold": result.long_threshold,
            "short_threshold": result.short_threshold,
            "max_hold_bars": V11_MAX_HOLD,
            "cooldown_bars": V11_COOLDOWN,
        },
        out_dir / "params_v13.pkl",
    )
    (out_dir / "feature_columns.json").write_text(
        json.dumps(result.feature_columns, indent=2),
        encoding="utf-8",
    )
    (out_dir / "config_v13.json").write_text(
        json.dumps(
            {
                "long_threshold": result.long_threshold,
                "short_threshold": result.short_threshold,
                "passed": result.report.passed,
                "n_features": len(result.feature_columns),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 v13 方向预测 XGBoost")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--gain", type=float, default=V13_GAIN_THRESHOLD)
    parser.add_argument("--quick", action="store_true", help="快速试跑（100 棵树）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_v13_training(
        symbol=args.symbol,
        horizon=args.horizon,
        gain=args.gain,
        quick=args.quick,
    )
    out_dir = _ROOT / "models" / args.symbol / "v13"
    save_v13_model(result, out_dir)

    report_dir = _ROOT / "data" / "training" / "reports" / "v13" / args.symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "train_report_v13.json").write_text(
        json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / "classification_report_v13.txt").write_text(result.clf_report, encoding="utf-8")

    logger.info("v13 passed=%s failures=%s", result.report.passed, result.report.failures)
    logger.info("oos=%s", result.report.metrics.get("test1"))
    return 0 if result.report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
