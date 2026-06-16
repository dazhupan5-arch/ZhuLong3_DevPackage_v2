"""v11 XGBoost 三分类训练与阈值。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score

from zhulong.training.lgb.acceptance import evaluate_lgb_acceptance
from zhulong.training.lgb.splits import split_indices
from zhulong.training.v10.backtest import backtest_both
from zhulong.training.v11.labels import DEFAULT_GAIN, DEFAULT_HORIZON

logger = logging.getLogger(__name__)

V11_MAX_HOLD = 12
V11_COOLDOWN = 18
V11_MAX_DAILY = 10

XGB_TRIPLE_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "reg_alpha": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "eval_metric": "mlogloss",
}


@dataclass
class TripleThresholds:
    long_thr: float
    short_thr: float
    long_precision: float
    short_precision: float
    weighted_precision: float
    signals_per_day: float


def proba_to_directions(
    proba: np.ndarray,
    long_thr: float,
    short_thr: float,
) -> np.ndarray:
    """proba columns: 0=flat, 1=long, 2=short"""
    n = len(proba)
    dirs = np.zeros(n, dtype=np.int8)
    p0, p1, p2 = proba[:, 0], proba[:, 1], proba[:, 2]
    for i in range(n):
        if p1[i] >= long_thr and p1[i] >= p2[i] and p1[i] > p0[i]:
            dirs[i] = 1
        elif p2[i] >= short_thr and p2[i] >= p1[i] and p2[i] > p0[i]:
            dirs[i] = -1
    return dirs


def tune_triple_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    times: pd.DatetimeIndex,
    lo: float = 0.40,
    hi: float = 0.80,
    step: float = 0.05,
    max_signals_per_day: float = 8.0,
    target_precision: float = 0.50,
) -> tuple[TripleThresholds, list[dict]]:
    days = max((times.max() - times.min()).total_seconds() / 86400.0, 1.0)
    rows: list[dict] = []
    best: TripleThresholds | None = None
    best_score = -1.0

    for thr in np.arange(lo, hi + step * 0.5, step):
        dirs = proba_to_directions(proba, thr, thr)
        pred_cls = np.zeros(len(dirs), dtype=int)
        pred_cls[dirs == 1] = 1
        pred_cls[dirs == -1] = 2
        mask = pred_cls > 0
        n_sig = int(mask.sum())
        if n_sig == 0:
            continue
        prec = float(precision_score(y_true[mask], pred_cls[mask], average="macro", zero_division=0))
        spd = n_sig / days
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
        if wprec >= target_precision and spd <= max_signals_per_day:
            score = wprec - spd * 0.01
            if score > best_score:
                best_score = score
                best = TripleThresholds(thr, thr, lp, sp, wprec, spd)

    if best is None and rows:
        r = max(rows, key=lambda x: (x["weighted_precision"], -x["signals_per_day"]))
        best = TripleThresholds(
            r["threshold"], r["threshold"], r["long_precision"], r["short_precision"],
            r["weighted_precision"], r["signals_per_day"],
        )
    if best is None:
        best = TripleThresholds(0.55, 0.55, 0.0, 0.0, 0.0, 0.0)
    return best, rows


def train_triple_xgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    quick: bool = False,
) -> xgb.XGBClassifier:
    n_est = 100 if quick else XGB_TRIPLE_PARAMS["n_estimators"]
    model = xgb.XGBClassifier(**{**XGB_TRIPLE_PARAMS, "n_estimators": n_est})
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


@dataclass
class V11TrainResult:
    model: xgb.XGBClassifier
    thresholds: TripleThresholds
    feature_columns: list[str]
    report: Any
    clf_report: str


def run_v11_training(
    feats: pd.DataFrame,
    labels: pd.Series,
    feature_columns: list[str],
    m5: pd.DataFrame,
    train_balanced: pd.DataFrame,
    quick: bool = False,
) -> V11TrainResult:
    aligned = feats.join(labels.rename("label"), how="inner")
    splits = split_indices(aligned.index)
    va_ix = splits.val.intersection(aligned.index)
    te_ix = splits.test1.intersection(aligned.index)

    cols = feature_columns
    X_tr = train_balanced[cols]
    y_tr = train_balanced["label"].values.astype(int)
    X_va = aligned.loc[va_ix, cols]
    y_va = aligned.loc[va_ix, "label"].values.astype(int)

    model = train_triple_xgb(X_tr, y_tr, X_va, y_va, quick=quick)
    proba_va = model.predict_proba(X_va)
    y_pred = model.predict(X_va)
    clf_rep = classification_report(y_va, y_pred, target_names=["flat", "long", "short"], zero_division=0)
    logger.info("val classification:\n%s", clf_rep)

    thr, sweep = tune_triple_thresholds(proba_va, y_va, va_ix)
    logger.info(
        "thresholds long/short=%.2f wprec=%.3f sig/day=%.1f",
        thr.long_thr, thr.weighted_precision, thr.signals_per_day,
    )

    proba_te = model.predict_proba(aligned.loc[te_ix, cols])
    dirs_te = proba_to_directions(proba_te, thr.long_thr, thr.short_thr)
    test1_bt = backtest_both(m5, te_ix, dirs_te, max_hold=V11_MAX_HOLD, cooldown_bars=V11_COOLDOWN, max_daily_signals=V11_MAX_DAILY)

    dirs_va = proba_to_directions(proba_va, thr.long_thr, thr.short_thr)
    val_metrics = {
        "precision": thr.weighted_precision,
        "recall": 0.0,
        "long_precision": thr.long_precision,
        "short_precision": thr.short_precision,
        "n_signals": int((dirs_va != 0).sum()),
    }

    report = evaluate_lgb_acceptance(val_metrics, test1_bt, {}, stage="v11")
    report.metrics["thresholds"] = thr.__dict__
    report.metrics["threshold_sweep"] = sweep

    return V11TrainResult(model, thr, cols, report, clf_rep)


def save_v11_model(result: V11TrainResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_model(str(out_dir / "xgb_triple.json"))
    joblib.dump(
        {
            "feature_columns": result.feature_columns,
            "long_threshold": result.thresholds.long_thr,
            "short_threshold": result.thresholds.short_thr,
            "max_hold_bars": V11_MAX_HOLD,
            "cooldown_bars": V11_COOLDOWN,
            "horizon": DEFAULT_HORIZON,
            "gain": DEFAULT_GAIN,
        },
        out_dir / "v11_meta.pkl",
    )
    (out_dir / "config_v11.json").write_text(
        json.dumps(
            {
                "long_threshold": result.thresholds.long_thr,
                "short_threshold": result.thresholds.short_thr,
                "passed": result.report.passed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
