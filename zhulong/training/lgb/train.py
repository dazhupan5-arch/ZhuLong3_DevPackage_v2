"""LightGBM v4 多分类训练、阈值调优、评估。"""

from __future__ import annotations

import itertools
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from zhulong.training.lgb.acceptance import (
    LgbAcceptanceReport,
    evaluate_lgb_acceptance,
    get_thresholds,
)
from zhulong.training.lgb.backtest import MAX_HOLD_BARS, backtest_signals
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB
from zhulong.training.lgb.splits import DataSplits, split_indices

EARLY_STOPPING_ROUNDS = 50
CLASS_SHORT = 0
CLASS_FLAT = 1
CLASS_LONG = 2

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    n_signals: int
    signals_per_day: float = 0.0


def _signals_per_day(times: pd.DatetimeIndex, n_signals: int) -> float:
    if n_signals == 0 or len(times) == 0:
        return 0.0
    days = max((times.max() - times.min()).total_seconds() / 86400.0, 1.0)
    return n_signals / days


def to_multiclass(labels: np.ndarray) -> np.ndarray:
    """-1/0/1 -> 0/1/2"""
    out = np.ones(len(labels), dtype=int)
    out[labels == -1] = CLASS_SHORT
    out[labels == 0] = CLASS_FLAT
    out[labels == 1] = CLASS_LONG
    return out


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_pred.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_signals": 0}
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_signals": int(y_pred.sum()),
    }


def threshold_sweep(
    y_true: np.ndarray,
    proba: np.ndarray,
    val_times: pd.DatetimeIndex,
    lo: float = 0.5,
    hi: float = 0.95,
    step: float = 0.02,
) -> list[ThresholdResult]:
    rows: list[ThresholdResult] = []
    for thr in np.arange(lo, hi + step * 0.5, step):
        pred = (proba >= thr).astype(int)
        m = _clf_metrics(y_true, pred)
        rows.append(
            ThresholdResult(
                threshold=float(round(thr, 4)),
                precision=m["precision"],
                recall=m["recall"],
                f1=m["f1"],
                n_signals=m["n_signals"],
                signals_per_day=_signals_per_day(val_times, m["n_signals"]),
            )
        )
    return rows


def format_threshold_table(rows: list[ThresholdResult]) -> str:
    lines = ["Threshold | Precision | Recall | Signals/day", "-" * 50]
    for r in rows:
        lines.append(
            f"{r.threshold:.2f}      | {r.precision:.2f}      | {r.recall:.2f}   | {r.signals_per_day:.1f}"
        )
    return "\n".join(lines)


def tune_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    val_times: pd.DatetimeIndex,
    target_precision: float = 0.55,
    max_signals_per_day: float = 12.0,
    min_precision: float = 0.50,
) -> tuple[ThresholdResult, list[ThresholdResult]]:
    rows = threshold_sweep(y_true, proba, val_times)

    candidates = [
        r for r in rows
        if r.precision >= target_precision
        and r.signals_per_day <= max_signals_per_day
        and r.n_signals > 0
    ]
    if candidates:
        return max(candidates, key=lambda r: (r.recall, r.precision)), rows

    candidates = [
        r for r in rows
        if r.precision >= min_precision
        and r.signals_per_day <= max_signals_per_day
        and r.n_signals > 0
    ]
    if candidates:
        return max(candidates, key=lambda r: (r.precision, r.recall)), rows

    candidates = [r for r in rows if r.signals_per_day <= max_signals_per_day and r.n_signals > 0]
    if candidates:
        return max(candidates, key=lambda r: (r.precision, r.recall)), rows

    if rows:
        return max(rows, key=lambda r: r.precision), rows
    return ThresholdResult(0.75, 0, 0, 0, 0), rows


def _param_grid(quick: bool = False, full_grid: bool = False) -> list[dict[str, Any]]:
    base = {
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.03, 0.05, 0.07],
        "n_estimators": [500, 1000, 1500],
        "min_child_samples": [100, 200, 500],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_lambda": [1.0, 3.0, 5.0],
    }
    if quick:
        base = {
            "num_leaves": [31, 63],
            "learning_rate": [0.05, 0.07],
            "n_estimators": [500, 1000],
            "min_child_samples": [100, 200],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1.0, 3.0],
        }
        max_configs = 16
    elif full_grid:
        max_configs = 128
    else:
        max_configs = 48
    keys = list(base.keys())
    combos = [dict(zip(keys, c)) for c in itertools.product(*[base[k] for k in keys])]
    random.shuffle(combos)
    return combos[:max_configs]


def train_multiclass_lgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    params: dict[str, Any],
) -> lgb.LGBMClassifier:
    p = dict(params)
    est = p.pop("n_estimators", 1000)
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=est,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **p,
    )
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return clf


def predict_directions(
    proba: np.ndarray,
    thr_long: float,
    thr_short: float,
) -> np.ndarray:
    """proba columns: [short, flat, long]"""
    p_s = proba[:, CLASS_SHORT]
    p_l = proba[:, CLASS_LONG]
    directions = np.zeros(len(proba), dtype=int)
    long_ok = (p_l >= thr_long) & (p_l >= p_s)
    short_ok = (p_s >= thr_short) & (p_s > p_l)
    directions[long_ok] = 1
    directions[short_ok] = -1
    return directions


def grid_search_multiclass(
    aligned: pd.DataFrame,
    splits: DataSplits,
    quick: bool = False,
    full_grid: bool = False,
    target_precision: float = 0.50,
    max_signals_per_day: float = 8.0,
) -> tuple[lgb.LGBMClassifier, dict[str, Any], ThresholdResult, ThresholdResult, list[ThresholdResult], list[ThresholdResult]]:
    tr_ix = splits.train.intersection(aligned.index)
    va_ix = splits.val.intersection(aligned.index)
    y_tr = to_multiclass(aligned.loc[tr_ix, "label"].values)
    y_va = to_multiclass(aligned.loc[va_ix, "label"].values)
    y_va_long = (aligned.loc[va_ix, "label"].values == 1).astype(int)
    y_va_short = (aligned.loc[va_ix, "label"].values == -1).astype(int)
    X_tr = aligned.loc[tr_ix, FEATURE_COLUMNS_LGB]
    X_va = aligned.loc[va_ix, FEATURE_COLUMNS_LGB]

    dist = {int(c): int((y_tr == c).sum()) for c in (0, 1, 2)}
    logger.info("multiclass train dist=%s val=%s", dist, {int(c): int((y_va == c).sum()) for c in (0, 1, 2)})

    best_clf: lgb.LGBMClassifier | None = None
    best_cfg: dict[str, Any] = {}
    best_thr_long = ThresholdResult(0.75, 0, 0, 0, 0)
    best_thr_short = ThresholdResult(0.75, 0, 0, 0, 0)
    best_rows_l: list[ThresholdResult] = []
    best_rows_s: list[ThresholdResult] = []
    best_score = -1.0

    per_side_max = max_signals_per_day / 2.0

    for cfg in _param_grid(quick, full_grid):
        try:
            clf = train_multiclass_lgb(X_tr, y_tr, X_va, y_va, dict(cfg))
            proba = clf.predict_proba(X_va)
            thr_l, rows_l = tune_threshold(
                y_va_long, proba[:, CLASS_LONG], va_ix,
                target_precision=target_precision,
                max_signals_per_day=per_side_max,
            )
            thr_s, rows_s = tune_threshold(
                y_va_short, proba[:, CLASS_SHORT], va_ix,
                target_precision=target_precision,
                max_signals_per_day=per_side_max,
            )
            dirs = predict_directions(proba, thr_l.threshold, thr_s.threshold)
            y_dir = aligned.loc[va_ix, "label"].values
            active = dirs != 0
            val_prec = float((dirs[active] == y_dir[active]).mean()) if active.any() else 0.0
            val_rec = float(
                (((dirs == 1) & (y_dir == 1)).sum() + ((dirs == -1) & (y_dir == -1)).sum())
                / max(((y_dir == 1) | (y_dir == -1)).sum(), 1)
            )
            spd = _signals_per_day(va_ix, int(active.sum()))
            score = val_prec * 0.5 + min(val_rec / 0.2, 1.0) * 0.2
            score += max(0.0, 1.0 - spd / max(max_signals_per_day, 1)) * 0.3
            if score > best_score:
                best_score = score
                best_clf = clf
                best_cfg = cfg
                best_thr_long = thr_l
                best_thr_short = thr_s
                best_rows_l = rows_l
                best_rows_s = rows_s
                logger.info(
                    "cfg=%s thr_l=%.2f thr_s=%.2f val_prec=%.3f val_rec=%.3f sig/day=%.1f logloss=%.4f",
                    cfg, thr_l.threshold, thr_s.threshold, val_prec, val_rec, spd,
                    clf.best_score_.get("valid_0", {}).get("multi_logloss", 0.0)
                    if hasattr(clf, "best_score_") else 0.0,
                )
        except Exception as ex:
            logger.warning("cfg failed: %s", ex)

    if best_clf is None:
        raise RuntimeError("no valid multiclass LGB model")
    logger.info("long threshold table:\n%s", format_threshold_table(best_rows_l))
    logger.info("short threshold table:\n%s", format_threshold_table(best_rows_s))
    return best_clf, best_cfg, best_thr_long, best_thr_short, best_rows_l, best_rows_s


@dataclass
class LgbTrainResult:
    symbol: str
    clf: lgb.LGBMClassifier
    thr_long: ThresholdResult
    thr_short: ThresholdResult
    report: LgbAcceptanceReport
    model_cfg: dict[str, Any]
    label_horizon: int = 12
    gain_threshold: float = 0.0025


def run_full_training(
    symbol: str,
    m5: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    quick: bool = False,
    full_grid: bool = False,
    acceptance_stage: str = "v42",
    label_horizon: int = 12,
    gain_threshold: float = 0.0025,
    target_precision: float = 0.50,
    max_signals_per_day: float = 8.0,
) -> LgbTrainResult:
    aligned = features.join(labels.rename("label"), how="inner")
    splits = split_indices(aligned.index)
    logger.info("split sizes: %s", {k: len(getattr(splits, k)) for k in ("train", "val", "test1", "stress")})

    clf, cfg, thr_long, thr_short, _, _ = grid_search_multiclass(
        aligned, splits, quick=quick, full_grid=full_grid,
        target_precision=target_precision,
        max_signals_per_day=max_signals_per_day,
    )

    va_ix = splits.val.intersection(aligned.index)
    X_va = aligned.loc[va_ix, FEATURE_COLUMNS_LGB]
    y_va = aligned.loc[va_ix, "label"].values
    proba_va = clf.predict_proba(X_va)
    dirs_va = predict_directions(proba_va, thr_long.threshold, thr_short.threshold)
    active = dirs_va != 0
    val_cls = {
        "precision": float((dirs_va[active] == y_va[active]).mean()) if active.any() else 0.0,
        "recall": float(
            (((dirs_va == 1) & (y_va == 1)).sum() + ((dirs_va == -1) & (y_va == -1)).sum())
            / max(((y_va == 1) | (y_va == -1)).sum(), 1)
        ),
        "f1": 0.0,
        "n_signals": int(active.sum()),
        "signals_per_day": _signals_per_day(va_ix, int(active.sum())),
        "multi_logloss": float(
            clf.best_score_.get("valid_0", {}).get("multi_logloss", 0.0)
            if hasattr(clf, "best_score_") else 0.0
        ),
    }
    if val_cls["precision"] + val_cls["recall"] > 0:
        val_cls["f1"] = 2 * val_cls["precision"] * val_cls["recall"] / (
            val_cls["precision"] + val_cls["recall"] + 1e-9
        )

    test_ix = splits.test1.intersection(aligned.index)
    proba_test = clf.predict_proba(aligned.loc[test_ix, FEATURE_COLUMNS_LGB])
    dirs_test = predict_directions(proba_test, thr_long.threshold, thr_short.threshold)
    test1_bt = backtest_signals(m5, test_ix, dirs_test)

    stress_ix = splits.stress.intersection(aligned.index)
    proba_stress = clf.predict_proba(aligned.loc[stress_ix, FEATURE_COLUMNS_LGB])
    dirs_stress = predict_directions(proba_stress, thr_long.threshold, thr_short.threshold)
    stress_bt = backtest_signals(m5, stress_ix, dirs_stress)

    th = get_thresholds(acceptance_stage)
    report = evaluate_lgb_acceptance(val_cls, test1_bt, stress_bt, thresholds=th, stage=acceptance_stage)
    report.metrics["thresholds"] = {
        "long": thr_long.__dict__,
        "short": thr_short.__dict__,
    }

    return LgbTrainResult(
        symbol=symbol,
        clf=clf,
        thr_long=thr_long,
        thr_short=thr_short,
        report=report,
        model_cfg=cfg,
        label_horizon=label_horizon,
        gain_threshold=gain_threshold,
    )


def save_lgb_models(result: LgbTrainResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.clf.booster_.save_model(str(out_dir / "lgb_multiclass.txt"))
    joblib.dump(
        {
            "thr_long": result.thr_long.threshold,
            "thr_short": result.thr_short.threshold,
            "feature_columns": FEATURE_COLUMNS_LGB,
            "label_horizon": result.label_horizon,
            "gain_threshold": result.gain_threshold,
            "class_map": {"short": CLASS_SHORT, "flat": CLASS_FLAT, "long": CLASS_LONG},
            "max_hold_bars": MAX_HOLD_BARS,
        },
        out_dir / "lgb_meta.pkl",
    )
    manifest = {
        "symbol": result.symbol,
        "kind": "production" if result.report.passed else "rejected",
        "acceptance_passed": result.report.passed,
        "acceptance_stage": result.report.acceptance_stage,
        "classifier_mode": "lgb_multiclass",
        "feature_mode": "lgb_tabular",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": result.report.to_dict(),
        "model_cfg": result.model_cfg,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir
