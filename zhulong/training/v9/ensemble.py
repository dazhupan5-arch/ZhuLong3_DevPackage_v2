"""v9 双分类集成（XGB + LGB 软投票）。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from zhulong.training.lgb.acceptance import evaluate_lgb_acceptance
from zhulong.training.lgb.splits import split_indices
from zhulong.training.lgb.train_binary import format_threshold_table, tune_threshold_binary
from zhulong.training.v9.backtest import V9_COOLDOWN_BARS, V9_MAX_HOLD, backtest_v9

logger = logging.getLogger(__name__)

XGB_CLS_PARAMS = {
    "objective": "binary:logistic",
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
    "eval_metric": "auc",
}


@dataclass
class V9EnsembleResult:
    xgb_model: xgb.XGBClassifier
    lgb_model: lgb.LGBMClassifier | lgb.Booster
    threshold: float
    feature_columns: list[str]
    report: Any
    val_metrics: dict[str, float]


def ensemble_proba(xgb_p: np.ndarray, lgb_p: np.ndarray, w: float = 0.5) -> np.ndarray:
    return w * xgb_p + (1.0 - w) * lgb_p


def train_xgb_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    quick: bool = False,
) -> xgb.XGBClassifier:
    pos = max(int(y_tr.sum()), 1)
    neg = max(int((y_tr == 0).sum()), 1)
    scale = (neg / pos) * 0.5 if pos / len(y_tr) < 0.2 else 1.0
    n_est = 100 if quick else XGB_CLS_PARAMS["n_estimators"]
    model = xgb.XGBClassifier(
        **{**XGB_CLS_PARAMS, "n_estimators": n_est, "scale_pos_weight": scale}
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def load_lgb_classifier(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def run_v9_pipeline(
    feats: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
    m5: pd.DataFrame,
    lgb_path: Path,
    quick: bool = False,
    xgb_weight: float = 0.5,
) -> V9EnsembleResult:
    aligned = feats.join(labels[["label_cls"]], how="inner")
    splits = split_indices(aligned.index)
    tr_ix = splits.train.intersection(aligned.index)
    va_ix = splits.val.intersection(aligned.index)
    te_ix = splits.test1.intersection(aligned.index)

    X_tr = aligned.loc[tr_ix, feature_columns].to_numpy(dtype=np.float32)
    y_tr = aligned.loc[tr_ix, "label_cls"].to_numpy(dtype=np.int8)
    X_va = aligned.loc[va_ix, feature_columns].to_numpy(dtype=np.float32)
    y_va = aligned.loc[va_ix, "label_cls"].to_numpy(dtype=np.int8)
    X_te = aligned.loc[te_ix, feature_columns].to_numpy(dtype=np.float32)

    xgb_model = train_xgb_classifier(X_tr, y_tr, X_va, y_va, quick=quick)
    lgb_booster = load_lgb_classifier(lgb_path)

    xgb_va = xgb_model.predict_proba(X_va)[:, 1]
    lgb_va = lgb_booster.predict(X_va)
    final_va = ensemble_proba(xgb_va, lgb_va, xgb_weight)

    thr_res, rows = tune_threshold_binary(
        y_va.astype(int),
        final_va,
        va_ix,
        target_precision=0.55,
        target_recall=0.10,
        max_signals_per_day=8.0,
        sweep_hi=0.85,
    )
    logger.info("v9 threshold table:\n%s", format_threshold_table(rows))

    val_pred = (final_va >= thr_res.threshold).astype(int)
    val_metrics = {
        "precision": float(precision_score(y_va, val_pred, zero_division=0)),
        "recall": float(recall_score(y_va, val_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_va, final_va)),
        "n_signals": int(val_pred.sum()),
        "signals_per_day": thr_res.signals_per_day,
    }

    xgb_te = xgb_model.predict_proba(X_te)[:, 1]
    lgb_te = lgb_booster.predict(X_te)
    final_te = ensemble_proba(xgb_te, lgb_te, xgb_weight)
    dirs_te = np.where(final_te >= thr_res.threshold, 1, 0)

    test1_bt = backtest_v9(
        m5,
        te_ix,
        dirs_te,
        threshold=thr_res.threshold,
        max_hold=V9_MAX_HOLD,
        cooldown_bars=V9_COOLDOWN_BARS,
        max_daily_signals=10,
        use_trailing_stop=False,
    )
    report = evaluate_lgb_acceptance(val_metrics, test1_bt, {}, stage="v9")
    report.metrics["threshold"] = thr_res.__dict__

    return V9EnsembleResult(
        xgb_model=xgb_model,
        lgb_model=lgb_booster,
        threshold=thr_res.threshold,
        feature_columns=feature_columns,
        report=report,
        val_metrics=val_metrics,
    )


def save_v9_models(result: V9EnsembleResult, out_dir: Path, lgb_source: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.xgb_model.save_model(str(out_dir / "xgb_classifier.json"))
    joblib.dump(
        {
            "feature_columns": result.feature_columns,
            "threshold": result.threshold,
            "xgb_weight": 0.5,
            "lgb_model_path": str(lgb_source),
            "max_hold_bars": V9_MAX_HOLD,
            "cooldown_bars": V9_COOLDOWN_BARS,
            "max_daily_signals": 10,
            "version": "v9",
        },
        out_dir / "v9_meta.pkl",
    )
    (out_dir / "config_v9.json").write_text(
        json.dumps({"threshold": result.threshold, "passed": result.report.passed}, indent=2),
        encoding="utf-8",
    )
