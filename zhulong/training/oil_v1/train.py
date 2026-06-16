"""USOIL v1 XGBoost 三分类训练（网格搜索 + 早停）。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score

from zhulong.training.lgb.acceptance import evaluate_lgb_acceptance
from zhulong.training.lgb.splits import split_indices
from zhulong.training.oil_v1.backtest import (
    OIL_COOLDOWN,
    OIL_LONG_THR,
    OIL_MAX_DAILY,
    OIL_MAX_HOLD,
    OIL_SHORT_THR,
    backtest_oil_v1,
    val_weighted_precision_oil,
)
from zhulong.training.oil_v1.labels import DEFAULT_GAIN_FIXED, DEFAULT_HORIZON
from zhulong.training.v11.train import proba_to_directions

logger = logging.getLogger(__name__)

OIL_XGB_GRID = {
    "max_depth": [5, 6, 7],
    "learning_rate": [0.02, 0.03, 0.05],
    "n_estimators": [500, 800],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_lambda": [5, 8],
    "reg_alpha": [1, 2],
}

OIL_XGB_BASE = {
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "eval_metric": "mlogloss",
}
EARLY_STOP_ROUNDS = 50


@dataclass
class OilTrainResult:
    model: xgb.XGBClassifier
    best_params: dict[str, Any]
    feature_columns: list[str]
    report: Any
    clf_report: str
    long_thr: float
    short_thr: float


def _grid_keys() -> list[dict[str, Any]]:
    keys = list(OIL_XGB_GRID.keys())
    combos = []
    for vals in product(*(OIL_XGB_GRID[k] for k in keys)):
        combos.append(dict(zip(keys, vals)))
    return combos


def train_oil_xgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    quick: bool = False,
    max_combos: int = 8,
) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    combos = _grid_keys()
    if quick:
        combos = combos[:3]
    elif len(combos) > max_combos:
        rng = np.random.default_rng(42)
        pick = rng.choice(len(combos), size=max_combos, replace=False)
        combos = [combos[i] for i in sorted(pick)]
        logger.info("grid search sampled %s/%s combos", max_combos, len(_grid_keys()))
    best_model: xgb.XGBClassifier | None = None
    best_loss = float("inf")
    best_params: dict[str, Any] = {}

    for i, params in enumerate(combos):
        model = xgb.XGBClassifier(
            **{**OIL_XGB_BASE, **params, "early_stopping_rounds": EARLY_STOP_ROUNDS}
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        loss = float(model.best_score) if hasattr(model, "best_score") and model.best_score else 0.0
        # use eval loss from mlogloss
        evals = model.evals_result()
        if evals and "validation_0" in evals:
            loss = float(evals["validation_0"]["mlogloss"][-1])
        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_params = params
        if (i + 1) % 10 == 0:
            logger.info("grid %s/%s best_loss=%.4f", i + 1, len(combos), best_loss)

    assert best_model is not None
    logger.info("best params: %s loss=%.4f", best_params, best_loss)
    return best_model, best_params


def tune_oil_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    times: pd.DatetimeIndex,
    m5: pd.DataFrame,
) -> tuple[float, float, list[dict]]:
    rows: list[dict] = []
    best_lp, best_sp, best_wprec = OIL_LONG_THR, OIL_SHORT_THR, 0.0
    for lt in np.arange(0.70, 0.90, 0.02):
        for st in np.arange(0.68, 0.86, 0.02):
            m = val_weighted_precision_oil(proba, y_true, times, m5, lt, st)
            rows.append({"long_thr": lt, "short_thr": st, **m})
            if m["precision"] >= 0.50 and m["signals_per_day"] <= 8:
                if m["precision"] > best_wprec:
                    best_wprec = m["precision"]
                    best_lp, best_sp = lt, st
    if best_wprec == 0.0 and rows:
        r = max(rows, key=lambda x: (x["precision"], -x["signals_per_day"]))
        best_lp, best_sp = r["long_thr"], r["short_thr"]
    return best_lp, best_sp, rows


def run_oil_v1_training(
    feats: pd.DataFrame,
    labels: pd.Series,
    feature_columns: list[str],
    m5: pd.DataFrame,
    train_balanced: pd.DataFrame,
    quick: bool = False,
) -> OilTrainResult:
    aligned = feats.join(labels.rename("label"), how="inner")
    splits = split_indices(aligned.index)
    va_ix = splits.val.intersection(aligned.index)
    te_ix = splits.test1.intersection(aligned.index)

    cols = feature_columns
    X_tr_full = train_balanced[cols]
    y_tr_full = train_balanced["label"].values.astype(int)
    # 每类最多 60k，加速训练且保持平衡
    max_per_class = 60_000
    picks: list[int] = []
    rng = np.random.default_rng(42)
    for cls in (0, 1, 2):
        ix = np.where(y_tr_full == cls)[0]
        n = min(len(ix), max_per_class)
        picks.extend(rng.choice(ix, n, replace=False).tolist())
    rng.shuffle(picks)
    X_tr = X_tr_full.iloc[picks]
    y_tr = y_tr_full[picks]
    logger.info("train subsample: %s -> %s rows", len(y_tr_full), len(y_tr))
    X_va = aligned.loc[va_ix, cols]
    y_va = aligned.loc[va_ix, "label"].values.astype(int)

    model, best_params = train_oil_xgb(X_tr, y_tr, X_va, y_va, quick=quick)
    proba_va = model.predict_proba(X_va)
    y_pred = model.predict(X_va)
    clf_rep = classification_report(y_va, y_pred, target_names=["flat", "long", "short"], zero_division=0)
    logger.info("val classification:\n%s", clf_rep)

    long_thr, short_thr, sweep = tune_oil_thresholds(proba_va, y_va, va_ix, m5)
    logger.info("thresholds long=%.2f short=%.2f", long_thr, short_thr)

    val_m = val_weighted_precision_oil(proba_va, y_va, va_ix, m5, long_thr, short_thr)
    proba_te = model.predict_proba(aligned.loc[te_ix, cols])
    dirs_te = proba_to_directions(proba_te, long_thr, short_thr)
    test1_bt = backtest_oil_v1(m5, te_ix, dirs_te, max_hold=OIL_MAX_HOLD, cooldown=OIL_COOLDOWN, max_daily_signals=OIL_MAX_DAILY)

    report = evaluate_lgb_acceptance(val_m, test1_bt, {}, stage="oil_v1")
    report.metrics["thresholds"] = {"long_thr": long_thr, "short_thr": short_thr}
    report.metrics["threshold_sweep"] = sweep
    report.metrics["best_xgb_params"] = best_params

    return OilTrainResult(model, best_params, cols, report, clf_rep, long_thr, short_thr)


def save_oil_v1_model(result: OilTrainResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.model.save_model(str(out_dir / "xgb_triple_oil.json"))
    joblib.dump(
        {
            "feature_columns": result.feature_columns,
            "long_threshold": result.long_thr,
            "short_threshold": result.short_thr,
            "max_hold_bars": OIL_MAX_HOLD,
            "cooldown_bars": OIL_COOLDOWN,
            "horizon": DEFAULT_HORIZON,
            "gain": DEFAULT_GAIN_FIXED,
            "best_xgb_params": result.best_params,
        },
        out_dir / "oil_v1_meta.pkl",
    )
    (out_dir / "config_oil_v1.json").write_text(
        json.dumps(
            {
                "long_threshold": result.long_thr,
                "short_threshold": result.short_thr,
                "max_hold_bars": OIL_MAX_HOLD,
                "cooldown_bars": OIL_COOLDOWN,
                "long_sl_atr": 1.5,
                "short_sl_atr": 1.2,
                "tp_atr": 2.5,
                "eia_blackout_before_min": 30,
                "eia_blackout_after_min": 15,
                "passed": result.report.passed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
