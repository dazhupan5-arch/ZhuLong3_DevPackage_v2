"""训练流水线：数据划分、训练、验收。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from zhulong.feature_engine import (
    FEATURE_COLUMNS,
    MTF_COLUMNS,
    build_fused_row,
    compute_m5_features,
    compute_mtf_trend_features,
    fused_feature_dim,
    precompute_hourly_backgrounds,
)
from zhulong.training.acceptance import (
    AcceptanceReport,
    AcceptanceThresholds,
    LabelParams,
    LABEL_MODE,
    OOS_BARS_2W,
    SL_ATR,
    TP_ATR,
    VAL_BARS_3M,
    VAL_FRAC,
    backtest_validation,
    directional_metrics,
    find_extreme_month,
    max_drawdown,
    predict_from_dual,
    proba_3class_from_dual,
    regressor_metrics,
    save_artifacts,
    consecutive_losses,
    _atr_series,
)
from zhulong.utils.paths import model_dir_for_symbol

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "training" / "cache"
FEATURE_MODE = "sequence_stats"
_scaler_cache: StandardScaler | None = None
_label_cache: dict[tuple, pd.DataFrame] = {}


def _load_scaler_cache() -> StandardScaler | None:
    p = _CACHE_DIR / "fused_scaler.pkl"
    if not p.is_file():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def _save_scaler_cache(scaler: StandardScaler) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, _CACHE_DIR / "fused_scaler.pkl")
    except Exception:
        pass


@dataclass
class DirectionModels:
    clf_long: xgb.XGBClassifier
    clf_short: xgb.XGBClassifier
    prob_threshold: float = 0.60
    short_max: float = 0.35

    def _cal_long(self, X: np.ndarray) -> np.ndarray:
        return self.clf_long.predict_proba(X)[:, 1]

    def _cal_short(self, X: np.ndarray) -> np.ndarray:
        return self.clf_short.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return proba_3class_from_dual(self._cal_long(X), self._cal_short(X))

    def predict_direction(self, X: np.ndarray) -> np.ndarray:
        return predict_from_dual(
            self._cal_long(X),
            self._cal_short(X),
            threshold=self.prob_threshold,
            short_max=self.short_max,
        )


@dataclass
class FeatureMatrix:
    m5: pd.DataFrame
    X_raw: np.ndarray
    bar_indices: np.ndarray
    feat_times: pd.DatetimeIndex
    H: np.ndarray
    train_end: int
    val_end: int
    oos_start: int
    fused: np.ndarray
    scaler: StandardScaler


def load_m5_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    return df.set_index("time").sort_index()


def build_labels_return_direction(
    m5: pd.DataFrame, horizon: int, threshold_pct: float = 0.15
) -> pd.DataFrame:
    key = ("return", horizon, threshold_pct)
    if key in _label_cache:
        return _label_cache[key]

    close = m5["close"].to_numpy()
    n = len(m5)
    label = np.zeros(n, dtype=int)
    future_ret = np.full(n, np.nan)
    for i in range(n - horizon):
        c = close[i]
        if c <= 0:
            continue
        ret = (close[i + horizon] - c) / c * 100.0
        future_ret[i] = ret
        if ret > threshold_pct:
            label[i] = 1
        elif ret < -threshold_pct:
            label[i] = -1

    out = pd.DataFrame(
        {"label": label, "entry_offset": future_ret / 100.0, "future_ret_pct": future_ret},
        index=m5.index,
    )
    _label_cache[key] = out
    logger.info(
        "return_direction h=%s thr=%.2f%% long=%s short=%s flat=%s",
        horizon,
        threshold_pct,
        int((label == 1).sum()),
        int((label == -1).sum()),
        int((label == 0).sum()),
    )
    return out


def build_labels_triple_barrier(m5: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """标签与验收回测一致：SL=1.2×ATR, TP=2.0×ATR。"""
    key = ("tb", horizon)
    if key in _label_cache:
        return _label_cache[key]

    atr = _atr_series(m5).to_numpy()
    close = m5["close"].to_numpy()
    high = m5["high"].to_numpy()
    low = m5["low"].to_numpy()
    n = len(m5)
    label = np.zeros(n, dtype=int)
    entry_off = np.zeros(n, dtype=float)

    for i in range(n - horizon):
        if i > 0 and i % 10000 == 0:
            logger.info("triple_barrier h=%s progress %s/%s", horizon, i, n - horizon)
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        c = close[i]
        sl_l, tp_l = c - SL_ATR * a, c + TP_ATR * a
        sl_s, tp_s = c + SL_ATR * a, c - TP_ATR * a
        long_out = short_out = 0
        min_low, max_high = c, c
        for j in range(i + 1, i + 1 + horizon):
            h, l = high[j], low[j]
            min_low = min(min_low, l)
            max_high = max(max_high, h)
            if long_out == 0:
                if l <= sl_l and h >= tp_l:
                    long_out = -1
                elif l <= sl_l:
                    long_out = -1
                elif h >= tp_l:
                    long_out = 1
            if short_out == 0:
                if h >= sl_s and l <= tp_s:
                    short_out = -1
                elif h >= sl_s:
                    short_out = -1
                elif l <= tp_s:
                    short_out = 1
        if long_out == 1 and short_out != 1:
            label[i] = 1
            entry_off[i] = (min_low - c) / c
        elif short_out == 1 and long_out != 1:
            label[i] = -1
            entry_off[i] = (max_high - c) / c

    out = pd.DataFrame({"label": label, "entry_offset": entry_off}, index=m5.index)
    _label_cache[key] = out
    logger.info(
        "triple_barrier h=%s long=%s short=%s",
        horizon, int((label == 1).sum()), int((label == -1).sum()),
    )
    return out


def build_labels(m5: pd.DataFrame, p: LabelParams) -> pd.DataFrame:
    h = p.horizon
    out = pd.DataFrame(index=m5.index)
    future_low = m5["low"].rolling(h).min().shift(-h)
    future_high = m5["high"].rolling(h).max().shift(-h)
    close = m5["close"]
    low_drop = (future_low - close) / close * 100
    high_rise = (future_high - close) / close * 100
    label = np.zeros(len(m5), dtype=int)
    long_mask = (low_drop >= -p.long_max_dd_pct) & (high_rise >= p.long_min_gain_pct)
    short_mask = (high_rise <= p.short_max_rise_pct) & (low_drop <= -p.short_min_drop_pct)
    label[long_mask] = 1
    label[short_mask] = -1
    out["label"] = label
    out["entry_offset"] = np.where(
        label == 1, low_drop / 100, np.where(label == -1, high_rise / 100, 0)
    )
    out["expected_return"] = (future_high - close) / close * 100
    out["expected_return"] = out["expected_return"].where(
        label != -1, (close - future_low) / close * 100
    )
    return out.dropna()


def build_feature_matrix(m5: pd.DataFrame, seq_len: int = 60) -> FeatureMatrix | None:
    feats = compute_m5_features(m5)
    mtf = compute_mtf_trend_features(m5).reindex(feats.index).fillna(0.0)
    if len(feats) < seq_len + 500:
        return None

    values = feats[FEATURE_COLUMNS].values
    mtf_values = mtf[MTF_COLUMNS].values
    rows_X, fused_rows, bar_idx = [], [], []
    H_table = precompute_hourly_backgrounds(m5, feats.index[seq_len:])

    for i in range(seq_len, len(feats)):
        window = values[i - seq_len : i].astype(np.float32)
        rows_X.append(window)
        bar_idx.append(i)
        hi = i - seq_len
        fused_rows.append(build_fused_row(window, mtf_values[i], H_table[hi]))

    X_raw = np.stack(rows_X)
    fused = np.stack(fused_rows).astype(np.float32)
    bar_indices = np.array(bar_idx)
    feat_times = feats.index[seq_len:]
    H = H_table

    n = len(fused)
    oos_start = max(0, n - OOS_BARS_2W)
    val_size = min(max(int(n * VAL_FRAC), 1), VAL_BARS_3M)
    val_start = max(0, oos_start - val_size)
    train_end = val_start
    val_end = oos_start

    global _scaler_cache
    scaler = _scaler_cache or _load_scaler_cache()
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(fused[:train_end])
        _save_scaler_cache(scaler)
    _scaler_cache = scaler

    fused = scaler.transform(fused).astype(np.float32)
    logger.info(
        "Feature matrix mode=%s dim=%s train=%s val=%s oos=%s",
        FEATURE_MODE, fused.shape[1], train_end, val_end - train_end, n - oos_start,
    )

    return FeatureMatrix(
        m5=m5,
        X_raw=X_raw,
        bar_indices=bar_indices,
        feat_times=feat_times,
        H=H,
        train_end=train_end,
        val_end=val_end,
        oos_start=oos_start,
        fused=fused,
        scaler=scaler,
    )


def labels_for_params(m5: pd.DataFrame, feat_times: pd.DatetimeIndex, p: LabelParams) -> tuple[np.ndarray, np.ndarray]:
    if LABEL_MODE == "return_direction":
        lab_df = build_labels_return_direction(m5, p.horizon, p.return_threshold_pct)
    else:
        lab_df = build_labels_triple_barrier(m5, p.horizon)
    aligned = lab_df.reindex(feat_times)
    y_raw = aligned["label"].fillna(0).astype(int).to_numpy()
    entry = aligned["entry_offset"].fillna(0.0).to_numpy()
    return y_raw, entry


@dataclass
class DatasetBundle:
    m5: pd.DataFrame
    y_raw: np.ndarray
    entry: np.ndarray
    fused: np.ndarray
    bar_indices: np.ndarray
    train_end: int
    val_end: int
    oos_start: int
    feat_times: pd.DatetimeIndex
    scaler: StandardScaler


def build_dataset(m5: pd.DataFrame, label_params: LabelParams, seq_len: int = 60) -> DatasetBundle | None:
    fm = build_feature_matrix(m5, seq_len)
    if fm is None:
        return None
    y_raw, entry = labels_for_params(m5, fm.feat_times, label_params)
    return DatasetBundle(
        m5=fm.m5,
        y_raw=y_raw,
        entry=entry,
        fused=fm.fused,
        bar_indices=fm.bar_indices,
        train_end=fm.train_end,
        val_end=fm.val_end,
        oos_start=fm.oos_start,
        feat_times=fm.feat_times,
        scaler=fm.scaler,
    )


def predict_entry_batch(
    models: DirectionModels,
    reg_long: xgb.XGBRegressor,
    reg_short: xgb.XGBRegressor,
    X: np.ndarray,
) -> np.ndarray:
    y_pred = models.predict_direction(X)
    out = np.zeros(len(X))
    long_ix = y_pred == 2
    short_ix = y_pred == 0
    if long_ix.any():
        out[long_ix] = reg_long.predict(X[long_ix])
    if short_ix.any():
        out[short_ix] = reg_short.predict(X[short_ix])
    return out


def _extreme_rs(
    models: DirectionModels,
    fused,
    m5,
    bar_indices,
    times,
    ex_start,
    ex_end,
    horizon,
) -> tuple[list[float], float]:
    mask = (times >= ex_start) & (times <= ex_end)
    if not mask.any():
        return [], 0.0
    idxs = np.where(mask)[0]
    y_pred = models.predict_direction(fused[idxs])
    close = m5["close"].to_numpy()
    rs: list[float] = []
    for j, idx in enumerate(idxs):
        if y_pred[j] not in (0, 2):
            continue
        bi = int(bar_indices[idx])
        if bi + horizon >= len(close):
            continue
        c0 = close[bi]
        ret = (close[bi + horizon] - c0) / c0 * 100.0
        direction = 1 if y_pred[j] == 2 else -1
        rs.append(direction * ret / 0.15)
    eq = np.cumsum(rs) if rs else np.array([0.0])
    return rs, max_drawdown(eq)


def run_acceptance(
    models: DirectionModels,
    reg_long: xgb.XGBRegressor,
    reg_short: xgb.XGBRegressor,
    bundle: DatasetBundle,
    label_params: LabelParams,
    data_range: dict[str, str],
    thresholds: AcceptanceThresholds | None = None,
) -> AcceptanceReport:
    th = thresholds or AcceptanceThresholds()
    y = bundle.y_raw + 1
    tr, va_end, oos = slice(0, bundle.train_end), slice(bundle.train_end, bundle.val_end), slice(bundle.oos_start, None)

    fused_tr, fused_va, fused_oos = bundle.fused[tr], bundle.fused[va_end], bundle.fused[oos]
    y_tr, y_va, y_oos = y[tr], y[va_end], y[oos]
    entry_va, entry_oos = bundle.entry[va_end], bundle.entry[oos]
    idx_va, idx_oos = bundle.bar_indices[va_end], bundle.bar_indices[oos]

    y_pred_tr = models.predict_direction(fused_tr)
    y_pred_va = models.predict_direction(fused_va)
    entry_pred_va = predict_entry_batch(models, reg_long, reg_short, fused_va)

    dir_tr = directional_metrics(y_tr, y_pred_tr)
    dir_va = directional_metrics(y_va, y_pred_va)
    reg_va = regressor_metrics(y_pred_va, y_va, entry_pred_va, entry_va)
    bt_va = backtest_validation(y_pred_va, entry_pred_va, bundle.m5, idx_va, label_params.horizon)

    failures: list[str] = []
    metrics: dict[str, Any] = {
        "train": dir_tr,
        "validation": dir_va,
        "regressor": reg_va,
        "backtest_validation": bt_va,
    }

    if dir_va["precision"] < th.precision_relaxed:
        failures.append(f"val_precision={dir_va['precision']:.3f}<{th.precision_relaxed}")
    if dir_va["recall"] < th.recall:
        failures.append(f"val_recall={dir_va['recall']:.3f}<{th.recall}")
    if dir_va["f1"] < th.f1:
        failures.append(f"val_f1={dir_va['f1']:.3f}<{th.f1}")
    if abs(dir_va["prec_long"] - dir_va["prec_short"]) > th.long_short_prec_gap:
        failures.append("long_short_precision_imbalance")
    if reg_va["mae_pct"] > th.reg_mae_pct and th.require_regressor:
        failures.append(f"reg_mae={reg_va['mae_pct']:.3f}%")
    if reg_va["sign_accuracy"] < th.reg_sign_acc and th.require_regressor:
        failures.append(f"reg_sign={reg_va['sign_accuracy']:.3f}")
    if bt_va["avg_rr"] < th.avg_rr:
        failures.append(f"avg_rr={bt_va['avg_rr']:.3f}")
    if bt_va["expectancy"] <= 0:
        failures.append("expectancy<=0")

    prec_gap = abs(dir_tr["precision"] - dir_va["precision"])
    metrics["train_val_precision_gap"] = prec_gap
    if prec_gap > th.train_val_prec_gap:
        failures.append(f"overfit_gap={prec_gap:.3f}")

    proba_oos = models.predict_proba(fused_oos)
    y_pred_oos = predict_from_dual(
        models._cal_long(fused_oos),
        models._cal_short(fused_oos),
        threshold=models.prob_threshold,
        short_max=models.short_max,
    )
    entry_pred_oos = predict_entry_batch(models, reg_long, reg_short, fused_oos)
    bt_oos = backtest_validation(y_pred_oos, entry_pred_oos, bundle.m5, idx_oos, label_params.horizon)
    dir_mask = np.isin(y_pred_oos, [0, 2])
    oos_wr = float((y_pred_oos[dir_mask] == y_oos[dir_mask]).mean()) if dir_mask.any() else 0.0
    metrics["oos_recent_2w"] = {**bt_oos, "signal_win_rate": oos_wr}
    if oos_wr < th.oos_win_rate:
        failures.append(f"oos_win_rate={oos_wr:.3f}")
    if bt_oos.get("expectancy", -1) <= 0:
        failures.append("oos_expectancy<=0")

    ex_start, ex_end = find_extreme_month(bundle.m5)
    rs_ex, dd_ex = _extreme_rs(
        models, bundle.fused, bundle.m5, bundle.bar_indices,
        bundle.feat_times, ex_start, ex_end, label_params.horizon,
    )
    streak = consecutive_losses(rs_ex)
    metrics["extreme_month"] = {
        "start": str(ex_start.date()),
        "end": str(ex_end.date()),
        "max_consec_losses": streak,
        "max_drawdown": dd_ex,
        "n_trades": len(rs_ex),
    }
    if streak > th.extreme_max_consec_loss:
        failures.append(f"extreme_consec_loss={streak}")
    if dd_ex > th.extreme_max_drawdown:
        failures.append(f"extreme_dd={dd_ex:.3f}")

    return AcceptanceReport(
        passed=len(failures) == 0,
        metrics=metrics,
        failures=failures,
        label_params=label_params.as_dict(),
        data_range=data_range,
    )


@dataclass
class TrainResult:
    passed: bool
    report: AcceptanceReport
    label_params: LabelParams
    bundle: DatasetBundle
    models: DirectionModels
    reg_long: xgb.XGBRegressor
    reg_short: xgb.XGBRegressor
    scaler: StandardScaler
    prob_threshold: float = 0.60


def _fit_direction_models(
    fused_tr: np.ndarray,
    fused_va: np.ndarray,
    y_raw_tr: np.ndarray,
    y_raw_va: np.ndarray,
    xgb_depth: int,
    xgb_trees: int,
    long_weight: float,
    short_weight: float,
    prob_threshold: float = 0.60,
    subsample: float = 0.7,
    colsample_bytree: float = 0.7,
) -> DirectionModels:
    y_long_tr = (y_raw_tr == 1).astype(int)
    y_short_tr = (y_raw_tr == -1).astype(int)
    y_long_va = (y_raw_va == 1).astype(int)
    y_short_va = (y_raw_va == -1).astype(int)

    common = dict(
        max_depth=xgb_depth,
        n_estimators=xgb_trees,
        learning_rate=0.03,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=10.0,
        reg_alpha=2.0,
        min_child_weight=10,
        gamma=0.3,
        eval_metric="logloss",
        early_stopping_rounds=20,
    )
    clf_long = xgb.XGBClassifier(objective="binary:logistic", scale_pos_weight=long_weight, **common)
    clf_short = xgb.XGBClassifier(objective="binary:logistic", scale_pos_weight=short_weight, **common)
    clf_long.fit(fused_tr, y_long_tr, eval_set=[(fused_va, y_long_va)], verbose=False)
    clf_short.fit(fused_tr, y_short_tr, eval_set=[(fused_va, y_short_va)], verbose=False)

    return DirectionModels(
        clf_long=clf_long,
        clf_short=clf_short,
        prob_threshold=prob_threshold,
        short_max=0.35,
    )


def train_one(
    m5: pd.DataFrame,
    label_params: LabelParams,
    seq_len: int = 60,
    xgb_depth: int = 3,
    xgb_trees: int = 150,
    long_weight: float = 1.0,
    short_weight: float = 1.0,
    prob_threshold: float = 0.60,
    subsample: float = 0.7,
    colsample_bytree: float = 0.7,
) -> TrainResult | None:
    bundle = build_dataset(m5, label_params, seq_len)
    if bundle is None:
        return None

    y = bundle.y_raw + 1
    tr = slice(0, bundle.train_end)
    va = slice(bundle.train_end, bundle.val_end)

    fused_tr, fused_va = bundle.fused[tr], bundle.fused[va]
    y_tr, y_va = y[tr], y[va]
    y_raw_tr, y_raw_va = bundle.y_raw[tr], bundle.y_raw[va]
    pos = bundle.y_raw != 0

    n_long = max(int((y_raw_tr == 1).sum()), 1)
    n_short = max(int((y_raw_tr == -1).sum()), 1)
    n_flat = max(int((y_raw_tr == 0).sum()), 1)
    # scale_pos_weight 上限 25，避免正类权重爆炸导致全量信号
    lw = float(np.clip(n_flat / n_long, 1.0, 25.0)) * float(np.clip(long_weight, 0.5, 2.0))
    sw = float(np.clip(n_flat / n_short, 1.0, 25.0)) * float(np.clip(short_weight, 0.5, 2.0))

    models = _fit_direction_models(
        fused_tr, fused_va, y_raw_tr, y_raw_va, xgb_depth, xgb_trees, lw, sw,
        prob_threshold=prob_threshold, subsample=subsample, colsample_bytree=colsample_bytree,
    )

    reg_long = xgb.XGBRegressor(max_depth=2, n_estimators=50, learning_rate=0.05, reg_lambda=5.0)
    reg_short = xgb.XGBRegressor(max_depth=2, n_estimators=50, learning_rate=0.05, reg_lambda=5.0)
    reg_long.fit(fused_tr[:1], np.array([0.0]))
    reg_short.fit(fused_tr[:1], np.array([0.0]))

    data_range = {
        "m5_start": str(m5.index.min()),
        "m5_end": str(m5.index.max()),
        "train_end": str(bundle.feat_times[bundle.train_end - 1]) if bundle.train_end else "",
        "val_end": str(bundle.feat_times[bundle.val_end - 1]) if bundle.val_end else "",
        "oos_start": str(bundle.feat_times[bundle.oos_start]) if bundle.oos_start < len(bundle.feat_times) else "",
    }
    report = run_acceptance(models, reg_long, reg_short, bundle, label_params, data_range)
    n_val = fused_va.shape[0]
    n_sig = int(np.isin(models.predict_direction(fused_va), [0, 2]).sum())
    logger.info(
        "验收 val_prec=%.3f recall=%.3f f1=%.3f signals=%s/%s (%.1f%%) passed=%s",
        report.metrics["validation"]["precision"],
        report.metrics["validation"]["recall"],
        report.metrics["validation"]["f1"],
        n_sig,
        n_val,
        100.0 * n_sig / max(n_val, 1),
        report.passed,
    )
    return TrainResult(
        passed=report.passed,
        report=report,
        label_params=label_params,
        bundle=bundle,
        models=models,
        reg_long=reg_long,
        reg_short=reg_short,
        scaler=bundle.scaler,
        prob_threshold=prob_threshold,
    )


def save_model(result: TrainResult, symbol: str, reports_dir: Path) -> Path:
    out_dir = model_dir_for_symbol(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.models.clf_long.save_model(str(out_dir / "xgb_classifier_long.json"))
    result.models.clf_short.save_model(str(out_dir / "xgb_classifier_short.json"))
    result.reg_long.save_model(str(out_dir / "xgb_regressor_long.json"))
    result.reg_short.save_model(str(out_dir / "xgb_regressor_short.json"))
    result.reg_long.save_model(str(out_dir / "xgb_regressor.json"))
    joblib.dump(result.scaler, out_dir / "scaler.pkl")

    y_va = result.bundle.y_raw[result.bundle.train_end : result.bundle.val_end] + 1
    fused_va = result.bundle.fused[result.bundle.train_end : result.bundle.val_end]
    y_pred_va = result.models.predict_direction(fused_va)

    manifest = {
        "symbol": symbol,
        "kind": "production",
        "acceptance_passed": True,
        "classifier_mode": "dual_binary",
        "feature_mode": FEATURE_MODE,
        "fused_feature_dim": int(result.bundle.fused.shape[1]),
        "prob_threshold": result.prob_threshold,
        "acceptance_at": datetime.now().isoformat(timespec="seconds"),
        "feature_dim": len(FEATURE_COLUMNS),
        "seq_len": 60,
        "label_params": result.label_params.as_dict(),
        "data_range": result.report.data_range,
        "metrics": result.report.metrics,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    sym_reports = reports_dir / symbol / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_artifacts(result.report, result.models.clf_long, y_va, y_pred_va, sym_reports)
    (sym_reports / "MODEL_README.md").write_text(
        _model_readme(result), encoding="utf-8"
    )
    return out_dir


def save_rejected(result: TrainResult, symbol: str, reports_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rej = reports_dir / "rejected" / symbol / ts
    rej.mkdir(parents=True, exist_ok=True)
    result.models.clf_long.save_model(str(rej / "xgb_classifier_long.json"))
    result.models.clf_short.save_model(str(rej / "xgb_classifier_short.json"))
    result.reg_long.save_model(str(rej / "xgb_regressor_long.json"))
    result.reg_short.save_model(str(rej / "xgb_regressor_short.json"))
    joblib.dump(result.scaler, rej / "scaler.pkl")
    manifest = {
        "symbol": symbol,
        "kind": "rejected",
        "acceptance_passed": False,
        "failures": result.report.failures,
        "metrics": result.report.metrics,
        "label_params": result.label_params.as_dict(),
    }
    (rej / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (rej / "acceptance_report.json").write_text(
        json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rej


def _model_readme(result: TrainResult) -> str:
    lp = result.label_params
    m = result.report.metrics
    return f"""# ZhuLong Model — {result.report.data_range.get('m5_end', '')}

## Label thresholds
- mode: {LABEL_MODE}
- return threshold: ±{lp.return_threshold_pct}%
- horizon: {lp.horizon} M5 bars

## Data split
- Train / Val / OOS-2w: chronological; OOS = last {OOS_BARS_2W} M5 bars held out
- Val fraction: {VAL_FRAC:.0%} of samples before OOS block

## Validation metrics
- Precision: {m['validation']['precision']:.4f}
- Recall: {m['validation']['recall']:.4f}
- F1: {m['validation']['f1']:.4f}
- Reg MAE: {m['regressor']['mae_pct']:.4f}%
- Backtest RR: {m['backtest_validation']['avg_rr']:.4f}
"""


def plot_oos_equity(result: TrainResult, path: Path) -> None:
    oos = slice(result.bundle.oos_start, None)
    fused = result.bundle.fused[oos]
    idx = result.bundle.bar_indices[oos]
    y_pred = result.models.predict_direction(fused)
    close = result.bundle.m5["close"].to_numpy()
    h = result.label_params.horizon
    rs: list[float] = []
    for j, pred in enumerate(y_pred):
        if pred not in (0, 2):
            continue
        bi = int(idx[j])
        if bi + h >= len(close):
            continue
        ret = (close[bi + h] - close[bi]) / close[bi] * 100.0
        direction = 1 if pred == 2 else -1
        rs.append(direction * ret / 0.15)
    eq = np.cumsum(rs) if rs else np.array([0.0])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq)
    ax.set_title("OOS 2-week equity (R multiples)")
    ax.set_xlabel("trade #")
    ax.set_ylabel("cumulative R")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
