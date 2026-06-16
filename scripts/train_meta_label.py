#!/usr/bin/env python3
"""元标签二分类：过滤主模型伪阳性信号。"""

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
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

from zhulong.training.lgb.backtest import _atr_series, simulate_trade
from zhulong.training.lgb.splits import split_indices
from zhulong.training.v11.train import proba_to_directions
from zhulong.training.v13.train_pipeline import load_training_bundle
from zhulong.training.v13.trade_sim import (
    META_QUALITY_MAX_MAE_PCT,
    META_QUALITY_MIN_R,
    is_quality_positive,
    simulate_trade_trailing,
)
from zhulong.training.v13.triple import atr_ok

logger = logging.getLogger(__name__)

META_EXTRA = ["proba_flat", "proba_long", "proba_short", "signal_conf", "atr_pct", "adx_norm"]


def _sim_quality_inline(
    m5: pd.DataFrame,
    t: pd.Timestamp,
    direction: int,
    *,
    sl_mult: float,
    tp_mult: float,
    max_hold: int,
    trailing: bool,
) -> tuple[float, float]:
    atr = _atr_series(m5)
    idx = m5.index.get_loc(t)
    if isinstance(idx, slice):
        return 0.0, 1.0
    a = float(atr.iloc[idx])
    entry = float(m5.loc[t, "close"])
    end = min(idx + 1 + max_hold, len(m5))
    hs = m5["high"].iloc[idx + 1 : end].to_numpy()
    ls = m5["low"].iloc[idx + 1 : end].to_numpy()
    cs = m5["close"].iloc[idx + 1 : end].to_numpy()
    sim = simulate_trade_trailing(
        int(direction), entry, a, hs, ls, cs,
        max_bars=max_hold, sl_mult=sl_mult, tp_mult=tp_mult, trailing=trailing,
    )
    return sim.r_multiple, sim.mae_pct


def _lookup_quality(
    quality_df: pd.DataFrame,
    t: pd.Timestamp,
    direction: int,
) -> tuple[float, float] | None:
    if t not in quality_df.index:
        return None
    row = quality_df.loc[t]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    r_col = "sim_r_long" if direction > 0 else "sim_r_short"
    mae_col = "sim_mae_long" if direction > 0 else "sim_mae_short"
    if r_col not in row.index or pd.isna(row[r_col]):
        return None
    return float(row[r_col]), float(row[mae_col])


def _build_quality_meta_labels(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
    quality_df: pd.DataFrame | None,
    *,
    sl_mult: float = 1.0,
    tp_mult: float = 2.0,
    max_hold: int = 12,
    trailing: bool = True,
    min_r: float = META_QUALITY_MIN_R,
    max_mae_pct: float = META_QUALITY_MAX_MAE_PCT,
) -> np.ndarray:
    labels = np.zeros(len(times), dtype=np.int8)
    sig_ix = np.where(directions != 0)[0]
    n_sig = len(sig_ix)
    logger.info("building quality meta labels for %d signals...", n_sig)
    for j, i in enumerate(sig_ix):
        if j > 0 and j % 5000 == 0:
            logger.info("  quality labels %d/%d", j, n_sig)
        t, d = times[i], int(directions[i])
        if t not in m5.index:
            continue
        pair = _lookup_quality(quality_df, t, d) if quality_df is not None else None
        if pair is None:
            r, mae = _sim_quality_inline(
                m5, t, d, sl_mult=sl_mult, tp_mult=tp_mult, max_hold=max_hold, trailing=trailing,
            )
        else:
            r, mae = pair
        labels[i] = 1 if is_quality_positive(r, mae, min_r=min_r, max_mae_pct=max_mae_pct) else 0
    return labels


def _build_meta_labels(
    m5: pd.DataFrame,
    times: pd.DatetimeIndex,
    directions: np.ndarray,
    sl_mult: float = 1.2,
    tp_mult: float = 2.0,
    max_hold: int = 12,
) -> np.ndarray:
    atr = _atr_series(m5)
    labels = np.zeros(len(times), dtype=np.int8)
    for i, (t, d) in enumerate(zip(times, directions)):
        if d == 0 or t not in m5.index:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            continue
        a = float(atr.iloc[idx])
        entry = float(m5.loc[t, "close"])
        if a <= 0:
            continue
        end = min(idx + 1 + max_hold, len(m5))
        r = simulate_trade(
            int(d), entry, a,
            m5["high"].iloc[idx + 1 : end].to_numpy(),
            m5["low"].iloc[idx + 1 : end].to_numpy(),
            m5["close"].iloc[idx + 1 : end].to_numpy(),
            max_bars=max_hold,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
        )
        labels[i] = 1 if r > 0 else 0
    return labels


def _load_primary_model(path: Path, model_type: str):
    if model_type == "lgb" or path.suffix == ".pkl":
        if path.suffix == ".pkl" and path.is_file():
            return joblib.load(path)
        for name in ("lgb_triple_enhanced.pkl", "lgb_triple_v3.pkl"):
            candidate = path.parent / name
            if candidate.is_file():
                return joblib.load(candidate)
    m = xgb.XGBClassifier()
    m.load_model(str(path))
    return m


def run_meta_training(
    root: Path,
    symbol: str = "XAUUSD",
    primary_model: Path | None = None,
    primary_type: str = "xgb",
    long_thr: float | None = None,
    short_thr: float | None = None,
    include_enhanced: bool = False,
    output_dir: Path | None = None,
    quality_file: Path | None = None,
    use_quality_labels: bool = False,
    label_sl_mult: float = 1.0,
    label_tp_mult: float = 2.0,
    label_trailing: bool = True,
    quality_min_r: float = META_QUALITY_MIN_R,
    quality_max_mae_pct: float = META_QUALITY_MAX_MAE_PCT,
) -> dict:
    bundle = load_training_bundle(root, symbol, include_enhanced=include_enhanced)
    m5, aligned, cols, va_ix = bundle["m5"], bundle["aligned"], bundle["cols"], bundle["va_ix"]
    tr_ix = split_indices(aligned.index).train.intersection(aligned.index)
    fit_ix = tr_ix.union(va_ix)

    pm = primary_model or root / "models" / symbol / "triple_barrier" / "xgb_triple_v3.json"
    if long_thr is None or short_thr is None:
        cfg_path = pm.parent / "config_v13.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            long_thr = long_thr if long_thr is not None else cfg.get("long_threshold", 0.5)
            short_thr = short_thr if short_thr is not None else cfg.get("short_threshold", 0.5)
        else:
            long_thr = long_thr or 0.5
            short_thr = short_thr or 0.5

    model = _load_primary_model(pm, primary_type)

    logger.info("primary predict on %d bars...", len(fit_ix))
    X = aligned.loc[fit_ix, cols]
    chunks: list[np.ndarray] = []
    step = 80_000
    for start in range(0, len(X), step):
        end = min(start + step, len(X))
        chunks.append(model.predict_proba(X.iloc[start:end]))
        logger.info("  predict %d/%d", end, len(X))
    proba = np.vstack(chunks) if len(chunks) > 1 else chunks[0]
    logger.info("generating directions thr=%.2f/%.2f", long_thr, short_thr)
    dirs = atr_ok(m5, fit_ix, proba_to_directions(proba, long_thr, short_thr))
    quality_df = None
    if quality_file and quality_file.is_file():
        quality_df = pd.read_parquet(quality_file)
        quality_df = quality_df.reindex(fit_ix, method="ffill")
        logger.info("quality features loaded: %d rows (ffill aligned)", len(quality_df))
    if use_quality_labels:
        meta_y = _build_quality_meta_labels(
            m5, fit_ix, dirs, quality_df,
            sl_mult=label_sl_mult, tp_mult=label_tp_mult,
            max_hold=12, trailing=label_trailing,
            min_r=quality_min_r, max_mae_pct=quality_max_mae_pct,
        )
    else:
        meta_y = _build_meta_labels(m5, fit_ix, dirs)

    sig_mask = dirs != 0
    if sig_mask.sum() < 50:
        raise RuntimeError(f"元标签样本不足: {int(sig_mask.sum())} 笔信号")

    meta_df = aligned.loc[fit_ix, cols].copy()
    meta_df["proba_flat"] = proba[:, 0]
    meta_df["proba_long"] = proba[:, 1]
    meta_df["proba_short"] = proba[:, 2]
    meta_df["signal_conf"] = np.max(proba[:, 1:3], axis=1)
    atr = _atr_series(m5)
    meta_df["atr_pct"] = (atr.reindex(fit_ix) / m5["close"].reindex(fit_ix)).fillna(0).values
    if "adx_norm" in meta_df.columns:
        pass
    else:
        meta_df["adx_norm"] = 0.0

    meta_cols = cols + [c for c in META_EXTRA if c not in cols]
    X_meta = meta_df.loc[sig_mask, meta_cols]
    y_meta = meta_y[sig_mask]

    split_pt = int(len(X_meta) * 0.8)
    X_tr, X_va = X_meta.iloc[:split_pt], X_meta.iloc[split_pt:]
    y_tr, y_va = y_meta[:split_pt], y_meta[split_pt:]

    meta_model = lgb.LGBMClassifier(
        objective="binary",
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        class_weight="balanced",
        verbose=-1,
    )
    meta_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    y_pred = meta_model.predict(X_va)
    rep = classification_report(y_va, y_pred, zero_division=0)
    auc = float(roc_auc_score(y_va, meta_model.predict_proba(X_va)[:, 1])) if len(np.unique(y_va)) > 1 else 0.0
    logger.info("meta val:\n%s auc=%.3f", rep, auc)

    out_dir = output_dir or (root / "models" / symbol / "meta_label")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = "meta_lgb_model.pkl" if "meta_lgb" in out_dir.as_posix() else "meta_label.pkl"
    joblib.dump(meta_model, out_dir / model_name)
    joblib.dump(
        {
            "feature_columns": meta_cols,
            "primary_model": str(pm),
            "primary_type": primary_type,
            "enhanced": include_enhanced,
            "long_thr": long_thr,
            "short_thr": short_thr,
            "meta_threshold": 0.6,
            "val_auc": auc,
            "n_signals": int(sig_mask.sum()),
            "positive_rate": float(y_meta.mean()),
            "use_quality_labels": use_quality_labels,
            "quality_file": str(quality_file) if quality_file else None,
        },
        out_dir / "meta_config.pkl",
    )
    (out_dir / "meta_feature_columns.json").write_text(json.dumps(meta_cols, indent=2), encoding="utf-8")
    (out_dir / "meta_features.json").write_text(json.dumps(meta_cols, indent=2), encoding="utf-8")

    return {
        "auc": auc,
        "n_signals": int(sig_mask.sum()),
        "positive_rate": float(y_meta.mean()),
        "report": rep,
        "output_dir": str(out_dir),
        "model_path": str(out_dir / model_name),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--primary-model", default="")
    parser.add_argument("--primary-type", choices=["xgb", "lgb"], default="xgb")
    parser.add_argument("--enhanced", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pm = Path(args.primary_model) if args.primary_model else None
    stats = run_meta_training(_ROOT, args.symbol, pm, args.primary_type, include_enhanced=args.enhanced)
    logger.info("meta done: %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
