"""V13 三分类回测运行器（供 backtest_triple / scan_meta_threshold 复用）。"""



from __future__ import annotations



import json

from pathlib import Path

from typing import Any



import joblib

import lightgbm as lgb

import pandas as pd

import xgboost as xgb



from zhulong.training.lgb.data_io import load_vendor_csv

from zhulong.training.lgb.splits import split_indices

from zhulong.training.v10.backtest import apply_adx_filter, backtest_both

from zhulong.training.v11.train import V11_COOLDOWN, V11_MAX_HOLD

from zhulong.training.v13.meta_filter import apply_meta_filter, build_meta_frame

from zhulong.training.v13.triple import TEST_END, TEST_START, atr_ok, postprocess_directions

from zhulong.training.v11.train import proba_to_directions





def load_primary_model(model_path: Path) -> tuple[Any, str]:

    if model_path.suffix in (".pkl", ".txt") or "lightgbm" in model_path.as_posix():

        pkl = model_path.with_suffix(".pkl") if model_path.suffix == ".txt" else model_path

        if pkl.is_file():

            return joblib.load(pkl), "lgb"

        if model_path.suffix == ".txt" and model_path.is_file():

            booster = lgb.Booster(model_file=str(model_path))

            wrapper = lgb.LGBMClassifier()

            wrapper._Booster = booster

            wrapper._n_classes = 3

            wrapper._classes = [0, 1, 2]

            return wrapper, "lgb"

    model = xgb.XGBClassifier()

    model.load_model(str(model_path))

    return model, "xgb"





def resolve_model_config(

    root: Path,

    symbol: str,

    model_path: Path | None,

    threshold: float | None,

    long_thr: float | None = None,

    short_thr: float | None = None,

) -> dict[str, Any]:

    model_dir = root / "models" / symbol / "triple_barrier"

    mp = model_path or (model_dir / "xgb_triple_v3.json")

    parent = mp.parent



    lt = long_thr if long_thr is not None else (0.55 if threshold is None else threshold)

    st = short_thr if short_thr is not None else (0.45 if threshold is None else threshold)

    cols: list[str] = []



    fc = parent / "feature_columns.json"

    if fc.is_file():

        cols = json.loads(fc.read_text(encoding="utf-8"))

    elif (parent / "params_v13.pkl").is_file():

        meta = joblib.load(parent / "params_v13.pkl")

        cols = meta["feature_columns"]

        if long_thr is None and short_thr is None and threshold is None:

            lt = meta.get("long_threshold", lt)

            st = meta.get("short_threshold", st)

    elif (model_dir / "params_v13_triple.pkl").is_file():

        meta = joblib.load(model_dir / "params_v13_triple.pkl")

        cols = meta["feature_columns"]

        if long_thr is None and short_thr is None and threshold is None:

            lt = meta.get("long_threshold", lt)

            st = meta.get("short_threshold", st)



    cfg_path = parent / "config_v13.json"

    if cfg_path.is_file() and long_thr is None and short_thr is None and threshold is None:

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        lt = cfg.get("long_threshold", lt)

        st = cfg.get("short_threshold", st)



    quality = "quality" in mp.as_posix() or len(cols) > 90
    enhanced = quality or "enhanced" in mp.as_posix() or len(cols) > 70
    feat_sub = "v13_quality" if quality else ("v13_enhanced" if enhanced else "v13")
    feat_path = root / "data" / "training" / feat_sub / symbol / "features.parquet"
    if not feat_path.is_file():
        feat_path = root / "data" / "training" / "v13" / symbol / "features.parquet"



    return {

        "model_path": mp,

        "feature_columns": cols,

        "long_threshold": lt,

        "short_threshold": st,

        "enhanced": enhanced,

        "feat_path": feat_path,

    }





def run_v13_backtest(

    root: Path,

    *,

    symbol: str = "XAUUSD",

    split: str = "test2025",

    model_path: str | Path | None = None,

    threshold: float | None = None,

    long_thr: float | None = None,

    short_thr: float | None = None,

    enhanced: bool | None = None,

    meta_model_path: str | Path | None = None,

    meta_threshold: float = 0.6,

    use_meta: bool = False,

    sl_mult: float = 1.0,

    tp_mult: float = 2.0,

    max_hold: int = V11_MAX_HOLD,

    cooldown_bars: int = V11_COOLDOWN,

    max_daily_signals: int = 8,

    trailing: bool = False,

    adx_filter: bool = False,

    adx_min: float = 25.0,

) -> dict[str, Any]:

    mp = Path(model_path) if model_path else None

    if mp and not mp.is_absolute():

        mp = root / mp



    cfg = resolve_model_config(root, symbol, mp, threshold, long_thr, short_thr)

    mp = cfg["model_path"]

    cols = cfg["feature_columns"]

    long_thr = cfg["long_threshold"]

    short_thr = cfg["short_threshold"]

    use_enhanced = enhanced if enhanced is not None else cfg["enhanced"]



    m5 = load_vendor_csv(root / "data" / "training" / "lgb" / symbol / f"{symbol}_M5.csv")

    feats = pd.read_parquet(cfg["feat_path"])

    splits = split_indices(feats.index)

    if split == "test2025":

        ix = feats.index[(feats.index >= TEST_START) & (feats.index <= TEST_END)]

    else:

        ix = getattr(splits, split).intersection(feats.index)



    model, mtype = load_primary_model(mp)

    proba = model.predict_proba(feats.loc[ix, cols])

    dirs = postprocess_directions(m5, ix, proba, long_thr, short_thr)

    if adx_filter:

        dirs = apply_adx_filter(m5, ix, dirs, adx_min=adx_min)



    bt_kw = dict(

        max_hold=max_hold,

        cooldown_bars=cooldown_bars,

        max_daily_signals=max_daily_signals,

        sl_mult=sl_mult,

        tp_mult=tp_mult,

        trailing=trailing,

        adx_min=None,

    )

    bt_base = backtest_both(m5, ix, dirs, **bt_kw)



    out: dict[str, Any] = {

        "model": str(mp),

        "model_type": mtype,

        "split": split,

        "enhanced_features": use_enhanced,

        "long_thr": long_thr,

        "short_thr": short_thr,

        "sl_mult": sl_mult,

        "tp_mult": tp_mult,

        "max_hold": max_hold,

        "trailing": trailing,

        "adx_filter": adx_filter,

        "adx_min": adx_min if adx_filter else None,

        "backtest_baseline": bt_base,

    }



    if use_meta and meta_model_path:

        mmp = Path(meta_model_path)

        if not mmp.is_absolute():

            mmp = root / mmp

        meta_model = joblib.load(mmp)

        meta_cfg_path = mmp.parent / "meta_config.pkl"

        meta_cols = json.loads((mmp.parent / "meta_feature_columns.json").read_text(encoding="utf-8"))

        if meta_cfg_path.is_file():

            pkl_cfg = joblib.load(meta_cfg_path)

            meta_cols = pkl_cfg.get("feature_columns", meta_cols)



        meta_frame = build_meta_frame(feats, ix, proba, m5, cols)

        dirs_f, meta_stats = apply_meta_filter(dirs, meta_model, meta_cols, meta_frame, meta_threshold)

        bt_meta = backtest_both(m5, ix, dirs_f, **bt_kw)

        out["meta_filter"] = meta_stats

        out["backtest_meta"] = bt_meta

        out["comparison"] = {

            "win_rate_before": bt_base.get("win_rate"),

            "win_rate_after": bt_meta.get("win_rate"),

            "n_trades_before": bt_base.get("n_trades"),

            "n_trades_after": bt_meta.get("n_trades"),

            "max_drawdown_before": bt_base.get("max_drawdown"),

            "max_drawdown_after": bt_meta.get("max_drawdown"),

            "avg_rr_before": bt_base.get("avg_rr"),

            "avg_rr_after": bt_meta.get("avg_rr"),

            "total_pnl_r_before": bt_base.get("total_pnl_r"),

            "total_pnl_r_after": bt_meta.get("total_pnl_r"),

        }



    return out


