#!/usr/bin/env python3
"""对比纯 V12 与 V12+结构过滤器（无需重训模型）。"""

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
import numpy as np
import pandas as pd
import xgboost as xgb

from zhulong.agent.structure_analyzer import FEATURE_NAMES, StructureAnalyzer
from zhulong.inference.v12 import V12Config
from zhulong.strategies.v12_structure_filter import V12WithStructureFilter
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
from zhulong.training.lgb.backtest import _atr_series
from zhulong.training.v10.backtest import backtest_both
from zhulong.training.v11.train import proba_to_directions
from zhulong.training.v13.triple import atr_ok

logger = logging.getLogger(__name__)


def directions_with_structure_filter(
    m5: pd.DataFrame,
    index: pd.DatetimeIndex,
    proba: np.ndarray,
    struct: np.ndarray,
    strategy: V12WithStructureFilter,
) -> np.ndarray:
    atr_s = _atr_series(m5)
    out = np.zeros(len(index), dtype=np.int8)
    for j, ts in enumerate(index):
        i = m5.index.get_loc(ts)
        if isinstance(i, slice):
            i = i.stop - 1 if i.stop else 0
        close = float(m5.iloc[i]["close"])
        atr = float(atr_s.iloc[i])
        if atr <= 0:
            continue
        out[j] = strategy.get_signal(struct[i], proba[j], atr, close, ts)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_xau_v12.json")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--quick", action="store_true", help="结构特征仅算末 5000 根（调试）")
    parser.add_argument("--jobs", type=int, default=0, help="结构特征并行 worker 数，0=自动")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    root = _ROOT
    cfg_path = root / args.config
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = V12Config.from_dict(raw)
    sf_raw = raw.get("structure_filter") or {}
    if not sf_raw:
        sf_raw = {
            "enabled": True,
            "long_prob_threshold": 0.65,
            "short_prob_threshold": 0.60,
            "min_support_strength": 0.4,
            "min_resistance_strength": 0.4,
            "max_support_dist": 0.5,
            "max_resistance_dist": 0.5,
            "require_breakout_confirm": False,
            "require_divergence": False,
            "allowed_hours": [9, 10, 11, 12, 13, 14, 15, 16],
        }

    m5 = load_vendor_csv(root / "data" / "training" / "lgb" / cfg.symbol / f"{cfg.symbol}_M5.csv")
    feat_cache = root / "data" / "training" / "v13" / cfg.symbol / "features.parquet"
    feats = pd.read_parquet(feat_cache) if feat_cache.is_file() else compute_features(m5, include_reversal=True)

    meta_path = root / cfg.meta_path
    if meta_path.is_file():
        meta = joblib.load(meta_path)
        cols = meta.get("feature_columns") or list(FEATURE_COLUMNS_LGB_V13)
    else:
        cols = list(FEATURE_COLUMNS_LGB_V13)

    model = xgb.XGBClassifier()
    model.load_model(str(root / cfg.model_path))

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    ix = feats.index[(feats.index >= start) & (feats.index <= end)]
    proba = model.predict_proba(feats.loc[ix, cols])

    sa_cfg = sf_raw.get("analyzer") or {"lookback": 200, "zigzag_atr_mult": 0.5}
    analyzer = StructureAnalyzer(sa_cfg)
    lookback = int(sa_cfg.get("lookback", 200))
    if args.quick:
        tail = m5.index[m5.index <= ix[-1]][-5000:]
        slice_ix = tail
    else:
        warm = m5.index[m5.index < start]
        warm_start = warm[-lookback] if len(warm) >= lookback else m5.index[0]
        slice_ix = m5.index[(m5.index >= warm_start) & (m5.index <= end)]
    struct_cache = root / "data" / "training" / "v13" / cfg.symbol / "structure_30d.parquet"
    if struct_cache.is_file():
        cached = pd.read_parquet(struct_cache)
        need = m5.loc[slice_ix].index
        missing = need.difference(cached.index)
        if len(missing) == 0:
            logger.info("loaded cached structure features: %s", struct_cache)
            struct_slice = cached.loc[need][list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        else:
            logger.info("结构缓存缺 %d 根，增量计算…", len(missing))
            struct_slice = analyzer.compute_all(
                m5.loc[slice_ix], progress_every=10000, n_jobs=args.jobs
            )
            df_new = pd.DataFrame(struct_slice, index=slice_ix, columns=list(FEATURE_NAMES))
            cached = pd.concat([cached, df_new]).sort_index()
            cached = cached[~cached.index.duplicated(keep="last")]
            struct_cache.parent.mkdir(parents=True, exist_ok=True)
            cached.to_parquet(struct_cache)
    else:
        logger.info("预计算结构特征 %d 根 (%s .. %s)…", len(slice_ix), slice_ix[0], slice_ix[-1])
        struct_slice = analyzer.compute_all(
            m5.loc[slice_ix], progress_every=10000, n_jobs=args.jobs
        )
        struct_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(struct_slice, index=slice_ix, columns=list(FEATURE_NAMES)).to_parquet(struct_cache)
    struct = np.zeros((len(m5), struct_slice.shape[1]), dtype=np.float32)
    locs = m5.index.get_indexer(slice_ix)
    struct[locs] = struct_slice

    baseline_dirs = proba_to_directions(proba, cfg.long_threshold, cfg.short_threshold)
    baseline_dirs = atr_ok(m5, ix, baseline_dirs)

    strategy = V12WithStructureFilter.from_dict(sf_raw)
    filtered_dirs = directions_with_structure_filter(m5, ix, proba, struct, strategy)
    filtered_dirs = atr_ok(m5, ix, filtered_dirs)

    cooldown = int(raw.get("long_cooldown_bars", 6))
    max_daily = int(cfg.max_daily_signals)
    max_hold = int(raw.get("max_hold_bars", 12))

    bt_base = backtest_both(
        m5, ix, baseline_dirs,
        max_hold=max_hold,
        cooldown_bars=cooldown,
        max_daily_signals=max_daily,
    )
    bt_struct = backtest_both(
        m5, ix, filtered_dirs,
        max_hold=max_hold,
        cooldown_bars=cooldown,
        max_daily_signals=max_daily,
    )

    report = {
        "config": str(cfg_path),
        "period": f"{args.start}..{args.end}",
        "structure_filter": sf_raw,
        "baseline_v12": bt_base,
        "v12_with_structure": bt_struct,
        "signal_counts": {
            "baseline_trades": int(np.count_nonzero(baseline_dirs)),
            "filtered_trades": int(np.count_nonzero(filtered_dirs)),
        },
    }

    out_dir = root / "data" / "training" / "reports" / "v12_structure" / cfg.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_compare.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
