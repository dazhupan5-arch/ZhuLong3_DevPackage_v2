#!/usr/bin/env python3
"""方案 A 离线回测：config_xau_v12 + v3 模型 + 移动止损/分批止盈。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import pandas as pd
import xgboost as xgb

from zhulong.inference.v12 import V12Config, load_v12_config
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
from zhulong.training.v10.backtest import backtest_both
from zhulong.training.v11.train import V11_COOLDOWN, V11_MAX_HOLD, proba_to_directions
from zhulong.training.v12.backtest import apply_short_trend_filter
from zhulong.training.v13.plan_a_backtest import PlanAPositionConfig, backtest_plan_a
from zhulong.training.v13.triple import TEST_END, TEST_START, postprocess_directions


def load_config(path: Path) -> V12Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return V12Config.from_dict(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_xau_v12.json")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--baseline", action="store_true", help="仅固定 SL/TP，无移动止损")
    args = parser.parse_args()

    root = _ROOT
    cfg_path = root / args.config
    cfg = load_config(cfg_path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    pm = raw.get("position_management") or {}
    trail = pm.get("trailing_stop") or {}
    partial = pm.get("partial_profit") or {}

    m5 = load_vendor_csv(root / "data" / "training" / "lgb" / cfg.symbol / f"{cfg.symbol}_M5.csv")
    feat_cache = root / "data" / "training" / "v13" / cfg.symbol / "features.parquet"
    if feat_cache.is_file():
        feats = pd.read_parquet(feat_cache)
    else:
        feats = compute_features(m5, include_reversal=True)

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    ix = feats.index[(feats.index >= start) & (feats.index <= end)]

    meta_path = root / cfg.meta_path if cfg.meta_path else root / "models" / cfg.symbol / "triple_barrier" / "params_v13_triple.pkl"
    if meta_path.is_file():
        meta = joblib.load(meta_path)
        cols = meta.get("feature_columns") or list(FEATURE_COLUMNS_LGB_V13)
    else:
        cols = list(FEATURE_COLUMNS_LGB_V13)

    model = xgb.XGBClassifier()
    model.load_model(str(root / cfg.model_path))

    proba = model.predict_proba(feats.loc[ix, cols])
    from zhulong.training.v13.triple import atr_ok

    dirs = proba_to_directions(proba, cfg.long_threshold, cfg.short_threshold)
    dirs = atr_ok(m5, ix, dirs)
    if raw.get("use_h1_trend_filter", raw.get("feature_set") == "v13"):
        from zhulong.training.v13.triple import apply_trend_filter_v3
        dirs = apply_trend_filter_v3(m5, ix, dirs)
    if raw.get("trend_filter_short") and not raw.get("use_h1_trend_filter"):
        dirs = apply_short_trend_filter(m5, feats.loc[ix], ix, dirs)

    cooldown = int(raw.get("long_cooldown_bars", 6))
    max_daily = int(cfg.max_daily_signals)

    plan_cfg = PlanAPositionConfig(
        sl_mult=float(cfg.long_sl_atr),
        tp_mult=float(cfg.tp_atr),
        max_hold=int(raw.get("max_hold_bars", V11_MAX_HOLD)),
        trailing_enabled=bool(trail.get("enabled", True)) and not args.baseline,
        trailing_activation_pct=float(trail.get("activation_pct", 0.15)),
        trailing_step_pct=float(trail.get("step_pct", 0.10)),
        trailing_tighten=float(trail.get("tighten_factor", 0.8)),
        partial_enabled=bool(partial.get("enabled", True)) and not args.baseline,
        partial_target1_pct=float(partial.get("target1_pct", 0.25)),
        partial_ratio1=float(partial.get("ratio1", 0.5)),
        partial_target2_pct=float(partial.get("target2_pct", 0.40)),
        partial_ratio2=float(partial.get("ratio2", 0.5)),
        profit_drawdown_ratio=float(pm.get("profit_drawdown_ratio", 0.4)),
    )

    bt_plan = backtest_plan_a(
        m5, ix, dirs, plan_cfg,
        cooldown_bars=cooldown,
        max_daily_signals=max_daily,
        min_atr_pct=cfg.min_atr_pct,
    )
    bt_fixed = backtest_both(
        m5, ix, dirs,
        max_hold=plan_cfg.max_hold,
        cooldown_bars=cooldown,
        max_daily_signals=max_daily,
    )

    report = {
        "config": str(cfg_path),
        "period": f"{args.start}..{args.end}",
        "thresholds": {"long": cfg.long_threshold, "short": cfg.short_threshold},
        "cooldown_bars": cooldown,
        "plan_a_managed": bt_plan,
        "fixed_sl_tp": bt_fixed,
    }
    out_dir = root / "data" / "training" / "reports" / "plan_a" / cfg.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_plan_a_2025.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== 方案 A 回测 ===")
    print(f"区间: {args.start} ~ {args.end}")
    print(f"阈值: long={cfg.long_threshold} short={cfg.short_threshold} cooldown={cooldown}bars")
    print("\n--- 固定 SL/TP（对照）---")
    for k, v in bt_fixed.items():
        print(f"  {k}: {v}")
    print("\n--- 移动止损 + 分批止盈 ---")
    for k, v in bt_plan.items():
        print(f"  {k}: {v}")
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
