#!/usr/bin/env python3
"""
训练 LightGBM 元标签模型（87维主模型 + 高质量正例定义）。

正例：模拟 R ≥ 1.5 且 持仓最大不利波动 < 5%

用法:
  py -3 scripts/precompute_simulated_trade_quality.py
  py -3 scripts/train_meta_label_lgb.py --enhanced
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.train_meta_label import run_meta_training

logger = logging.getLogger(__name__)

DEFAULT_QUALITY = "data/features/simulated_trade_quality.parquet"
DEFAULT_PRIMARY = "models/XAUUSD/lightgbm/lgb_triple_enhanced.pkl"
DEFAULT_OUT = "models/XAUUSD/meta_lgb"


def main() -> int:
    parser = argparse.ArgumentParser(description="LGB 元标签（质量正例）")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--enhanced", action="store_true", help="87维增强特征")
    parser.add_argument("--primary-model", default=DEFAULT_PRIMARY)
    parser.add_argument("--quality-file", default=DEFAULT_QUALITY)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--long-thr", type=float, default=0.55)
    parser.add_argument("--short-thr", type=float, default=0.45)
    parser.add_argument("--legacy-labels", action="store_true", help="使用旧版 R>0 标签")
    parser.add_argument("--quality-min-r", type=float, default=1.0, help="正例最低 R（默认1.0=达TP）")
    parser.add_argument("--quality-max-mae", type=float, default=0.02, help="正例最大不利波动占比")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pm = _ROOT / args.primary_model
    qf = _ROOT / args.quality_file
    out_dir = _ROOT / args.output_dir

    if not qf.is_file() and not args.legacy_labels:
        logger.warning("质量文件不存在 %s，将在线计算（较慢）", qf)

    stats = run_meta_training(
        _ROOT,
        symbol=args.symbol,
        primary_model=pm,
        primary_type="lgb",
        long_thr=args.long_thr,
        short_thr=args.short_thr,
        include_enhanced=args.enhanced or True,
        output_dir=out_dir,
        quality_file=qf if qf.is_file() else None,
        use_quality_labels=not args.legacy_labels,
        label_sl_mult=1.0,
        label_tp_mult=2.0,
        label_trailing=True,
        quality_min_r=args.quality_min_r,
        quality_max_mae_pct=args.quality_max_mae,
    )

    cfg = {
        "enabled": True,
        "threshold": 0.55,
        "model_path": str(Path(args.output_dir) / "meta_lgb_model.pkl"),
        "primary_model_path": args.primary_model,
        "primary_long_threshold": args.long_thr,
        "primary_short_threshold": args.short_thr,
        "enhanced": True,
        "use_quality_labels": not args.legacy_labels,
        "quality_file": args.quality_file,
        "val_auc": stats["auc"],
        "positive_rate": stats["positive_rate"],
    }
    (out_dir / "config_meta_lgb.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("LGB quality meta saved: %s", stats["model_path"])
    logger.info(
        "AUC=%.3f signals=%d positive_rate=%.1f%%",
        stats["auc"], stats["n_signals"], 100 * stats["positive_rate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
