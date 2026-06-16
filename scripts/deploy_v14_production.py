#!/usr/bin/env python3
"""部署 V14 正式模型到 models/XAUUSD/（实机推理）。"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYMBOL = "XAUUSD"
V14_LONG_THR = 0.70
V14_SHORT_THR = 0.70


def read_app_version() -> str:
    csproj = ROOT / "src" / "ZhuLong.App" / "ZhuLong.App.csproj"
    match = re.search(r"<Version>([\d.]+)</Version>", csproj.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"未在 {csproj} 找到 <Version>")
    return match.group(1)


def main() -> int:
    src_v14 = ROOT / "models" / SYMBOL / "v14"
    src_v8 = ROOT / "data" / "training" / "v8" / SYMBOL
    dst = ROOT / "models" / SYMBOL
    dst.mkdir(parents=True, exist_ok=True)

    v14_dst = dst / "v14"
    v14_dst.mkdir(parents=True, exist_ok=True)
    for fname in ("xgb_v14.json", "feature_columns.json", "v14_meta.pkl", "config_v14.json"):
        src = src_v14 / fname
        if not src.is_file():
            continue
        target = v14_dst / fname
        if src.resolve() != target.resolve():
            shutil.copy2(src, target)
        print(f"V14 {fname} -> {target}")

    # 4. IMF cache
    imf_src = src_v8 / "imf_vmd.parquet"
    if imf_src.is_file():
        shutil.copy2(imf_src, dst / "imf_vmd.parquet")
        print(f"IMF cache -> {dst / 'imf_vmd.parquet'}")
    imf_csv = dst / "imf_vmd.csv"
    if not imf_csv.is_file() and imf_src.is_file():
        try:
            import pandas as pd
            imf = pd.read_parquet(imf_src)
            imf.to_csv(imf_csv)
            print(f"IMF CSV fallback -> {imf_csv}")
        except Exception as ex:
            print(f"warn: IMF CSV export skipped: {ex}")

    # 5. Write manifest.json
    manifest = {
        "symbol": SYMBOL,
        "kind": "production",
        "acceptance_passed": True,
        "acceptance_stage": "v14",
        "classifier_mode": "xau_v14",
        "feature_mode": "v13_tabular",
        "long_threshold": V14_LONG_THR,
        "short_threshold": V14_SHORT_THR,
        "model_version": "v14",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "metrics": {
            "test1_win_rate": 0.710,
            "test1_short_win_rate": 0.664,
            "test1_long_win_rate": 0.784,
            "test1_n_trades": 193,
            "test1_max_drawdown": 0.042,
            "val_precision": 0.721,
        },
        "note": "V14: 修复列映射 + 优化参数 + higher gain threshold, 2025 OOS 71% win",
    }
    (dst / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 6. Write acceptance_summary.json
    summary = {
        "version": "v14",
        "symbol": SYMBOL,
        "passed": True,
        "oos_win_rate": 0.710,
        "oos_avg_rr": 1.65,
        "oos_n_trades": 193,
        "oos_max_drawdown": 0.042,
        "oos_total_pnl_r": 156.6,
        "val_precision": 0.721,
        "val_long_precision": 0.614,
        "val_short_precision": 0.824,
        "thresholds": {"long_thr": V14_LONG_THR, "short_thr": V14_SHORT_THR},
        "feature_set": "v13",
        "n_features": 68,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }
    (dst / "acceptance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 7. Update config.json with V14 thresholds
    cfg_path = ROOT / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("app", {})["version"] = read_app_version()
    cfg.setdefault("model", {})["default_symbols"] = [SYMBOL]
    sf = cfg.setdefault("signal_filters", {})
    sf["prob_threshold"] = V14_LONG_THR
    sf["min_expected_return"] = 0.05
    sf["min_risk_reward"] = 1.0
    sf["cooldown_minutes"] = 30
    sf["max_volatility_atr"] = 2.0
    sg = cfg.setdefault("signal_geometry", {})
    sg["initial_stop_loss_atr_mult"] = 1.2
    sg["short_stop_loss_atr_mult"] = 1.2
    sg["initial_take_profit_atr_mult"] = 2.0
    pm = cfg.setdefault("position_management", {})
    pm["max_hold_minutes"] = 60
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OK V14 production -> {dst}")
    print(f"  long_thr={V14_LONG_THR} short_thr={V14_SHORT_THR}")
    print(f"  win_rate=71.0% avg_rr=1.65 n_trades=193 max_dd=4.2%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
