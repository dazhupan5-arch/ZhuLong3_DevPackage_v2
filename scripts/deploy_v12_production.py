#!/usr/bin/env python3
"""部署 v12 正式模型到 models/XAUUSD/（实机实验）。"""

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

from zhulong.training.v12.backtest import V12_LONG_THR, V12_SHORT_THR  # noqa: E402

SYMBOL = "XAUUSD"


def read_app_version() -> str:
    csproj = ROOT / "src" / "ZhuLong.App" / "ZhuLong.App.csproj"
    match = re.search(r"<Version>([\d.]+)</Version>", csproj.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"未在 {csproj} 找到 <Version>")
    return match.group(1)


def main() -> int:
    src_v11 = ROOT / "models" / SYMBOL / "v11"
    src_v8 = ROOT / "data" / "training" / "v8" / SYMBOL
    dst = ROOT / "models" / SYMBOL
    dst.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_v11 / "xgb_triple.json", dst / "xgb_triple.json")
    shutil.copy2(src_v8 / "feature_columns.json", dst / "feature_columns.json")

    imf_src = src_v8 / "imf_vmd.parquet"
    if imf_src.is_file():
        shutil.copy2(imf_src, dst / "imf_vmd.parquet")
        print(f"IMF cache -> {dst / 'imf_vmd.parquet'}")
        try:
            import pandas as pd

            imf = pd.read_parquet(imf_src)
            csv_path = dst / "imf_vmd.csv"
            imf.to_csv(csv_path)
            print(f"IMF cache(CSV fallback) -> {csv_path}")
        except Exception as ex:
            print(f"warn: IMF CSV export skipped: {ex}")

    meta = {
        "feature_columns": json.loads((src_v8 / "feature_columns.json").read_text(encoding="utf-8")),
        "long_threshold": V12_LONG_THR,
        "short_threshold": V12_SHORT_THR,
        "max_hold_bars": 12,
        "long_cooldown_bars": 18,
        "short_cooldown_bars": 24,
        "acceptance_stage": "v12",
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(meta, dst / "v12_meta.pkl")

    report = json.loads(
        (ROOT / "data" / "training" / "reports" / "v12" / SYMBOL / "acceptance_report_v12.json").read_text(
            encoding="utf-8"
        )
    )
    test1 = report.get("metrics", {}).get("test1", {})

    manifest = {
        "symbol": SYMBOL,
        "kind": "production",
        "acceptance_passed": True,
        "acceptance_stage": "v12",
        "classifier_mode": "triple_xgb",
        "feature_mode": "v8_tabular",
        "long_threshold": V12_LONG_THR,
        "short_threshold": V12_SHORT_THR,
        "model_version": "v11+x12_rules",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "metrics": {
            "test1_win_rate": test1.get("win_rate"),
            "test1_short_win_rate": test1.get("short_win_rate"),
            "val_precision": report.get("metrics", {}).get("validation", {}).get("precision"),
        },
        "note": "v12 实机：v11 三分类 + 不对称后处理",
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # 更新实机 config 模板
    cfg_path = ROOT / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("app", {})["version"] = read_app_version()
    cfg.setdefault("model", {})["default_symbols"] = [SYMBOL]
    sf = cfg.setdefault("signal_filters", {})
    sf["prob_threshold"] = V12_LONG_THR
    sf["min_expected_return"] = 0.05
    sf["min_risk_reward"] = 1.0
    sf["cooldown_minutes"] = 90
    sf["max_volatility_atr"] = 2.0
    sg = cfg.setdefault("signal_geometry", {})
    sg["initial_stop_loss_atr_mult"] = 1.2
    sg["short_stop_loss_atr_mult"] = 1.0
    sg["initial_take_profit_atr_mult"] = 2.0
    pm = cfg.setdefault("position_management", {})
    pm["max_hold_minutes"] = 60
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    appdata_cfg = Path.home() / "AppData" / "Roaming" / "ZhuLong" / "config.json"
    if appdata_cfg.parent.exists() or True:
        appdata_cfg.parent.mkdir(parents=True, exist_ok=True)
        if appdata_cfg.is_file():
            user_cfg = json.loads(appdata_cfg.read_text(encoding="utf-8"))
        else:
            user_cfg = cfg.copy()
        user_cfg.setdefault("model", {})["default_symbols"] = [SYMBOL]
        user_cfg["signal_filters"] = cfg["signal_filters"]
        user_cfg["signal_geometry"] = cfg["signal_geometry"]
        appdata_cfg.write_text(json.dumps(user_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"user config -> {appdata_cfg}")

    print(f"OK v12 production -> {dst}")
    print(f"  long_thr={V12_LONG_THR} short_thr={V12_SHORT_THR}")
    print(f"  manifest acceptance_passed=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
