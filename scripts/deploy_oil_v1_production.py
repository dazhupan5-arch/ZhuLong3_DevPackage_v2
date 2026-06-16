#!/usr/bin/env python3
"""部署 USOIL v1 正式模型到 models/USOIL/（实机 + 安装包）。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYMBOL = "USOIL"


def _read_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if text.startswith("\ufeff"):
        text = text[1:]
    return json.loads(text)


def _feature_columns(root: Path) -> list[str]:
    for p in (
        root / "models" / SYMBOL / "v1" / "feature_columns.json",
        root / "data" / "training" / "oil_v1" / SYMBOL / "feature_columns.json",
    ):
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("缺少 feature_columns.json（v1/ 或 data/training/oil_v1/USOIL/）")


def _thresholds(root: Path) -> tuple[float, float]:
    cfg_path = root / "models" / SYMBOL / "v1" / "config_oil_v1.json"
    if cfg_path.is_file():
        cfg = _read_json(cfg_path)
        return float(cfg.get("long_threshold", 0.82)), float(cfg.get("short_threshold", 0.84))

    rep_path = root / "data" / "training" / "reports" / "oil_v1" / SYMBOL / "acceptance_report_oil_v1.json"
    if rep_path.is_file():
        thr = _read_json(rep_path).get("metrics", {}).get("thresholds", {})
        return float(thr.get("long_thr", 0.82)), float(thr.get("short_thr", 0.84))

    return 0.82, 0.84


def deploy_to(root: Path, dst: Path) -> None:
    src_v1 = root / "models" / SYMBOL / "v1"
    model_src = src_v1 / "xgb_triple_oil.json"
    if not model_src.is_file():
        raise FileNotFoundError(f"缺少原油模型: {model_src}")

    dst.mkdir(parents=True, exist_ok=True)
    v1_dst = dst / "v1"
    v1_dst.mkdir(parents=True, exist_ok=True)

    cols = _feature_columns(root)
    long_thr, short_thr = _thresholds(root)

    model_dst = v1_dst / "xgb_triple_oil.json"
    if model_src.resolve() != model_dst.resolve():
        shutil.copy2(model_src, model_dst)
    elif not model_dst.is_file():
        raise FileNotFoundError(f"缺少原油模型: {model_src}")
    (v1_dst / "feature_columns.json").write_text(
        json.dumps(cols, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (dst / "feature_columns.json").write_text(
        json.dumps(cols, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    cfg_src = src_v1 / "config_oil_v1.json"
    cfg_dst = v1_dst / "config_oil_v1.json"
    if cfg_src.is_file() and cfg_src.resolve() != cfg_dst.resolve():
        shutil.copy2(cfg_src, cfg_dst)

    meta = {
        "feature_columns": cols,
        "long_threshold": long_thr,
        "short_threshold": short_thr,
        "max_hold_bars": 18,
        "cooldown_bars": 18,
        "acceptance_stage": "oil_v1",
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(meta, v1_dst / "oil_v1_meta.pkl")

    imf_src = root / "data" / "training" / "oil_v1" / SYMBOL / "imf_vmd.parquet"
    if imf_src.is_file():
        for imf_dst in (dst / "imf_vmd.parquet", v1_dst / "imf_vmd.parquet"):
            if imf_src.resolve() != imf_dst.resolve():
                shutil.copy2(imf_src, imf_dst)

    rep_path = root / "data" / "training" / "reports" / "oil_v1" / SYMBOL / "acceptance_report_oil_v1.json"
    test1: dict = {}
    val: dict = {}
    passed = True
    if rep_path.is_file():
        rep = _read_json(rep_path)
        passed = bool(rep.get("passed", True))
        test1 = rep.get("metrics", {}).get("test1", {})
        val = rep.get("metrics", {}).get("validation", {})

    manifest = {
        "symbol": SYMBOL,
        "kind": "production",
        "acceptance_passed": passed,
        "acceptance_stage": "oil_v1",
        "classifier_mode": "oil_v1",
        "feature_mode": "oil_v1_tabular",
        "long_threshold": long_thr,
        "short_threshold": short_thr,
        "model_version": "oil_v1",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "metrics": {
            "test1_win_rate": test1.get("win_rate"),
            "test1_short_win_rate": test1.get("short_win_rate"),
            "val_precision": val.get("precision"),
        },
        "note": "USOIL v1 三分类 + EIA 屏蔽 + 趋势过滤",
    }
    (dst / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        help="可选：复制 USOIL 到该目录（安装包 staging 用 models/ 根）",
    )
    args = parser.parse_args()

    repo_dst = ROOT / "models" / SYMBOL
    deploy_to(ROOT, repo_dst)
    print(f"OK oil v1 production -> {repo_dst}")

    if args.output_dir:
        stage_models = Path(args.output_dir)
        deploy_to(ROOT, stage_models / SYMBOL)
        print(f"OK staged -> {stage_models / SYMBOL}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
