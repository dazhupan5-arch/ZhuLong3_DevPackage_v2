#!/usr/bin/env python3
"""FRED 宏观序列离线拉取 → data/fred_latest.json"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_data_dir() -> Path:
    env = os.environ.get("ZHULONG_DATA_DIR")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    p = Path(appdata) / "ZhuLong" / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_secret_file(name: str) -> str:
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    path = Path(appdata) / "ZhuLong" / "secrets" / name
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def load_config() -> dict:
    cfg_path = ROOT / "config.json"
    if not cfg_path.is_file():
        return {}
    with cfg_path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    macro = load_config().get("macro", {}) or {}
    fred_cfg = macro.get("fred", {}) or {}
    series = fred_cfg.get("series") or ["GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "T10YIE"]
    api_key = read_secret_file("fred_api_key.txt") or (fred_cfg.get("api_key") or os.environ.get("FRED_API_KEY") or "").strip()
    out_rel = fred_cfg.get("json_path") or "data/fred_latest.json"
    out_name = Path(out_rel).name
    out_path = resolve_data_dir() / out_name

    if not api_key:
        print("警告: 未配置 FRED API Key（config macro.fred.api_key 或 FRED_API_KEY），保留现有文件")
        return 0 if out_path.is_file() else 1

    try:
        from fredapi import Fred
    except ImportError:
        print("请安装: pip install fredapi pandas")
        return 1

    fred = Fred(api_key=api_key)
    payload: dict = {"updated_at": datetime.now(timezone.utc).isoformat()}
    for sid in series:
        try:
            s = fred.get_series(sid)
            s = s.dropna().tail(24)
            payload[sid] = [
                {"date": idx.strftime("%Y-%m-%d"), "value": float(val)}
                for idx, val in s.items()
            ]
            print(f"  {sid}: {len(payload[sid])} 点")
        except Exception as exc:
            print(f"  {sid} 失败: {exc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
