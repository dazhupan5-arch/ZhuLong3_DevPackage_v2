#!/usr/bin/env python3
"""KN1 实机状态快检：配置漂移 + validate + tick 认知链。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

INSTALL = Path(r"C:\Program Files\ZhuLong")
APPDATA = Path.home() / "AppData" / "Roaming" / "ZhuLong" / "config_agent.json"
INST_CFG = INSTALL / "config" / "config_agent.json"
CLI = INSTALL / "ZhuLong.PythonEngine" / "inference_cli.py"
CSV = Path(r"C:\Users\xiaomi\Desktop\XAUUSD5.csv")
META = INSTALL / "models" / "knowledge_net.meta.json"


def run_cli(req: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fin:
        json.dump(req, fin, ensure_ascii=False)
        req_path = fin.name
    out_path = req_path + ".out"
    subprocess.run(
        [sys.executable, str(CLI), "--input", req_path, "--output", out_path],
        cwd=str(INSTALL),
        timeout=180,
        check=False,
    )
    return json.loads(Path(out_path).read_text(encoding="utf-8-sig")) if Path(out_path).is_file() else {}


def main() -> int:
    app = json.loads(APPDATA.read_text(encoding="utf-8-sig"))
    inst = json.loads(INST_CFG.read_text(encoding="utf-8-sig"))
    meta = json.loads(META.read_text(encoding="utf-8"))

    print("=== KN1 LIVE STATUS ===")
    print(f"ZhuLong config (effective): {APPDATA}")
    print(f"Model meta: input_dim={meta.get('input_dim')} agreement={meta.get('v14_agreement', 0):.2%}")

    drift = []
    if app.get("kn2", {}).get("enabled") != inst.get("kn2", {}).get("enabled"):
        drift.append(f"kn2.enabled AppData={app.get('kn2', {}).get('enabled')} Install={inst.get('kn2', {}).get('enabled')}")
    if app.get("knowledge_net", {}).get("input_dim") != meta.get("input_dim"):
        drift.append(
            f"input_dim AppData={app.get('knowledge_net', {}).get('input_dim')} model={meta.get('input_dim')}"
        )
    print("Config drift:", drift or "none")

    for label, cfg in [("Install", str(INST_CFG)), ("AppData", str(APPDATA))]:
        v = run_cli({"cmd": "agent_validate", "root": str(INSTALL), "config_path": cfg})
        print(
            f"validate[{label}]: ok={v.get('ok')} kn2_enabled={v.get('kn2_enabled')} "
            f"knowledge_ready={v.get('knowledge_ready')} action={v.get('action')}"
        )

    df = pd.read_csv(CSV, header=None, names=["date", "time", "open", "high", "low", "close", "volume"])
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), utc=True)
    df = df.sort_values("datetime").tail(300)
    bars = [
        [int(r.datetime.timestamp()), float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume or 0)]
        for r in df.itertuples()
    ]
    tick = run_cli(
        {
            "cmd": "agent_tick",
            "root": str(INSTALL),
            "config_path": str(APPDATA),
            "symbols": ["XAUUSD"],
            "primary_symbol": "XAUUSD",
            "m5_includes_forming": False,
            "m5_bars_by_symbol": {"XAUUSD": bars},
        }
    )
    r = (tick.get("results") or [{}])[-1]
    cog = r.get("cognition") or {}
    print(
        f"tick: ok={tick.get('ok')} knowledge_ready={r.get('knowledge_ready')} "
        f"action={r.get('action')} filter={r.get('filter_reason')}"
    )
    print(f"  cognition_dir={r.get('cognition_direction')} conf={r.get('cognition_confidence')}")
    print(f"  rl_raw={r.get('rl_raw_action')} regime={cog.get('regime')} should_trade={cog.get('should_trade')}")
    chain = str(cog.get("reasoning_chain") or "")[:120]
    if chain:
        print(f"  reasoning: {chain}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
