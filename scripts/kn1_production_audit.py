#!/usr/bin/env python3
"""KN1 生产环境实机审计：inference_cli agent_validate + 真实 M5 agent_tick。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INSTALL = Path(r"C:\Program Files\ZhuLong")
APPDATA_CFG = Path.home() / "AppData" / "Roaming" / "ZhuLong" / "config_agent.json"
PYTHON = Path(sys.executable)
CLI = INSTALL / "ZhuLong.PythonEngine" / "inference_cli.py"
CSV = Path(r"C:\Users\xiaomi\Desktop\XAUUSD5.csv")

checks: dict[str, bool] = {}


def chk(name: str, ok: bool, detail: str = "") -> None:
    checks[name] = bool(ok)
    mark = "[OK]" if ok else "[FAIL]"
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    print("=" * 70)
    print("  KN1 PRODUCTION LIVE AUDIT")
    print(f"  Install: {INSTALL}")
    print(f"  Config:  {APPDATA_CFG}")
    print("=" * 70)

    # ---- 1. Config ----
    print("\n--- 1. Config (AppData effective) ---")
    if not APPDATA_CFG.is_file():
        chk("AppData config_agent.json", False, "missing")
        return 1
    cfg = json.loads(APPDATA_CFG.read_text(encoding="utf-8-sig"))
    chk("agent.enabled", cfg.get("enabled") is True)
    chk("kn2.enabled=false", cfg.get("kn2", {}).get("enabled") is False, "KN1 mode")
    chk("primary_symbol=XAUUSD", cfg.get("primary_symbol") == "XAUUSD")
    kn_dim = int((cfg.get("knowledge_net") or {}).get("input_dim", 0))
    chk("knowledge_net.input_dim=68", kn_dim == 68, f"={kn_dim}")

    # ---- 2. Model files ----
    print("\n--- 2. KN1 model artifacts ---")
    for rel in (
        "models/knowledge_net.onnx",
        "models/knowledge_net.meta.json",
        "models/knowledge_scaler.pkl",
    ):
        p = INSTALL / rel.replace("/", "\\")
        ok = p.is_file()
        sz = p.stat().st_size if ok else 0
        chk(rel, ok, f"{sz:,} bytes" if ok else "MISSING")

    meta_path = INSTALL / "models" / "knowledge_net.meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chk("meta train_mode=v14_distill", meta.get("train_mode") == "v14_distill", str(meta.get("train_mode")))
        chk("meta input_dim=68", int(meta.get("input_dim", 0)) == 68, str(meta.get("input_dim")))
        chk("meta v14_agreement>=0.70", float(meta.get("v14_agreement", 0)) >= 0.70,
            f"{float(meta.get('v14_agreement', 0)):.2%}")

    # ---- 3. inference_cli agent_validate ----
    print("\n--- 3. inference_cli agent_validate ---")
    req = {
        "cmd": "agent_validate",
        "root": str(INSTALL),
        "config_path": str(APPDATA_CFG),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fin:
        json.dump(req, fin, ensure_ascii=False)
        req_path = fin.name
    out_path = req_path + ".out"
    proc = subprocess.run(
        [str(PYTHON), str(CLI), "--input", req_path, "--output", out_path],
        capture_output=True,
        text=True,
        cwd=str(INSTALL),
        timeout=180,
    )
    validate_out: dict = {}
    if Path(out_path).is_file():
        validate_out = json.loads(Path(out_path).read_text(encoding="utf-8-sig"))
    chk("agent_validate exit=0", proc.returncode == 0, (proc.stderr or "")[:120])
    chk("agent_validate ok", validate_out.get("ok") is True, str(validate_out.get("error", ""))[:120])
    chk("kn2_enabled=false", validate_out.get("kn2_enabled") is False, str(validate_out.get("kn2_enabled")))
    chk("knowledge_ready", validate_out.get("knowledge_ready") is True)

    # ---- 4. agent_tick on real M5 tail (via inference_cli, same as ZhuLong.exe) ----
    print("\n--- 4. agent_tick (real XAUUSD M5 tail, inference_cli) ---")
    if not CSV.is_file():
        chk("XAUUSD CSV", False, str(CSV))
    else:
        df = pd.read_csv(CSV, header=None, names=["date", "time", "open", "high", "low", "close", "volume"])
        df = df.dropna(subset=["open", "high", "low", "close"])
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), utc=True)
        df = df.sort_values("datetime").tail(500)
        bars = []
        for _, row in df.iterrows():
            ts = int(row["datetime"].timestamp())
            bars.append([
                ts,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row.get("volume") or 0),
            ])
        tick_req = {
            "cmd": "agent_tick",
            "root": str(INSTALL),
            "config_path": str(APPDATA_CFG),
            "symbols": ["XAUUSD"],
            "primary_symbol": "XAUUSD",
            "m5_includes_forming": False,
            "m5_bars_by_symbol": {"XAUUSD": bars},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fin:
            json.dump(tick_req, fin, ensure_ascii=False)
            tick_req_path = fin.name
        tick_out_path = tick_req_path + ".out"
        tick_proc = subprocess.run(
            [str(PYTHON), str(CLI), "--input", tick_req_path, "--output", tick_out_path],
            capture_output=True,
            text=True,
            cwd=str(INSTALL),
            timeout=180,
        )
        tick_out: dict = {}
        if Path(tick_out_path).is_file():
            tick_out = json.loads(Path(tick_out_path).read_text(encoding="utf-8-sig"))
        chk("agent_tick exit=0", tick_proc.returncode == 0, (tick_proc.stderr or "")[:120])
        chk("agent_tick ok", tick_out.get("ok") is True, str(tick_out.get("error", ""))[:120])
        results = tick_out.get("results") or []
        chk("agent_tick has results", len(results) > 0)
        first = results[-1] if results else {}
        chk("kn2_mode=false", first.get("kn2_mode") in (False, None), str(first.get("kn2_mode")))
        chk("knowledge_ready", first.get("knowledge_ready") is True, str(first.get("knowledge_ready")))
        meta = first.get("metadata") or {}
        kn_probs = first.get("knowledge_probs") or meta.get("knowledge_probs")
        if kn_probs:
            arr = np.asarray(kn_probs, dtype=np.float64).reshape(-1)[:3]
            s = float(arr.sum())
            pred = int(arr.argmax())
            names = ["short", "flat", "long"]
            chk("knowledge_probs sum~1", 0.99 <= s <= 1.01, f"sum={s:.4f}")
            print(f"    last bar KN probs: short={arr[0]:.3f} flat={arr[1]:.3f} long={arr[2]:.3f} -> {names[pred]}")
        sig = first.get("signal") or {}
        print(f"    action={first.get('action')} direction={sig.get('direction')} conf={sig.get('confidence')}")
        if first.get("action") in ("long", "short"):
            chk("signal direction when trade", sig.get("direction") in ("buy", "sell"), str(sig.get("direction")))

    # ---- 5. ZhuLong process ----
    print("\n--- 5. ZhuLong runtime ---")
    proc_out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq ZhuLong.exe"],
        capture_output=True,
        text=True,
    )
    running = "ZhuLong.exe" in proc_out.stdout
    chk("ZhuLong.exe running", running)

    # ---- Summary ----
    print("\n" + "=" * 70)
    passed = sum(checks.values())
    total = len(checks)
    fails = [k for k, v in checks.items() if not v]
    for k in fails:
        print(f"  [FAIL] {k}")
    print(f"\n  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  *** KN1 PRODUCTION READY ***")
    print("=" * 70)
    return 0 if passed == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
