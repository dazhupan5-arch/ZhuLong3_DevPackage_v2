#!/usr/bin/env python3
"""V15 必须通过验收后才允许 KN1 蒸馏与部署。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _v15_passed() -> tuple[bool, dict]:
    cfg = ROOT / "models" / "XAUUSD" / "v15" / "config_v15.json"
    rep = ROOT / "data" / "training" / "reports" / "v15" / "XAUUSD" / "train_report_v15.json"
    if rep.is_file():
        data = json.loads(rep.read_text(encoding="utf-8"))
        return bool(data.get("passed")), data
    if cfg.is_file():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return bool(data.get("passed")), data
    return False, {}


def main() -> int:
    steps = [
        ([sys.executable, "-u", str(ROOT / "scripts" / "train_v15.py"), "--symbol", "XAUUSD"], "V15 XGBoost"),
    ]
    for cmd, name in steps:
        print(f"\n=== {name} ===", flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"ABORT: {name} 未通过 (exit {rc})，禁止 KN1 蒸馏/部署", flush=True)
            return rc

    ok, rep = _v15_passed()
    if not ok:
        print("ABORT: V15 验收 passed=false", rep.get("failures", []), flush=True)
        return 2

    kn_steps = [
        ([sys.executable, "-u", str(ROOT / "scripts" / "prepare_knowledge_data_v15.py")], "KN data"),
        ([sys.executable, "-u", str(ROOT / "scripts" / "train_knowledge_net_v15.py")], "KN distill"),
        ([sys.executable, str(ROOT / "scripts" / "convert_knowledge_net_to_onnx.py"),
          "--model", str(ROOT / "models" / "knowledge_net_v15.pth"),
          "--out", str(ROOT / "models" / "knowledge_net_v15.onnx"), "--no-benchmark"], "KN ONNX"),
    ]
    cache = ROOT / "data" / "training_data_v15_v15_teacher.npy"
    if cache.is_file():
        cache.unlink()
        print(f"已清除旧 V15 教师缓存: {cache}")

    for cmd, name in kn_steps:
        print(f"\n=== {name} ===", flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"ABORT: {name} failed (exit {rc})", flush=True)
            return rc

    print("\nOK: V15 + KN1 全部通过，可部署", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
