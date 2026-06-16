#!/usr/bin/env python3
"""Parse ZhuLong Serilog for live closure checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    log_dir = Path.home() / "AppData" / "Roaming" / "ZhuLong" / "logs"
    logs = sorted(log_dir.glob("log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        print(json.dumps({"ok": False, "error": "no log"}))
        return 1
    tail = logs[0].read_text(encoding="utf-8", errors="replace").splitlines()[-800:]
    m1 = any("M1 XAUUSD" in ln for ln in tail)
    pipe = any("已连接数据管道" in ln or "MT5 已连接数据管道" in ln for ln in tail)
    infer_fail = [ln for ln in tail if "信号失败 XAUUSD" in ln]
    infer_ok = [ln for ln in tail if "推理完成 XAUUSD" in ln]
    infer_start = [ln for ln in tail if "推理开始 XAUUSD" in ln]
    model_ready = any("正式模型已就绪" in ln for ln in tail)
    out = {
        "ok": True,
        "log": logs[0].name,
        "m1": m1,
        "pipe": pipe,
        "model_ready": model_ready,
        "infer_ok": bool(infer_ok),
        "infer_fail_last": infer_fail[-1].strip() if infer_fail else "",
        "infer_ok_last": infer_ok[-1].strip() if infer_ok else "",
        "infer_started": bool(infer_start),
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
