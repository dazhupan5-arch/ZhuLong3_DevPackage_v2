#!/usr/bin/env python3
"""Audit live log for V16 agent validate + tick closure."""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

LOG_DIR = Path.home() / "AppData" / "Roaming" / "ZhuLong" / "logs"


def read_log(path: Path) -> str:
    for _ in range(8):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            time.sleep(2)
    return ""


def main() -> int:
    logs = sorted(LOG_DIR.glob("log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        print("[FAIL] no log file")
        return 1
    log = logs[0]
    text = read_log(log)
    if not text:
        print(f"[FAIL] cannot read {log}")
        return 1

    start = text.rfind("智能体环境校验开始")
    segment = text[start:] if start >= 0 else text[-200000:]
    tail = text[-120000:]

    has_validate = bool(re.search(
        r"智能体环境校验通过|agent_validate 子进程超时，已通过 C# 原生快检|V16 全栈热加载完成|V16 全栈已就绪",
        segment,
    ))
    has_ready = bool(re.search(
        r"智能体模式已就绪|V16 全栈热加载完成|V16 全栈已就绪|engine_preloaded",
        segment + tail,
    ))
    has_tick = bool(re.search(
        r"子进程智能体完成|Worker 智能体完成|TradingAgent RL 智能体已启用|开机智能体评估完成|\[V16·Horizon\]|\\[V16.Horizon\\]",
        segment + tail,
    ))
    has_fail = bool(re.search(
        r"No module named 'mt5_ops'|智能体环境校验失败|推理子进程超时 \(120s\)",
        segment,
    ))

    print(f"log: {log.name}")
    print(f"  validate_pass: {has_validate}")
    print(f"  agent_ready:   {has_ready}")
    print(f"  agent_tick:    {has_tick}")
    print(f"  hard_fail:     {has_fail}")

    ok = has_tick and not has_fail and (has_validate or has_ready or bool(re.search(r"\[V16", tail)))
    print("=== LIVE LOG: PASS ===" if ok else "=== LIVE LOG: FAIL ===")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
