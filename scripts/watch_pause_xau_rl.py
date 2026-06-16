#!/usr/bin/env python3
"""R3 回测写入 attempts.jsonl 后，终止 until-pass 循环（避免自动进入 R4）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
PAUSE = _ROOT / "logs" / "training" / "pause_after_round.json"
ATTEMPTS = _ROOT / "logs" / "training" / "xau_rl_attempts.jsonl"
LOG = _ROOT / "logs" / "training" / "pause_watch.log"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _attempts_done(cycle: int, round_n: int) -> bool:
    if not ATTEMPTS.is_file():
        return False
    for line in ATTEMPTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("cycle", 0)) == cycle and int(row.get("round", 0)) == round_n:
            return True
    return False


def _kill_until_pass() -> None:
    if os.name != "nt":
        return
    subprocess.run(
        ["wmic", "process", "where", "CommandLine like '%train_xau_rl_until_pass%'", "call", "terminate"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["wmic", "process", "where", "CommandLine like '%train_rl_agent%'", "call", "terminate"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    if not PAUSE.is_file():
        _log(f"Missing {PAUSE}")
        return 1
    cfg = json.loads(PAUSE.read_text(encoding="utf-8"))
    cycle = int(cfg["cycle"])
    round_n = int(cfg["round"])
    _log(f"Watching for cycle={cycle} round={round_n} backtest in {ATTEMPTS.name}")

    while not _attempts_done(cycle, round_n):
        time.sleep(30)

    _log(f"Round {round_n} backtest recorded — stopping training loop")
    time.sleep(2)
    _kill_until_pass()
    _log("Stop signal sent. until-pass loop should not enter next round.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
