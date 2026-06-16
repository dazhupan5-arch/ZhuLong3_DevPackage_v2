#!/usr/bin/env python3
"""R11 回测结束后：PASS 则不动；FAIL 则部署 R5 并终止 until-pass 循环（避免进入 Cycle 2）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
ATTEMPTS = _ROOT / "logs" / "training" / "xau_rl_attempts.jsonl"
LOG = _ROOT / "logs" / "training" / "deploy_r5_watch_runner.log"
CYCLE = 1
ROUND = 4
R11_NOTE = "R11"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _find_r11_result() -> dict | None:
    if not ATTEMPTS.is_file():
        return None
    hit = None
    for line in ATTEMPTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("cycle", 0)) != CYCLE or int(row.get("round", 0)) != ROUND:
            continue
        if R11_NOTE not in str(row.get("note", "")):
            continue
        hit = row
    return hit


def _kill_until_pass() -> None:
    if os.name != "nt":
        return
    for pattern in ("%train_xau_rl_until_pass%", "%train_rl_agent%"):
        subprocess.run(
            ["wmic", "process", "where", f"CommandLine like '{pattern}'", "call", "terminate"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> int:
    _log(f"Watching {ATTEMPTS.name} for cycle={CYCLE} round={ROUND} ({R11_NOTE})")

    while True:
        row = _find_r11_result()
        if row is None:
            time.sleep(30)
            continue

        passed = bool(row.get("passed"))
        _log(f"R11 result: passed={passed} {json.dumps(row, ensure_ascii=False)}")

        if passed:
            _log("R11 PASS — no R5 fallback deploy")
            return 0

        _log("R11 FAIL — deploying R5 fallback")
        ec = subprocess.call(
            [sys.executable, "-u", str(_ROOT / "scripts" / "deploy_xau_r5_fallback.py")],
            cwd=str(_ROOT),
        )
        if ec != 0:
            _log(f"R5 deploy failed exit={ec}")
            return ec

        time.sleep(2)
        _kill_until_pass()
        _log("R5 deployed; until-pass loop stopped before Cycle 2")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
