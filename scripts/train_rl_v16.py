#!/usr/bin/env python3
"""V16 正式 PPO 训练（horizon_v16 NPZ + ONNX，与实盘 Agent 状态一致）。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cmd = [
        sys.executable,
        "-u",
        str(_ROOT / "scripts" / "train_rl_agent.py"),
        "--v16",
        "--symbol",
        "XAUUSD",
        *sys.argv[1:],
    ]
    return subprocess.call(cmd, cwd=str(_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
