#!/usr/bin/env python3
"""带崩溃保护的 RL 训练：异常时自动保存 checkpoint，支持断点续训。"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

from zhulong.agent.training_utils import ensure_logs_dir

CHECKPOINT = _ROOT / "models" / "rl_agent_xau_checkpoint.zip"
CRASH_LOG = ensure_logs_dir() / "rl_crash.log"


def _trace(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(CRASH_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    _trace("safe_train started")
    chk = None
    try:
        # Import inside try so we catch import errors too
        from scripts.train_rl_agent import main as train_main
        import argparse

        # Monkey-patch sys.argv for train_rl_agent
        old_argv = sys.argv[:]
        sys.argv = [
            "train_rl_agent.py",
            "--symbol", "XAUUSD",
            "--config", "config_rl_200w_2025.yaml",
        ]
        try:
            code = train_main()
        finally:
            sys.argv = old_argv

        _trace(f"train_main returned {code}")
        return code

    except KeyboardInterrupt:
        _trace("KeyboardInterrupt - exiting")
        return 1
    except SystemExit as e:
        _trace(f"SystemExit code={e.code}")
        return e.code if isinstance(e.code, int) else 1
    except Exception:
        tb = traceback.format_exc()
        _trace(f"CRASHED:\n{tb}")

        # Try to find and save the model reference from train_rl_agent's scope
        # This is a best-effort; the model variable may be out of scope
        _trace("attempting emergency save...")
        try:
            import gc
            from stable_baselines3 import PPO
            for obj in gc.get_objects():
                if isinstance(obj, PPO):
                    obj.save(str(CHECKPOINT.with_suffix("")))
                    _trace(f"saved checkpoint to {CHECKPOINT}")
                    break
        except Exception as e2:
            _trace(f"emergency save also failed: {e2}")

        return 2


if __name__ == "__main__":
    raise SystemExit(main())
