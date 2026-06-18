#!/usr/bin/env python3
"""Horizon V16 post-restart check (log + optional CLI when app stopped)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from v16_cli_runner import APPDATA, run_cli  # noqa: E402

INSTALL = Path(r"C:\Program Files\ZhuLong")
CFG = APPDATA / "config_agent.json"
LOG_DIR = APPDATA / "logs"


def _latest_log() -> Path | None:
    files = sorted(LOG_DIR.glob("log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _log_has(patterns: list[str]) -> bool:
    log = _latest_log()
    if not log:
        return False
    text = log.read_text(encoding="utf-8", errors="replace")
    tail = text[-80000:]
    return any(re.search(p, tail) for p in patterns)


def _zhulong_running() -> bool:
    try:
        import subprocess
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-Process ZhuLong -EA SilentlyContinue).Count -gt 0"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip().lower() == "true"
    except Exception:
        return False


def main() -> int:
    print("=== Horizon V16 POST-RESTART CHECK ===")
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    meta_path = APPDATA / "models" / "horizon_v16.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    hp = (cfg.get("architecture") or {}).get("horizon_predictor") or {}
    print(f"config: min_conf={hp.get('min_direction_confidence')} kn2.enabled={(cfg.get('kn2') or {}).get('enabled')}")
    print(f"meta: trial={meta.get('trial')} f1={meta.get('macro_f1')} passed={meta.get('passed')}")

    zhu_long_running = _zhulong_running()
    if zhu_long_running and _log_has([
        r"智能体环境校验通过",
        r"agent_validate 子进程超时，已通过 C# 原生快检",
        r"智能体模式已就绪",
    ]):
        print("validate: ok=True (from live log — app running)")
        v_ok = True
    else:
        v = run_cli({"cmd": "agent_validate", "config_path": str(CFG)})
        v_ok = bool(v.get("ok"))
        print(f"validate: ok={v_ok} arch={v.get('architecture')} error={v.get('error')}")

    tick_ok = _log_has([r"子进程智能体完成", r"TradingAgent RL 智能体已启用", r"开机智能体评估"])
    print(f"live tick in log: {tick_ok}")
    if not v_ok:
        return 1
    if zhu_long_running and not tick_ok:
        print("WARN: app running but no agent tick in log yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
