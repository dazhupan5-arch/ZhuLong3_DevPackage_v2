#!/usr/bin/env python3
"""确保 R5 模型存在并部署到实机（PF 1.35 历史最优，未 PASS 时的回退方案）。"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

BASE_CONFIG = _ROOT / "config_training.yaml"
LOG_DIR = _ROOT / "logs" / "training"
DEPLOY_ROOT = Path(r"d:\Program Files\ZhuLong")
R5_PATCH = LOG_DIR / "xau_rl_r5_fallback.yaml"
R5_MODEL = _ROOT / "models" / "rl_agent_xau_r5.zip"
SYMBOL = "XAUUSD"
R5_BACKTEST = {
    "winrate": 0.36,
    "profit_factor": 1.352461577992917,
    "max_drawdown": 0.001226488967303655,
    "trades": 25,
}


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _run(args: list[str], *, label: str) -> int:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(args), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.call([sys.executable, "-u", *args], cwd=str(_ROOT), env=env)


def ensure_r5_model(*, retrain: bool = False) -> Path:
    if R5_MODEL.is_file() and not retrain:
        print(f"R5 model ready: {R5_MODEL}", flush=True)
        return R5_MODEL

    base_cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    patch = yaml.safe_load(R5_PATCH.read_text(encoding="utf-8"))
    cfg_path = LOG_DIR / "xau_rl_r5_deploy_train.yaml"
    merged = _deep_merge(base_cfg, patch)
    cfg_path.write_text(yaml.dump(merged, allow_unicode=True, default_flow_style=False), encoding="utf-8")

    timesteps = int((patch.get("rl") or {}).get("total_timesteps") or 600_000)
    ec = _run(
        [
            "scripts/train_rl_agent.py",
            "--symbol",
            SYMBOL,
            "--config",
            str(cfg_path.relative_to(_ROOT)),
            "--timesteps",
            str(timesteps),
        ],
        label=f"train R5 fallback ({timesteps} steps)",
    )
    if ec != 0:
        raise SystemExit(ec)

    src = _ROOT / "models" / "rl_agent_xau.zip"
    if not src.is_file():
        raise FileNotFoundError(f"Training finished but missing {src}")
    shutil.copy2(src, R5_MODEL)
    print(f"Saved R5 artifact: {R5_MODEL}", flush=True)
    return R5_MODEL


def deploy_to_live() -> None:
    print("\n== Deploy R5 fallback to live ==", flush=True)
    if not DEPLOY_ROOT.is_dir():
        print(f"WARN: deploy root missing {DEPLOY_ROOT}", flush=True)
        return

    shutil.copy2(R5_MODEL, _ROOT / "models" / "rl_agent_xau.zip")
    models = DEPLOY_ROOT / "models"
    models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(R5_MODEL, models / "rl_agent_xau.zip")

    scaler = _ROOT / "data" / "agent_state_scaler_xauusd.json"
    if scaler.is_file():
        data = DEPLOY_ROOT / "data"
        data.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scaler, data / "agent_state_scaler_xauusd.json")

    agent_src = _ROOT / "zhulong" / "agent"
    agent_dst = DEPLOY_ROOT / "zhulong" / "agent"
    if agent_src.is_dir():
        if agent_dst.is_dir():
            shutil.rmtree(agent_dst, ignore_errors=True)
        shutil.copytree(agent_src, agent_dst)

    exe = DEPLOY_ROOT / "ZhuLong.exe"
    if exe.is_file():
        subprocess.call(["taskkill", "/IM", "ZhuLong.exe", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        subprocess.Popen([str(exe)], cwd=str(DEPLOY_ROOT))

    summary = {
        "symbol": SYMBOL,
        "deployed_model": "R5_fallback",
        "acceptance_passed": False,
        "backtest_reference": R5_BACKTEST,
        "model_path": str(R5_MODEL),
        "deployed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "note": "R11 cycle ended without PASS; deployed best historical R5 (PF 1.35)",
    }
    out = _ROOT / "models" / "XAUUSD" / "agent_acceptance_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("R5 deploy complete.", flush=True)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train (if needed) and deploy R5 XAU RL fallback")
    parser.add_argument("--retrain", action="store_true", help="Force retrain R5 even if artifact exists")
    parser.add_argument("--train-only", action="store_true", help="Only ensure rl_agent_xau_r5.zip exists")
    args = parser.parse_args()

    ensure_r5_model(retrain=args.retrain)
    if not args.train_only:
        deploy_to_live()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
