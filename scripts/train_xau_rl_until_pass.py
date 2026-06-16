#!/usr/bin/env python3
"""XAUUSD PPO：循环训练 + 2025 回测，直至 PASS 后自动部署实机。"""

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
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

BASE_CONFIG = _ROOT / "config_training.yaml"
LOG_DIR = _ROOT / "logs" / "training"
DEPLOY_ROOT = Path(r"d:\Program Files\ZhuLong")
SYMBOL = "XAUUSD"
MAX_CYCLES = 1
R5_MODEL = _ROOT / "models" / "rl_agent_xau_r5.zip"
R5_PATCH = LOG_DIR / "xau_rl_r5_fallback.yaml"

# 每轮覆盖 config_training.yaml 中的片段（深度合并）
# 新方案：修复 R1-R3 暴露的奖惩不对称 + 探索不足（认知对齐 1:1, 降低回撤惩罚, 提高探索率）
RL_ROUNDS: list[dict[str, Any]] = [
    {
        "note": "R8 R5++ higher open bonus + wider DD 800k",
        "rl": {
            "total_timesteps": 800_000,
            "xau": {
                "open_reward_bonus": 0.08,
                "cognition_align": {"same_direction_bonus": 0.10, "opposite_direction_penalty": 0.06},
                "forced_explore": {"steps": 100000, "prob": 0.55},
                "ent_coef_start": 0.15,
                "ent_coef_end": 0.02,
            },
        },
        "env": {"drawdown_penalty_coef": 0.02, "drawdown_penalty_trigger": 0.25},
    },
    {
        "note": "R9 higher open incentive + longer 1M",
        "rl": {
            "total_timesteps": 1_000_000,
            "xau": {
                "open_reward_bonus": 0.10,
                "cognition_align": {"same_direction_bonus": 0.12, "opposite_direction_penalty": 0.05},
                "forced_explore": {"steps": 120000, "prob": 0.55},
                "ent_coef_start": 0.18,
                "ent_coef_end": 0.015,
            },
        },
        "env": {"drawdown_penalty_coef": 0.02, "drawdown_penalty_trigger": 0.28},
    },
    {
        "note": "R10 heavy explore + 1.2M converge",
        "rl": {
            "total_timesteps": 1_200_000,
            "xau": {
                "open_reward_bonus": 0.12,
                "cognition_align": {"same_direction_bonus": 0.12, "opposite_direction_penalty": 0.04},
                "forced_explore": {"steps": 150000, "prob": 0.55},
                "ent_coef_start": 0.20,
                "ent_coef_end": 0.01,
            },
        },
        "env": {"drawdown_penalty_coef": 0.01, "drawdown_penalty_trigger": 0.30},
    },
    {
        "note": "R11 1.5M conservative fine-tune after explore",
        "rl": {
            "total_timesteps": 1_500_000,
            "xau": {
                "open_reward_bonus": 0.10,
                "cognition_align": {"same_direction_bonus": 0.10, "opposite_direction_penalty": 0.05},
                "forced_explore": {"steps": 100000, "prob": 0.45},
                "ent_coef_start": 0.12,
                "ent_coef_end": 0.008,
            },
        },
        "env": {"drawdown_penalty_coef": 0.01, "drawdown_penalty_trigger": 0.30},
    },
]


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


def _write_round_config(base: dict, patch: dict, path: Path) -> None:
    merged = _deep_merge(base, patch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(merged, allow_unicode=True, default_flow_style=False), encoding="utf-8")


def _load_backtest_report() -> dict[str, Any]:
    p = LOG_DIR / "backtest_XAUUSD_2025.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _backup_model(tag: str) -> None:
    src = _ROOT / "models" / "rl_agent_xau.zip"
    if src.is_file():
        dst = _ROOT / "models" / f"rl_agent_xau_{tag}.zip"
        shutil.copy2(src, dst)


def _deploy_r5_fallback() -> None:
    ec = subprocess.call(
        [sys.executable, "-u", str(_ROOT / "scripts" / "deploy_xau_r5_fallback.py")],
        cwd=str(_ROOT),
    )
    if ec != 0:
        raise SystemExit(ec)


def _deploy_to_live() -> None:
    print("\n== Deploy PASS model to live ==", flush=True)
    if not DEPLOY_ROOT.is_dir():
        print(f"WARN: deploy root missing {DEPLOY_ROOT}", flush=True)
        return
    models = DEPLOY_ROOT / "models"
    data = DEPLOY_ROOT / "data"
    models.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_ROOT / "models" / "rl_agent_xau.zip", models / "rl_agent_xau.zip")
    scaler = _ROOT / "data" / "agent_state_scaler_xauusd.json"
    if scaler.is_file():
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
    print("Deploy complete.", flush=True)


def _train_and_backtest(cfg_path: Path, timesteps: int) -> bool:
    ec = _run(
        ["scripts/train_rl_agent.py", "--symbol", SYMBOL, "--config", str(cfg_path.relative_to(_ROOT)), "--timesteps", str(timesteps)],
        label=f"train_rl_agent ({timesteps} steps)",
    )
    if ec != 0:
        return False
    ec = _run(
        ["scripts/backtest_rl.py", "--symbol", SYMBOL, "--config", str(cfg_path.relative_to(_ROOT))],
        label="backtest_rl 2025",
    )
    return ec == 0


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    master_log = LOG_DIR / "xau_rl_until_pass.log"
    base_cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cycle = 0

    with master_log.open("a", encoding="utf-8") as logfh:
        def log(msg: str) -> None:
            line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}"
            print(line, flush=True)
            logfh.write(line + "\n")
            logfh.flush()

        log("XAU RL until-pass loop started")

        while True:
            cycle += 1
            for i, spec in enumerate(RL_ROUNDS, 1):
                note = spec.get("note", f"round{i}")
                patch = {k: v for k, v in spec.items() if k != "note"}
                timesteps = int((patch.get("rl") or {}).get("total_timesteps") or base_cfg.get("rl", {}).get("total_timesteps", 800_000))
                cfg_path = LOG_DIR / f"xau_rl_round_c{cycle}_r{i}.yaml"
                _write_round_config(base_cfg, patch, cfg_path)

                log(f"=== Cycle {cycle} Round {i}/{len(RL_ROUNDS)}: {note} ({timesteps} steps) ===")
                _backup_model(f"c{cycle}_r{i}_before")

                if _train_and_backtest(cfg_path, timesteps):
                    report = _load_backtest_report()
                    log(f"PASS: {json.dumps(report, ensure_ascii=False)}")
                    summary = {
                        "symbol": SYMBOL,
                        "acceptance_passed": True,
                        "cycle": cycle,
                        "round": i,
                        "note": note,
                        "timesteps": timesteps,
                        "backtest": report,
                        "passed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    }
                    out = _ROOT / "models" / "XAUUSD" / "agent_acceptance_summary.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                    _deploy_to_live()
                    log("All done.")
                    return 0

                report = _load_backtest_report()
                log(f"FAIL: {json.dumps(report, ensure_ascii=False)}")
                fail_log = LOG_DIR / "xau_rl_attempts.jsonl"
                with fail_log.open("a", encoding="utf-8") as af:
                    af.write(json.dumps({"cycle": cycle, "round": i, "note": note, **report}, ensure_ascii=False) + "\n")

                pause_file = LOG_DIR / "pause_after_round.json"
                if pause_file.is_file():
                    try:
                        pause_cfg = json.loads(pause_file.read_text(encoding="utf-8"))
                        if int(pause_cfg.get("cycle", 0)) == cycle and int(pause_cfg.get("round", 0)) == i:
                            log(f"Paused after cycle {cycle} round {i}: {pause_cfg.get('reason', 'pause_after_round.json')}")
                            return 0
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

                if cycle >= MAX_CYCLES and i >= len(RL_ROUNDS):
                    log("Last round failed without PASS — deploying R5 fallback")
                    _deploy_r5_fallback()
                    log("All done (R5 fallback deployed).")
                    return 0

            if cycle >= MAX_CYCLES:
                log(f"Cycle {cycle} exhausted without PASS — deploying R5 fallback")
                _deploy_r5_fallback()
                log("All done (R5 fallback deployed).")
                return 0

            log(f"Cycle {cycle} exhausted, starting cycle {cycle + 1}...")


if __name__ == "__main__":
    raise SystemExit(main())
