#!/usr/bin/env python3
"""USOIL 智能体：清理不合格产物 → 循环训练直至 KN + PPO 回测 PASS。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

NPZ_PATH = _ROOT / "data" / "oil_training_data.npz"

# struct30 2-block ResNet (prepare 较慢，但之前验证过 acc=68%)
KN_ATTEMPTS = [
    {
        "note": "struct30 2-block h64 chrono",
        "prepare": "struct30",
        "epochs": 120,
        "lr": 0.001,
        "hidden_dim": 64,
        "patience": 20,
        "num_res_blocks": 2,
        "no_shuffle": True,
        "val_ratio": 0.12,
    },
    {
        "note": "struct30 2-block h96 chrono",
        "prepare": False,
        "epochs": 150,
        "lr": 0.0008,
        "hidden_dim": 96,
        "patience": 25,
        "num_res_blocks": 2,
        "no_shuffle": True,
        "val_ratio": 0.12,
    },
    {
        "note": "struct30 2-block smote",
        "prepare": False,
        "epochs": 180,
        "lr": 0.001,
        "hidden_dim": 64,
        "patience": 30,
        "num_res_blocks": 2,
        "no_shuffle": True,
        "val_ratio": 0.10,
        "smote_ratio": 0.5,
    },
    {
        "note": "struct30 2-block h96 lr low",
        "prepare": False,
        "epochs": 200,
        "lr": 0.0005,
        "hidden_dim": 96,
        "patience": 35,
        "num_res_blocks": 2,
        "no_shuffle": True,
        "val_ratio": 0.12,
    },
    {
        "note": "struct30 2-block final",
        "prepare": False,
        "epochs": 250,
        "lr": 0.0004,
        "hidden_dim": 96,
        "patience": 40,
        "num_res_blocks": 2,
        "no_shuffle": True,
        "val_ratio": 0.10,
    },
]

RL_TIMESTEPS = [500_000, 650_000, 800_000, 1_000_000, 1_200_000]

OIL_ARTIFACTS = [
    "models/knowledge_net_oil.pth",
    "models/knowledge_net_oil.onnx",
    "models/knowledge_net_oil.meta.json",
    "models/knowledge_scaler_oil.pkl",
    "models/rl_agent_oil.zip",
    "models/USOIL/agent_acceptance_summary.json",
    "logs/training/backtest_USOIL_2025.json",
    "logs/training/knowledge_USOIL.log",
    "logs/training/rl_metrics_USOIL.jsonl",
    "logs/training/rl_USOIL.json",
]


def _run(args: list[str], *, label: str) -> int:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(args), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.call([sys.executable, "-u", *args], cwd=str(_ROOT), env=env)


def _npz_is_valid() -> bool:
    if not NPZ_PATH.is_file():
        return False
    try:
        data = np.load(NPZ_PATH)
        struct = data["struct"]
        return struct.ndim == 2 and struct.shape[0] >= 10_000
    except Exception:
        return False


def _clean_oil_agent() -> None:
    print("\n== 清理不合格 USOIL 智能体产物 ==", flush=True)
    for rel in OIL_ARTIFACTS:
        p = _ROOT / rel
        if p.is_file():
            p.unlink()
            print(f"  removed {rel}")
    rl_dir = _ROOT / "logs" / "rl" / "usoil"
    if rl_dir.is_dir():
        shutil.rmtree(rl_dir, ignore_errors=True)
        print("  removed logs/rl/usoil/")
    best = _ROOT / "models" / "best_model.zip"
    if best.is_file():
        best.unlink()


def _set_agent_enabled(enabled: bool) -> None:
    cfg_path = _ROOT / "config" / "config_agent.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    cfg["enabled"] = enabled
    usoil = (cfg.get("symbols") or {}).get("USOIL")
    if isinstance(usoil, dict):
        usoil["enabled"] = enabled
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"config_agent.enabled = {enabled}, symbols.USOIL.enabled = {enabled}")


def _ensure_struct_npz(force: bool = False) -> bool:
    if not force and _npz_is_valid():
        data = np.load(NPZ_PATH)
        print(f"已有有效 npz: {data['struct'].shape}，跳过 prepare", flush=True)
        return True
    if NPZ_PATH.is_file():
        print(f"强制重建: 删除旧 npz ({NPZ_PATH.stat().st_size} bytes)", flush=True)
        NPZ_PATH.unlink()
    ec = _run(
        ["scripts/prepare_training_data.py", "--symbol", "USOIL", "--n-jobs", "1", "--max-rows", "100000"],
        label="prepare_training_data (struct30, n_jobs=1, max_rows=100k)",
    )
    return ec == 0 and _npz_is_valid()


def _train_knowledge_once(spec: dict) -> bool:
    prepare = spec.get("prepare")
    if prepare == "struct30":
        if not _ensure_struct_npz(force=True):
            return False
    elif prepare:
        ec = _run(
            ["scripts/prepare_knowledge_data.py", "--symbol", "USOIL"],
            label="prepare_knowledge_data (V14 68d)",
        )
        if ec != 0:
            return False

    kn_args = [
        "scripts/train_knowledge_net.py",
        "--symbol",
        "USOIL",
        "--epochs",
        str(spec.get("epochs", 150)),
        "--lr",
        str(spec.get("lr", 0.001)),
        "--hidden-dim",
        str(spec.get("hidden_dim", 64)),
        "--patience",
        str(spec.get("patience", 25)),
        "--select-by",
        "accuracy",
    ]
    if spec.get("num_res_blocks"):
        kn_args.extend(["--num-res-blocks", str(spec["num_res_blocks"])])
    if spec.get("no_shuffle"):
        kn_args.append("--no-shuffle")
    if "smote_ratio" in spec:
        kn_args.extend(["--smote-ratio", str(spec["smote_ratio"])])
    if "val_ratio" in spec:
        kn_args.extend(["--val-ratio", str(spec["val_ratio"])])

    ec = _run(kn_args, label=f"train_knowledge_net ({spec['note']})")
    return ec == 0


def _train_rl_and_backtest(timesteps: int) -> bool:
    ec = _run(
        ["scripts/train_rl_agent.py", "--symbol", "USOIL", "--timesteps", str(timesteps)],
        label=f"train_rl_agent ({timesteps} steps)",
    )
    if ec != 0:
        return False
    ec = _run(
        ["scripts/backtest_rl.py", "--symbol", "USOIL", "--year", "2025"],
        label="backtest_rl 2025",
    )
    return ec == 0


def _run_knowledge_phase() -> bool:
    for i, spec in enumerate(KN_ATTEMPTS, 1):
        print(f"\n--- KnowledgeNet 尝试 {i}/{len(KN_ATTEMPTS)}: {spec['note']} ---", flush=True)
        if i > 1:
            _clean_oil_agent()
        if _train_knowledge_once(spec):
            ec = _run(
                ["scripts/convert_knowledge_net_to_onnx.py", "--symbol", "USOIL", "--no-benchmark"],
                label="convert_knowledge_net_onnx",
            )
            if ec == 0:
                return True
        print("KnowledgeNet 未达标，继续下一组参数...", flush=True)
    return False


def _run_rl_phase() -> bool:
    for j, steps in enumerate(RL_TIMESTEPS, 1):
        print(f"\n--- PPO+回测 尝试 {j}/{len(RL_TIMESTEPS)} ({steps} steps) ---", flush=True)
        if j > 1:
            p = _ROOT / "models" / "rl_agent_oil.zip"
            if p.is_file():
                p.unlink()
        if _train_rl_and_backtest(steps):
            return True
        print("PPO 回测未达标，增加 timesteps 重训...", flush=True)
    return False


def _write_pass_summary() -> Path:
    summary = {
        "symbol": "USOIL",
        "acceptance_passed": True,
        "knowledge_net": "models/knowledge_net_oil.onnx",
        "rl_agent": "models/rl_agent_oil.zip",
        "passed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "note": "KnowledgeNet + PPO 回测均已 PASS，可装机",
    }
    out = _ROOT / "models" / "USOIL" / "agent_acceptance_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> int:
    _set_agent_enabled(False)
    cycle = 0
    while True:
        cycle += 1
        print(f"\n========== USOIL 智能体训练 第 {cycle} 轮 ==========", flush=True)
        _clean_oil_agent()

        if not _ensure_struct_npz(force=False):
            print("数据准备失败，60 秒后重试...", flush=True)
            time.sleep(60)
            continue

        if not _run_knowledge_phase():
            print("\nKnowledgeNet 全部尝试均未 PASS，重建 struct 缓存后重试...", flush=True)
            cache = _ROOT / "data" / "training" / "struct" / "USOIL" / "struct_features.npz"
            if cache.is_file():
                cache.unlink()
            if NPZ_PATH.is_file():
                NPZ_PATH.unlink()
            time.sleep(30)
            continue

        if _run_rl_phase():
            break

        print("\nPPO 回测未 PASS，整轮重试（保留 struct 缓存）...", flush=True)
        time.sleep(30)

        out = _write_pass_summary()
        _set_agent_enabled(True)
        print("\n=== 全部验收 PASS — config_agent.enabled=true，可装机 ===", flush=True)
        print(f"摘要: {out}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
