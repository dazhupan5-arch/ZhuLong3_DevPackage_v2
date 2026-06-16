#!/usr/bin/env python3
"""V16 正式上线管线：全量数据 → Horizon → ONNX → PPO → 验收（passed 才可部署）。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, args: list[str]) -> int:
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}", flush=True)
    cmd = [sys.executable, "-u", *args]
    rc = subprocess.call(cmd, cwd=str(_ROOT))
    if rc != 0:
        print(f"FAILED: {label} (exit {rc})", flush=True)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--skip-accept", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--rl-quick", action="store_true", help="PPO 冒烟 5000 步（仅调试）")
    parser.add_argument("--retrain", action="store_true", help="Horizon 重训调参（更强方向类权重 + 更长训练）")
    args = parser.parse_args()

    train_extra: list[str] = []
    if args.retrain:
        args.epochs = max(args.epochs, 100)
        train_extra = [
            "--epochs", str(args.epochs),
            "--class-weights", "2.8,0.65,2.8",
            "--smote-ratio", "0.58",
            "--patience", "22",
            "--lr", "0.0004",
            "--hidden-dim", "128",
            "--focal-gamma", "2.0",
            "--log-suffix", "retrain2",
        ]

    if not args.skip_prepare:
        rc = _run(
            "Step 1/6 prepare full data (jobs=1)",
            ["scripts/prepare_horizon_v16_data.py", "--symbol", "XAUUSD", "--jobs", str(args.jobs)],
        )
        if rc != 0:
            return rc

    if not args.skip_enrich:
        rc = _run(
            "Step 2/6 enrich NPZ (OHLCV fallback)",
            ["scripts/enrich_horizon_v16_npz.py", "--symbol", "XAUUSD"],
        )
        if rc != 0:
            return rc

    if not args.skip_train:
        train_args = ["scripts/train_horizon_v16.py"]
        if train_extra:
            train_args.extend(train_extra)
        else:
            train_args.extend(["--epochs", str(args.epochs)])
        rc = _run(
            f"Step 3/6 train horizon ({args.epochs} epochs)",
            train_args,
        )
        if rc != 0:
            return rc

    if not args.skip_onnx:
        rc = _run(
            "Step 4/6 export Horizon ONNX",
            [
                "scripts/convert_knowledge_net_to_onnx.py",
                "--model",
                "models/horizon_v16.pth",
                "--out",
                "models/horizon_v16.onnx",
                "--no-benchmark",
            ],
        )
        if rc != 0:
            return rc

    if not args.skip_rl:
        rl_args = ["scripts/train_rl_v16.py"]
        if args.rl_quick:
            rl_args.append("--quick")
        rc = _run("Step 5/6 train PPO (V16 horizon state)", rl_args)
        if rc != 0:
            return rc

    if not args.skip_accept:
        rc = _run(
            "Step 6/6 full acceptance (Horizon + PPO + Agent) + apply",
            ["scripts/accept_horizon_v16.py", "--apply"],
        )
        if rc != 0:
            return rc

    print("\nV16 pipeline complete. Deploy only if acceptance_report.json passed=true.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
