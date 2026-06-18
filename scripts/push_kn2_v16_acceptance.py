#!/usr/bin/env python3
"""KN2 V16 冲刺验收：多组 class_weights 试训 → accept_kn2_v16。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

WEIGHT_TRIALS = [
    {"name": "cw_default", "class_weights": "0.85,2.5,2.5,1.0,1.0,1.0", "lr": 0.0004},
    {"name": "cw_aggressive", "class_weights": "0.70,3.0,3.0,1.0,1.0,1.0", "lr": 0.00045},
    {"name": "cw_very_aggressive", "class_weights": "0.60,3.5,3.5,1.0,1.0,1.0", "lr": 0.0005},
    {"name": "cw_balanced", "class_weights": "0.75,2.8,2.8,1.0,1.0,1.0", "lr": 0.00042},
]


def _run_train(npz: Path, out: Path, trial: dict, *, epochs: int, patience: int, batch_size: int, device: str) -> int:
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "train_kn2_v16.py"),
        "--npz",
        str(npz),
        "--output",
        str(out),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--class-weights",
        trial["class_weights"],
        "--lr",
        str(trial["lr"]),
        "--mode",
        "fast",
    ]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(_ROOT))


def _accept(model: Path, npz: Path) -> tuple[int, dict]:
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "accept_kn2_v16.py"),
        "--model",
        str(model),
        "--npz",
        str(npz),
    ]
    rc = subprocess.call(cmd, cwd=str(_ROOT))
    report_path = _ROOT / "data" / "training" / "reports" / "kn2_v16" / "acceptance_report.json"
    detail = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    return rc, detail


def _score(acc: dict) -> float:
    pc = (acc.get("val_eval") or {}).get("per_class") or {}
    score = 0.0
    for side in ("long", "short"):
        sc = pc.get(side) or {}
        score += float(sc.get("precision", 0)) + float(sc.get("pred_pct", 0))
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/clean/kn2_training_v16.npz")
    parser.add_argument("--output", default="models/kn2_trader_v16.pth")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trials", default="")
    args = parser.parse_args()

    npz = _ROOT / args.npz
    out = _ROOT / args.output
    if not npz.is_file():
        print(f"缺少 {npz}")
        return 1

    names = [t.strip() for t in args.trials.split(",") if t.strip()]
    trials = [t for t in WEIGHT_TRIALS if not names or t["name"] in names]

    trial_dir = _ROOT / "models" / "tune_trials" / "kn2_v16_push"
    trial_dir.mkdir(parents=True, exist_ok=True)
    report_dir = _ROOT / "data" / "training" / "reports" / "kn2_v16"
    report_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    best: dict | None = None
    best_score = -1.0

    for trial in trials:
        print(f"\n=== KN2 PUSH {trial['name']} ===", flush=True)
        trial_out = trial_dir / f"{trial['name']}_kn2_trader_v16.pth"
        rc = _run_train(
            npz,
            trial_out,
            trial,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=args.device,
        )
        if rc != 0:
            results.append({"name": trial["name"], "train_rc": rc, "error": "train_failed"})
            continue
        acc_rc, acc = _accept(trial_out, npz)
        r = {
            "name": trial["name"],
            "class_weights": trial["class_weights"],
            "model_path": str(trial_out),
            "acceptance_passed": acc_rc == 0,
            "acceptance": acc,
            "push_score": _score(acc),
        }
        results.append(r)
        if r["push_score"] > best_score:
            best_score = r["push_score"]
            best = r
        if acc_rc == 0:
            print(f"KN2 acceptance PASSED on {trial['name']}", flush=True)
            break

    summary = {"mode": "kn2_push_acceptance", "best_trial": best["name"] if best else None, "best_score": best_score, "trials": results}
    (report_dir / "kn2_push_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)

    if best and Path(best["model_path"]).is_file():
        shutil.copy2(best["model_path"], out)
        meta_src = Path(best["model_path"]).with_suffix(".meta.json")
        if meta_src.is_file():
            shutil.copy2(meta_src, out.with_suffix(".meta.json"))
        return 0 if best.get("acceptance_passed") else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
