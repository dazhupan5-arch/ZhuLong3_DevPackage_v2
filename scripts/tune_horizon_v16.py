#!/usr/bin/env python3
"""Horizon V16 多组超参试训，选 macro_f1 最高者写入 models/horizon_v16.pth。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: F401

from zhulong.agent.knowledge_net import train_knowledge_net
from zhulong.agent.training_utils import ensure_logs_dir, load_npz, signed_to_class
from zhulong.utils.device import print_gpu_status

TARGET_F1 = 0.45

TRIALS = [
    {
        "name": "refine_baseline",
        "class_weights": [2.5, 0.9, 2.5],
        "smote_ratio": 0.5,
        "lr": 0.0005,
        "patience": 18,
        "epochs": 100,
        "hidden_dim": 96,
        "focal_gamma": 0.0,
        "temporal_val": False,
    },
    {
        "name": "focal_mild",
        "class_weights": [2.6, 0.75, 2.6],
        "smote_ratio": 0.55,
        "lr": 0.00045,
        "patience": 20,
        "epochs": 100,
        "hidden_dim": 96,
        "focal_gamma": 1.5,
        "temporal_val": False,
    },
    {
        "name": "focal_wide128",
        "class_weights": [2.8, 0.65, 2.8],
        "smote_ratio": 0.58,
        "lr": 0.0004,
        "patience": 22,
        "epochs": 120,
        "hidden_dim": 128,
        "focal_gamma": 2.0,
        "temporal_val": False,
    },
    {
        "name": "focal_low_flat",
        "class_weights": [3.0, 0.55, 3.0],
        "smote_ratio": 0.62,
        "lr": 0.0004,
        "patience": 20,
        "epochs": 110,
        "hidden_dim": 128,
        "focal_gamma": 1.8,
        "temporal_val": False,
    },
    {
        "name": "temporal_focal",
        "class_weights": [2.7, 0.7, 2.7],
        "smote_ratio": 0.55,
        "lr": 0.00042,
        "patience": 20,
        "epochs": 100,
        "hidden_dim": 128,
        "focal_gamma": 1.6,
        "temporal_val": True,
    },
]


def _run_trial(
    trial: dict,
    x,
    y,
    times,
    out_dir: Path,
) -> dict:
    name = trial["name"]
    out_pth = out_dir / f"{name}.pth"
    out_scaler = out_dir / f"{name}_scaler.pkl"
    log_path = ensure_logs_dir() / f"horizon_tune_{name}.log"
    print(f"\n=== TRIAL {name} ===", flush=True)
    t0 = time.perf_counter()
    stats = train_knowledge_net(
        x,
        y,
        val_ratio=0.15,
        epochs=int(trial["epochs"]),
        batch_size=512,
        lr=float(trial["lr"]),
        patience=int(trial["patience"]),
        hidden_dim=int(trial["hidden_dim"]),
        embed_dim=32,
        out_path=out_pth,
        scaler_path=out_scaler,
        shuffle_train=not trial["temporal_val"],
        log_path=log_path,
        use_smote=True,
        smote_ratio=float(trial["smote_ratio"]),
        class_weights=list(trial["class_weights"]),
        device="auto",
        num_res_blocks=2,
        select_by="f1",
        times=times if trial["temporal_val"] else None,
        train_end="2024-12-31",
        focal_gamma=float(trial["focal_gamma"]),
    )
    elapsed = time.perf_counter() - t0
    result = {
        "name": name,
        "macro_f1": float(stats.get("macro_f1", 0)),
        "val_accuracy": float(stats.get("val_accuracy", 0)),
        "model_path": str(out_pth),
        "scaler_path": str(out_scaler),
        "elapsed_sec": round(elapsed, 1),
        **trial,
    }
    meta_path = out_pth.with_suffix(".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["trial"] = name
        meta["passed"] = result["macro_f1"] >= TARGET_F1
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"TRIAL {name}: f1={result['macro_f1']:.4f} acc={result['val_accuracy']:.2%} "
        f"elapsed={result['elapsed_sec']}s",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/training_horizon_v16.npz")
    parser.add_argument("--trials", default="", help="comma names to run subset")
    parser.add_argument("--target-f1", type=float, default=TARGET_F1)
    args = parser.parse_args()

    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}")
        return 1

    print_gpu_status()
    data = load_npz(npz_path)
    x = data["struct"]
    y = signed_to_class(data["labels"])
    times = data.get("time")
    print(f"samples={len(x):,} dim={x.shape[1]}")

    trial_dir = _ROOT / "models" / "tune_trials" / "horizon_v16"
    trial_dir.mkdir(parents=True, exist_ok=True)
    report_dir = _ROOT / "data" / "training" / "reports" / "v16"
    report_dir.mkdir(parents=True, exist_ok=True)

    names = [t.strip() for t in args.trials.split(",") if t.strip()]
    trials = [t for t in TRIALS if not names or t["name"] in names]

    results: list[dict] = []
    best: dict | None = None
    for trial in trials:
        try:
            r = _run_trial(trial, x, y, times, trial_dir)
            results.append(r)
            if best is None or r["macro_f1"] > best["macro_f1"]:
                best = r
            if r["macro_f1"] >= args.target_f1:
                print(f"Target F1 reached by {r['name']}, stopping early.", flush=True)
                break
        except Exception as ex:
            print(f"TRIAL {trial['name']} FAILED: {ex}", flush=True)
            results.append({"name": trial["name"], "error": str(ex), "macro_f1": 0.0})

    if not best:
        print("No successful trials.")
        return 1

    dest_pth = _ROOT / "models" / "horizon_v16.pth"
    dest_scaler = _ROOT / "models" / "horizon_v16_scaler.pkl"
    existing_f1 = 0.0
    meta_dest = dest_pth.with_suffix(".meta.json")
    if meta_dest.is_file():
        try:
            existing_f1 = float(json.loads(meta_dest.read_text(encoding="utf-8")).get("macro_f1", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            existing_f1 = 0.0

    src_pth = Path(best["model_path"])
    src_scaler = Path(best["scaler_path"])
    if best["macro_f1"] > existing_f1:
        shutil.copy2(src_pth, dest_pth)
        shutil.copy2(src_scaler, dest_scaler)
        meta_src = src_pth.with_suffix(".meta.json")
        if meta_src.is_file():
            shutil.copy2(meta_src, meta_dest)
        print(f"Best model copied to {dest_pth} (f1 {best['macro_f1']:.4f} > {existing_f1:.4f})")
    else:
        print(
            f"Best trial f1={best['macro_f1']:.4f} <= current {existing_f1:.4f}; "
            f"keeping {dest_pth} (restore backup if needed).",
            flush=True,
        )

    summary = {
        "target_f1": args.target_f1,
        "best_trial": best["name"],
        "best_macro_f1": best["macro_f1"],
        "best_val_accuracy": best.get("val_accuracy"),
        "passed": best["macro_f1"] >= args.target_f1,
        "trials": results,
    }
    report_path = report_dir / "horizon_tune_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if best["macro_f1"] >= args.target_f1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
