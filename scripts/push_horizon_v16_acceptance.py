#!/usr/bin/env python3
"""Horizon V16 冲刺验收：降 gain 重标 + CE refine → ONNX → accept_horizon_v16。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch  # noqa: F401

from zhulong.agent.knowledge_net import train_knowledge_net
from zhulong.agent.training_utils import build_signed_labels, ensure_logs_dir, load_npz, signed_to_class
from zhulong.utils.device import print_gpu_status

TARGET_F1 = 0.45
EMBED_DIM = 32
HIDDEN_DIM = 96

# 全部 CE（focal_gamma=0），flat 权重 ≥ 1.0；针对降 gain 后 ~77% flat
PUSH_TRIALS = [
    {
        "name": "g017_warm",
        "class_weights": [2.4, 1.0, 2.4],
        "smote_ratio": 0.55,
        "lr": 0.0003,
        "patience": 18,
        "epochs": 80,
        "focal_gamma": 0.0,
        "temporal_val": False,
        "warm_start": True,
    },
    {
        "name": "g017_flat115",
        "class_weights": [2.3, 1.15, 2.3],
        "smote_ratio": 0.58,
        "lr": 0.00028,
        "patience": 20,
        "epochs": 90,
        "focal_gamma": 0.0,
        "temporal_val": False,
        "warm_start": True,
    },
    {
        "name": "g017_smote62",
        "class_weights": [2.2, 1.05, 2.2],
        "smote_ratio": 0.62,
        "lr": 0.00032,
        "patience": 22,
        "epochs": 100,
        "focal_gamma": 0.0,
        "temporal_val": False,
        "warm_start": True,
    },
    {
        "name": "g017_temporal",
        "class_weights": [2.5, 1.0, 2.5],
        "smote_ratio": 0.55,
        "lr": 0.00035,
        "patience": 20,
        "epochs": 90,
        "focal_gamma": 0.0,
        "temporal_val": True,
        "warm_start": True,
    },
    {
        "name": "g016_aggressive",
        "gain": 0.0016,
        "class_weights": [2.1, 1.0, 2.1],
        "smote_ratio": 0.60,
        "lr": 0.00035,
        "patience": 22,
        "epochs": 100,
        "focal_gamma": 0.0,
        "temporal_val": False,
        "warm_start": True,
    },
    {
        "name": "g017_scratch",
        "class_weights": [2.5, 1.0, 2.5],
        "smote_ratio": 0.55,
        "lr": 0.0005,
        "patience": 20,
        "epochs": 100,
        "focal_gamma": 0.0,
        "temporal_val": False,
        "warm_start": False,
    },
]


def _relabel(data: dict, gain: float, horizon: int) -> np.ndarray:
    close = np.asarray(data["close"], dtype=np.float64)
    return build_signed_labels(close, horizon=horizon, thr=gain)


def _label_stats(y: np.ndarray) -> dict:
    n = len(y)
    c = {int(k): int((y == k).sum()) for k in (0, 1, 2)}
    return {
        "rows": n,
        "short_pct": round(c[0] / n, 4),
        "flat_pct": round(c[1] / n, 4),
        "long_pct": round(c[2] / n, 4),
    }


def _run_trial(
    trial: dict,
    x: np.ndarray,
    y: np.ndarray,
    times,
    out_dir: Path,
    init_pth: Path,
    init_scaler: Path,
    *,
    val_ratio: float,
    train_end: str,
) -> dict:
    name = trial["name"]
    out_pth = out_dir / f"{name}.pth"
    out_scaler = out_dir / f"{name}_scaler.pkl"
    log_path = ensure_logs_dir() / f"horizon_push_{name}.log"
    print(f"\n=== PUSH {name} ===", flush=True)
    t0 = time.perf_counter()

    kw: dict = dict(
        val_ratio=val_ratio,
        epochs=int(trial["epochs"]),
        batch_size=512,
        lr=float(trial["lr"]),
        patience=int(trial["patience"]),
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
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
        train_end=train_end,
        focal_gamma=float(trial["focal_gamma"]),
    )
    if trial.get("warm_start") and init_pth.is_file():
        kw["init_from"] = init_pth
        if init_scaler.is_file():
            kw["init_scaler_from"] = init_scaler

    stats = train_knowledge_net(x, y, **kw)
    elapsed = time.perf_counter() - t0
    result = {
        "name": name,
        "macro_f1": float(stats.get("macro_f1", 0)),
        "val_accuracy": float(stats.get("val_accuracy", 0)),
        "model_path": str(out_pth),
        "scaler_path": str(out_scaler),
        "elapsed_sec": round(elapsed, 1),
        **{k: trial[k] for k in trial if k != "name"},
    }
    print(
        f"PUSH {name}: f1={result['macro_f1']:.4f} acc={result['val_accuracy']:.2%} "
        f"elapsed={result['elapsed_sec']}s",
        flush=True,
    )
    return result


def _export_onnx() -> int:
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "convert_knowledge_net_to_onnx.py"),
        "--model",
        "models/horizon_v16.pth",
        "--out",
        "models/horizon_v16.onnx",
        "--no-benchmark",
    ]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(_ROOT))


def _run_acceptance() -> int:
    cmd = [sys.executable, str(_ROOT / "scripts" / "accept_horizon_v16.py")]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/clean/training_horizon_v16.npz")
    parser.add_argument("--gain", type=float, default=0.0017, help="默认重标 gain")
    parser.add_argument("--init-model", default="models/horizon_v16.pth")
    parser.add_argument("--init-scaler", default="models/horizon_v16_scaler.pkl")
    parser.add_argument("--trials", default="", help="逗号分隔 trial 名")
    parser.add_argument("--target-f1", type=float, default=TARGET_F1)
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-accept", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-end", default="2024-12-31")
    args = parser.parse_args()

    init_pth = _ROOT / args.init_model
    init_scaler = _ROOT / args.init_scaler
    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}")
        return 1

    print_gpu_status()
    data = load_npz(npz_path)
    x = np.asarray(data["struct"], dtype=np.float32)
    horizon = int(data.get("horizon", [12])[0]) if "horizon" in data else 12
    times = data.get("time")

    baseline_f1 = 0.0
    meta_path = init_pth.with_suffix(".meta.json")
    if meta_path.is_file():
        baseline_f1 = float(json.loads(meta_path.read_text(encoding="utf-8")).get("macro_f1", 0))

    names = [t.strip() for t in args.trials.split(",") if t.strip()]
    trials = [t for t in PUSH_TRIALS if not names or t["name"] in names]

    report_dir = _ROOT / "data" / "training" / "reports" / "v16"
    trial_dir = _ROOT / "models" / "tune_trials" / "horizon_v16_push"
    report_dir.mkdir(parents=True, exist_ok=True)
    trial_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    best_f1 = baseline_f1
    best: dict | None = None
    current_gain = args.gain

    for trial in trials:
        gain = float(trial.get("gain", current_gain))
        labels = _relabel(data, gain, horizon)
        y = signed_to_class(labels)
        stats = _label_stats(y)
        print(f"\n--- gain={gain} labels {stats} ---", flush=True)

        try:
            r = _run_trial(
                trial,
                x,
                y,
                times,
                trial_dir,
                init_pth,
                init_scaler,
                val_ratio=args.val_ratio,
                train_end=args.train_end,
            )
            r["gain"] = gain
            r["label_stats"] = stats
            results.append(r)
            if r["macro_f1"] > best_f1:
                best_f1 = r["macro_f1"]
                best = r
            if r["macro_f1"] >= args.target_f1:
                print(f"Target F1 {args.target_f1} reached by {r['name']}", flush=True)
                break
        except Exception as ex:
            print(f"PUSH {trial['name']} FAILED: {ex}", flush=True)
            results.append({"name": trial["name"], "error": str(ex), "macro_f1": 0.0})

    summary = {
        "mode": "push_acceptance",
        "baseline_f1": baseline_f1,
        "default_gain": args.gain,
        "target_f1": args.target_f1,
        "best_trial": best["name"] if best else None,
        "best_macro_f1": best_f1,
        "best_val_accuracy": best.get("val_accuracy") if best else None,
        "training_passed": best_f1 >= args.target_f1,
        "trials": results,
    }
    report_path = report_dir / "horizon_push_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if not best or best_f1 <= baseline_f1:
        print(f"No improvement over baseline f1={baseline_f1:.4f}")
        return 2

    dest_pth = _ROOT / "models" / "horizon_v16.pth"
    dest_scaler = _ROOT / "models" / "horizon_v16_scaler.pkl"
    shutil.copy2(best["model_path"], dest_pth)
    shutil.copy2(best["scaler_path"], dest_scaler)
    meta = {
        "input_dim": int(x.shape[1]),
        "feature_dim_original": int(x.shape[1]),
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_res_blocks": 2,
        "scaler_path": str(best["scaler_path"]),
        "val_accuracy": best.get("val_accuracy"),
        "macro_f1": best_f1,
        "trial": best["name"],
        "gain_threshold": best.get("gain", args.gain),
        "label_stats": best.get("label_stats"),
        "passed": best_f1 >= args.target_f1,
        "push_from": str(init_pth),
    }
    dest_pth.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Best model → {dest_pth} (f1={best_f1:.4f})", flush=True)

    if best_f1 < args.target_f1:
        return 2

    if not args.skip_onnx:
        if _export_onnx() != 0:
            return 3

    if not args.skip_accept:
        acc_rc = _run_acceptance()
        summary["full_acceptance_passed"] = acc_rc == 0
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return acc_rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
