#!/usr/bin/env python3
"""从 F1≈0.41 的 near-threshold checkpoint 做小步 refine（禁止 focal 压 flat）。"""

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
BASELINE_F1 = 0.410272276643247

# 0.41 模型 recipe（train_horizon_v16.py 默认 + backup log 验证）
BASELINE_RECIPE = {
    "class_weights": [2.5, 1.0, 2.5],
    "smote_ratio": 0.5,
    "lr": 0.0005,
    "patience": 15,
    "epochs": 80,
    "hidden_dim": 96,
    "embed_dim": 32,
    "focal_gamma": 0.0,
    "temporal_val": False,
    "val_ratio": 0.15,
}

# 在 baseline 附近小步搜索；全部 CE（focal_gamma=0），flat 权重 ≥ 1.0
REFINE_TRIALS = [
    {
        "name": "warm_same",
        "class_weights": [2.5, 1.0, 2.5],
        "smote_ratio": 0.5,
        "lr": 0.0002,
        "patience": 12,
        "epochs": 50,
        "hidden_dim": 96,
        "focal_gamma": 0.0,
        "temporal_val": False,
        "warm_start": True,
    },
    {
        "name": "flat_boost",
        "class_weights": [2.4, 1.1, 2.4],
        "smote_ratio": 0.52,
        "lr": 0.00025,
        "patience": 14,
        "epochs": 60,
        "hidden_dim": 96,
        "focal_gamma": 0.0,
        "temporal_val": False,
        "warm_start": True,
    },
    {
        "name": "dir_mild",
        "class_weights": [2.55, 1.0, 2.55],
        "smote_ratio": 0.53,
        "lr": 0.0003,
        "patience": 14,
        "epochs": 60,
        "hidden_dim": 96,
        "focal_gamma": 0.0,
        "temporal_val": False,
        "warm_start": True,
    },
    {
        "name": "smote_055",
        "class_weights": [2.5, 1.05, 2.5],
        "smote_ratio": 0.55,
        "lr": 0.00025,
        "patience": 15,
        "epochs": 70,
        "hidden_dim": 96,
        "focal_gamma": 0.0,
        "temporal_val": False,
        "warm_start": True,
    },
    {
        "name": "longer_ce",
        "class_weights": [2.5, 1.0, 2.5],
        "smote_ratio": 0.5,
        "lr": 0.00035,
        "patience": 20,
        "epochs": 100,
        "hidden_dim": 96,
        "focal_gamma": 0.0,
        "temporal_val": False,
        "warm_start": True,
    },
]


def _print_analysis() -> None:
    print(
        """
=== 0.41 baseline 分析 ===
标签分布: short 8.8% | flat 81.7% | long 9.4%  (gain=0.002, horizon=12)
最佳 epoch 47  per-class recall: flat≈0.46 short≈0.67 long≈0.64  → macro F1=0.410

失败 focal 调参共性: flat recall 被压到 0.02~0.09（几乎不预测 flat）
  → macro F1 暴跌至 0.17~0.22（三类 F1 平均被 flat 拖垮）

结论: 在 82% flat 标签下，必须保持 flat 权重≥1.0 + CE，禁止 focal/过低 flat 权重。
refine 策略: 从 horizon_v16.pth warm-start，小步调 cw/smote/lr。
"""
    )


def _run_trial(
    trial: dict,
    x,
    y,
    times,
    out_dir: Path,
    init_pth: Path,
    init_scaler: Path,
) -> dict:
    name = trial["name"]
    out_pth = out_dir / f"{name}.pth"
    out_scaler = out_dir / f"{name}_scaler.pkl"
    log_path = ensure_logs_dir() / f"horizon_refine_{name}.log"
    print(f"\n=== REFINE {name} ===", flush=True)
    t0 = time.perf_counter()

    kw: dict = dict(
        val_ratio=BASELINE_RECIPE["val_ratio"],
        epochs=int(trial["epochs"]),
        batch_size=512,
        lr=float(trial["lr"]),
        patience=int(trial["patience"]),
        hidden_dim=int(trial["hidden_dim"]),
        embed_dim=BASELINE_RECIPE["embed_dim"],
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
        **trial,
    }
    meta_path = out_pth.with_suffix(".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["trial"] = name
        meta["refine_from"] = str(init_pth)
        meta["passed"] = result["macro_f1"] >= TARGET_F1
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"REFINE {name}: f1={result['macro_f1']:.4f} acc={result['val_accuracy']:.2%} "
        f"elapsed={result['elapsed_sec']}s",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/clean/training_horizon_v16.npz")
    parser.add_argument("--init-model", default="models/horizon_v16.pth")
    parser.add_argument("--init-scaler", default="models/horizon_v16_scaler.pkl")
    parser.add_argument("--trials", default="", help="comma names subset")
    parser.add_argument("--target-f1", type=float, default=TARGET_F1)
    args = parser.parse_args()

    _print_analysis()

    init_pth = _ROOT / args.init_model
    init_scaler = _ROOT / args.init_scaler
    if not init_pth.is_file():
        print(f"缺少 baseline 权重 {init_pth}，请先 restore_v16_backup.ps1")
        return 1

    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}")
        return 1

    print_gpu_status()
    data = load_npz(npz_path)
    x = data["struct"]
    y = signed_to_class(data["labels"])
    times = data.get("time")
    print(f"samples={len(x):,} dim={x.shape[1]} baseline_f1={BASELINE_F1:.4f} gap={args.target_f1 - BASELINE_F1:.4f}")

    trial_dir = _ROOT / "models" / "tune_trials" / "horizon_v16_refine"
    trial_dir.mkdir(parents=True, exist_ok=True)
    report_dir = _ROOT / "data" / "training" / "reports" / "v16"
    report_dir.mkdir(parents=True, exist_ok=True)

    names = [t.strip() for t in args.trials.split(",") if t.strip()]
    trials = [t for t in REFINE_TRIALS if not names or t["name"] in names]

    results: list[dict] = []
    best_f1 = BASELINE_F1
    best: dict | None = None
    for trial in trials:
        try:
            r = _run_trial(trial, x, y, times, trial_dir, init_pth, init_scaler)
            results.append(r)
            if r["macro_f1"] > best_f1:
                best_f1 = r["macro_f1"]
                best = r
            if r["macro_f1"] >= args.target_f1:
                print(f"Target F1 reached by {r['name']}", flush=True)
                break
        except Exception as ex:
            print(f"REFINE {trial['name']} FAILED: {ex}", flush=True)
            results.append({"name": trial["name"], "error": str(ex), "macro_f1": 0.0})

    summary = {
        "mode": "refine_from_baseline",
        "baseline_f1": BASELINE_F1,
        "baseline_recipe": BASELINE_RECIPE,
        "target_f1": args.target_f1,
        "best_trial": best["name"] if best else "baseline_unchanged",
        "best_macro_f1": best_f1,
        "best_val_accuracy": best.get("val_accuracy") if best else None,
        "passed": best_f1 >= args.target_f1,
        "trials": results,
    }
    report_path = report_dir / "horizon_refine_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if best and best["macro_f1"] > BASELINE_F1:
        dest_pth = _ROOT / "models" / "horizon_v16.pth"
        dest_scaler = _ROOT / "models" / "horizon_v16_scaler.pkl"
        shutil.copy2(best["model_path"], dest_pth)
        shutil.copy2(best["scaler_path"], dest_scaler)
        meta_src = Path(best["model_path"]).with_suffix(".meta.json")
        if meta_src.is_file():
            shutil.copy2(meta_src, dest_pth.with_suffix(".meta.json"))
        print(f"Improved model copied to {dest_pth} (f1 {best['macro_f1']:.4f} > {BASELINE_F1:.4f})")
    else:
        print(f"No trial beat baseline {BASELINE_F1:.4f}; keeping current horizon_v16.pth")

    return 0 if best_f1 >= args.target_f1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
