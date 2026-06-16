#!/usr/bin/env python3
"""训练 V16 HorizonPredictor（Structure 30 维 → 1h 三分类）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: F401

from zhulong.agent.knowledge_net import train_knowledge_net
from zhulong.agent.training_utils import ensure_logs_dir, load_npz, signed_to_class
from zhulong.utils.device import print_gpu_status

MIN_MACRO_F1 = 0.45


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--npz", default="data/clean/training_horizon_v16.npz")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--smote-ratio", type=float, default=0.5)
    parser.add_argument(
        "--class-weights",
        default="2.5,1.0,2.5",
        help="short,flat,long CrossEntropy weights",
    )
    parser.add_argument("--log-suffix", default="", help="append to log filename, e.g. retrain1")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="0=CE, 1.5~2.0 recommended")
    parser.add_argument("--temporal-val", action="store_true", help="train<=2024, val=2025")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-f1", type=float, default=0.0, help="exit 1 if macro_f1 below (0=ignore)")
    parser.add_argument("--out", default="models/horizon_v16.pth")
    args = parser.parse_args()

    try:
        class_weights = [float(x.strip()) for x in args.class_weights.split(",")]
        if len(class_weights) != 3:
            raise ValueError
    except ValueError:
        print("Invalid --class-weights, expected three comma-separated floats")
        return 1

    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}，请先运行 prepare_horizon_v16_data.py")
        return 1

    print_gpu_status()
    data = load_npz(npz_path)
    x = data["struct"]
    y = signed_to_class(data["labels"])
    times = data.get("time")
    print(f"train samples: {len(x)} x {x.shape[1]}")

    out_model = _ROOT / args.out
    out_scaler = out_model.with_name("horizon_v16_scaler.pkl")
    log_name = "horizon_v16_train.log"
    if args.log_suffix:
        log_name = f"horizon_v16_train_{args.log_suffix}.log"
    log_path = ensure_logs_dir() / log_name

    stats = train_knowledge_net(
        x,
        y,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=512,
        lr=args.lr,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        out_path=out_model,
        scaler_path=out_scaler,
        shuffle_train=not args.temporal_val,
        log_path=log_path,
        use_smote=True,
        smote_ratio=args.smote_ratio,
        class_weights=class_weights,
        device="auto",
        num_res_blocks=2,
        select_by="f1",
        times=times if args.temporal_val else None,
        train_end=args.train_end,
        focal_gamma=args.focal_gamma,
    )

    meta = {
        "input_dim": int(x.shape[1]),
        "model_id": "horizon_v16",
        "horizon_bars": int(data.get("horizon", [12])[0]) if "horizon" in data else 12,
        "gain_threshold": float(data.get("gain", [0.002])[0]) if "gain" in data else 0.002,
        "val_accuracy": stats.get("val_accuracy", 0),
        "macro_f1": stats.get("macro_f1", 0),
        "training_rows": int(len(x)),
        "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "num_res_blocks": 2,
        "class_weights": class_weights,
        "smote_ratio": args.smote_ratio,
        "lr": args.lr,
        "patience": args.patience,
        "focal_gamma": args.focal_gamma,
        "temporal_val": args.temporal_val,
        "epochs_requested": args.epochs,
        "retrain_tag": args.log_suffix or None,
        "passed": bool(stats.get("macro_f1", 0) >= MIN_MACRO_F1),
    }
    out_model.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    f1 = float(stats.get("macro_f1", 0))
    acc = float(stats.get("val_accuracy", 0))
    ok = f1 >= MIN_MACRO_F1
    print("=== Horizon V16 ===")
    print(f"acc={acc:.2%} f1={f1:.4f} (need>={MIN_MACRO_F1})")
    print(f"model={out_model}")
    print("PASS" if ok else "FAIL (model saved anyway for backtest)")

    min_f1 = float(args.min_f1)
    if min_f1 > 0 and f1 < min_f1:
        print(f"exit: macro_f1 {f1:.4f} < --min-f1 {min_f1}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
