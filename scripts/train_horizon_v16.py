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

from zhulong.agent.horizon_location_labels import resolve_horizon_training_labels
from zhulong.agent.knowledge_net import train_knowledge_net
from zhulong.agent.training_utils import ensure_logs_dir, load_npz, resolve_v16_paths, signed_to_class, TRAIN_END_DEFAULT
from zhulong.utils.device import print_gpu_status

MIN_MACRO_F1 = 0.50


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD", "USOIL"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--npz",
        default="",
        help="训练 NPZ（默认按 --symbol 从 resolve_v16_paths 解析）",
    )
    parser.add_argument(
        "--label-mode",
        default="auto",
        choices=("auto", "legacy", "location"),
        help="auto=NPZ 含 loc_horizon_version 则用位置门控标签",
    )
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
    parser.add_argument(
        "--temporal-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开启：train<=train-end, val=之后（禁止随机 val 泄露）",
    )
    parser.add_argument("--train-end", default=TRAIN_END_DEFAULT)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-f1", type=float, default=0.0, help="exit 1 if macro_f1 below (0=ignore)")
    parser.add_argument("--out", default="", help="输出 .pth（默认按 symbol）")
    args = parser.parse_args()

    v16 = resolve_v16_paths(args.symbol)
    if not args.npz:
        args.npz = str(v16["horizon_location_npz"].relative_to(_ROOT)).replace("\\", "/")
    if not args.out:
        args.out = str(v16["horizon_pth"].relative_to(_ROOT)).replace("\\", "/")
    if args.symbol == "USOIL" and args.hidden_dim == 96:
        args.hidden_dim = int(v16["hidden_dim"])

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
    y_signed, label_version = resolve_horizon_training_labels(data, label_mode=args.label_mode)
    y = signed_to_class(y_signed)
    times = data.get("time")
    print(f"train samples: {len(x)} x {x.shape[1]} label_mode={args.label_mode} ({label_version})")
    if not args.temporal_val:
        print("ERROR: --no-temporal-val 已禁用（禁止随机验证泄露）。请使用 --temporal-val。")
        return 1

    out_model = _ROOT / args.out
    out_scaler = v16["horizon_scaler"]
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
        "temporal_val": True,
        "train_end": args.train_end,
        "pipeline_contract": "v16_no_leak_1",
        "epochs_requested": args.epochs,
        "retrain_tag": args.log_suffix or None,
        "label_mode": args.label_mode,
        "label_version": label_version,
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
