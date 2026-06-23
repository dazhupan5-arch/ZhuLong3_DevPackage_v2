#!/usr/bin/env python3
"""Horizon V16 验证集阈值校准：搜 flat_scale / dir_margin / min_confidence，与实机推理一致。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score

from zhulong.agent.horizon_location_labels import resolve_horizon_training_labels
from zhulong.agent.horizon_predictor import horizon_probs_to_classes
from zhulong.agent.knowledge_net import _knowledge_net_class
from zhulong.agent.training_utils import (
    TRAIN_END_DEFAULT,
    VAL_YEAR_DEFAULT,
    load_npz,
    signed_to_class,
    temporal_train_val_masks,
)

MIN_MACRO_F1 = 0.50


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _default_min_confidence(root: Path) -> float:
    cfg_path = root / "config" / "config_agent.json"
    if not cfg_path.is_file():
        return 0.42
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        hp = (cfg.get("architecture") or {}).get("horizon_predictor") or {}
        return float(hp.get("min_direction_confidence", 0.42))
    except Exception:
        return 0.42


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/horizon_v16.pth")
    parser.add_argument("--scaler", default="models/horizon_v16_scaler.pkl")
    parser.add_argument("--npz", default="data/clean/training_horizon_v16_location.npz")
    parser.add_argument("--target-f1", type=float, default=MIN_MACRO_F1)
    parser.add_argument(
        "--label-mode",
        default="location",
        choices=("auto", "legacy", "location"),
        help="须与 train_horizon_v16.py 一致（retrain 默认 location）",
    )
    parser.add_argument(
        "--temporal-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认：val=val_year 全年 OOS（与 accept_horizon_v16 测试集一致）",
    )
    parser.add_argument("--train-end", default=TRAIN_END_DEFAULT)
    parser.add_argument("--val-year", type=int, default=VAL_YEAR_DEFAULT)
    parser.add_argument("--apply", action="store_true", help="写入 meta 校准参数（不修改 passed）")
    args = parser.parse_args()

    model_path = _ROOT / args.model
    scaler_path = _ROOT / args.scaler
    npz_path = _ROOT / args.npz
    meta_path = model_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig")) if meta_path.is_file() else {}

    data = load_npz(npz_path)
    x = np.asarray(data["struct"], dtype=np.float32)
    y_signed, label_version = resolve_horizon_training_labels(data, label_mode=args.label_mode)
    y = signed_to_class(y_signed)
    times = data.get("time")

    if not args.temporal_val:
        print("ERROR: --no-temporal-val 已禁用")
        return 1
    if times is None:
        print("ERROR: --temporal-val 需要 NPZ 含 time 列")
        return 1

    _, val_mask = temporal_train_val_masks(
        np.asarray(times),
        train_end=args.train_end,
        val_year=int(args.val_year),
    )
    if int(val_mask.sum()) < 500:
        print(f"ERROR: OOS val 样本不足 n={int(val_mask.sum())} val_year={args.val_year}")
        return 1

    x_va = x[val_mask]
    y_va = y[val_mask]
    scaler = joblib.load(scaler_path)
    x_va = scaler.transform(x_va).astype(np.float32)

    hidden = int(meta.get("hidden_dim", 96))
    embed = int(meta.get("embed_dim", 32))
    blocks = int(meta.get("num_res_blocks", 2))
    inp = int(meta.get("input_dim", x.shape[1]))

    KnCls, _ = _knowledge_net_class(num_res_blocks=blocks)
    device = torch.device("cpu")
    model = KnCls(inp, hidden_dim=hidden, embed_dim=embed, num_res_blocks=blocks).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        logits, _, _ = model(torch.tensor(x_va, device=device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    base_conf = _default_min_confidence(_ROOT)
    base_pred = horizon_probs_to_classes(probs, min_confidence=base_conf)
    base_f1 = _macro_f1(y_va, base_pred)
    print(
        f"baseline runtime-rule f1={base_f1:.4f} n_val={len(y_va)} "
        f"val_year={args.val_year} label={label_version}"
    )

    best_f1 = base_f1
    best = {
        "flat_scale": 1.0,
        "dir_margin": 0.0,
        "min_confidence": round(base_conf, 3),
        "macro_f1": round(base_f1, 6),
    }

    for min_conf in np.arange(0.36, 0.54, 0.02):
        for flat_scale in np.arange(0.75, 1.20, 0.05):
            for dir_margin in np.arange(0.0, 0.12, 0.01):
                pred = horizon_probs_to_classes(
                    probs,
                    min_confidence=float(min_conf),
                    flat_scale=float(flat_scale),
                    dir_margin=float(dir_margin),
                )
                f1 = _macro_f1(y_va, pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best = {
                        "flat_scale": round(float(flat_scale), 3),
                        "dir_margin": round(float(dir_margin), 3),
                        "min_confidence": round(float(min_conf), 3),
                        "macro_f1": round(f1, 6),
                    }

    print(json.dumps({"baseline_f1": base_f1, "best": best, "target_f1": args.target_f1}, indent=2))

    if args.apply and best_f1 >= base_f1:
        meta["calibration"] = best
        meta["macro_f1"] = best_f1
        meta["macro_f1_raw_argmax"] = float(_macro_f1(y_va, probs.argmax(axis=1)))
        meta["calibration_val_year"] = int(args.val_year)
        meta["label_mode"] = args.label_mode
        meta["label_version"] = label_version
        meta["passed"] = False
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Updated {meta_path} (passed=false until accept_horizon_v16 --apply)")

    ok = best_f1 >= args.target_f1
    print("PASS" if ok else f"FAIL: macro_f1 {best_f1:.4f} < target {args.target_f1}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
