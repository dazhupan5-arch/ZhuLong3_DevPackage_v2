#!/usr/bin/env python3
"""Horizon V16 验证集阈值校准：在 argmax 基础上搜 flat_penalty / margin 以最大化 macro F1。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score

from zhulong.agent.knowledge_net import _knowledge_net_class, _time_split_mask
from zhulong.agent.training_utils import load_npz, signed_to_class


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _predict_adj(probs: np.ndarray, flat_scale: float, dir_margin: float) -> np.ndarray:
    """flat_scale<1 压低 flat；dir_margin 要求方向概率超过 flat+margin 才交易。"""
    p = probs.copy()
    p[:, 1] *= flat_scale
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-9)
    pred = np.full(len(p), 1, dtype=np.int64)
    for i in range(len(p)):
        ps, pf, pl = p[i]
        best_dir = 0 if ps >= pl else 2
        p_dir = max(ps, pl)
        if p_dir >= pf + dir_margin:
            pred[i] = best_dir
        else:
            pred[i] = 1
    return pred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/horizon_v16.pth")
    parser.add_argument("--scaler", default="models/horizon_v16_scaler.pkl")
    parser.add_argument("--npz", default="data/clean/training_horizon_v16.npz")
    parser.add_argument("--target-f1", type=float, default=0.45)
    parser.add_argument("--temporal-val", action="store_true")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--apply", action="store_true", help="写入 meta 校准参数")
    args = parser.parse_args()

    model_path = _ROOT / args.model
    scaler_path = _ROOT / args.scaler
    npz_path = _ROOT / args.npz
    meta_path = model_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    data = load_npz(npz_path)
    x = np.asarray(data["struct"], dtype=np.float32)
    y = signed_to_class(data["labels"])
    times = data.get("time")

    if args.temporal_val and times is not None:
        train_mask, val_mask = _time_split_mask(np.asarray(times), args.train_end)
    else:
        n = len(x)
        split = int(n * (1.0 - args.val_ratio))
        val_mask = np.zeros(n, dtype=bool)
        val_mask[split:] = True
        train_mask = ~val_mask

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

    base_pred = probs.argmax(axis=1)
    base_f1 = _macro_f1(y_va, base_pred)
    print(f"baseline argmax f1={base_f1:.4f} n_val={len(y_va)}")

    best_f1 = base_f1
    best = {"flat_scale": 1.0, "dir_margin": 0.0, "macro_f1": base_f1}

    for flat_scale in np.arange(0.75, 1.05, 0.05):
        for dir_margin in np.arange(0.0, 0.12, 0.01):
            pred = _predict_adj(probs, float(flat_scale), float(dir_margin))
            f1 = _macro_f1(y_va, pred)
            if f1 > best_f1:
                best_f1 = f1
                best = {
                    "flat_scale": round(float(flat_scale), 3),
                    "dir_margin": round(float(dir_margin), 3),
                    "macro_f1": round(f1, 6),
                }

    print(json.dumps({"baseline_f1": base_f1, "best": best, "target_f1": args.target_f1}, indent=2))

    if args.apply and best_f1 >= base_f1:
        meta["calibration"] = best
        meta["macro_f1"] = best_f1
        meta["macro_f1_raw_argmax"] = base_f1
        meta["passed"] = best_f1 >= args.target_f1
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Updated {meta_path}")

    return 0 if best_f1 >= args.target_f1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
