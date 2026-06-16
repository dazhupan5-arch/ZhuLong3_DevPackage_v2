#!/usr/bin/env python3
"""KnowledgeNet V15 教师蒸馏训练（76 维特征 + V15 XGBoost 软标签）。"""

from __future__ import annotations

import torch  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.knowledge_net import train_knowledge_net_v14_distill, v14_proba_to_kn
from zhulong.agent.training_utils import (
    ensure_logs_dir,
    load_npz,
    load_training_config,
    signed_to_class,
)
from zhulong.utils.device import print_gpu_status
from zhulong.v14_live import load_v15_bundle


def _v15_acceptance_passed(root: Path) -> bool:
    rep = root / "data" / "training" / "reports" / "v15" / "XAUUSD" / "train_report_v15.json"
    cfg = root / "models" / "XAUUSD" / "v15" / "config_v15.json"
    for p in (rep, cfg):
        if p.is_file():
            if json.loads(p.read_text(encoding="utf-8-sig")).get("passed"):
                return True
    return False


def compute_v15_teacher_probs(features: np.ndarray, bundle: dict, batch_size: int = 8192) -> np.ndarray:
    model = bundle["model"]
    feat_cols: list[str] = bundle["columns"]
    n_feat = len(feat_cols)
    if features.shape[1] < n_feat:
        raise ValueError(f"特征维 {features.shape[1]} < V15 需要 {n_feat}")
    x = features[:, :n_feat].astype(np.float32)
    probs_v15 = np.zeros((len(x), 3), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        probs_v15[start:end] = model.predict_proba(x[start:end])
    return v14_proba_to_kn(probs_v15)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--npz", default=None)
    parser.add_argument("--config", default="config_training.yaml")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--train-end", default="")
    args = parser.parse_args()

    cfg = load_training_config(_ROOT / args.config)
    print_gpu_status()
    kn_cfg = cfg.get("knowledge_net") or {}
    distill_cfg = kn_cfg.get("v15_distill") or kn_cfg.get("v14_distill") or {}
    dev_cfg = cfg.get("device") or {}
    rl_cfg = cfg.get("rl") or {}

    npz_path = Path(args.npz) if args.npz else _ROOT / "data" / "training_data_v15.npz"
    if not npz_path.is_file():
        print(f"缺少 {npz_path}，请先运行 prepare_knowledge_data_v15.py")
        return 1

    data = load_npz(npz_path)
    x = data["struct"]
    y = signed_to_class(data["labels"])
    times = data["time"]
    print(f"NPZ: {len(x)} rows x {x.shape[1]} dims")

    if not _v15_acceptance_passed(_ROOT):
        print("V15 验收未通过 — 请先运行 scripts/train_v15.py 且 passed=true")
        return 2

    print("加载 V15 教师模型...", flush=True)
    bundle = load_v15_bundle(args.symbol, root=_ROOT)
    cache_path = npz_path.with_name(npz_path.stem + "_v15_teacher.npy")
    if cache_path.is_file():
        teacher_probs = np.load(cache_path)
        if teacher_probs.shape[0] != len(x):
            teacher_probs = compute_v15_teacher_probs(x, bundle)
            np.save(cache_path, teacher_probs)
    else:
        teacher_probs = compute_v15_teacher_probs(x, bundle)
        np.save(cache_path, teacher_probs)
        print(f"软标签缓存: {cache_path}")

    out_model = _ROOT / "models" / "knowledge_net_v15.pth"
    out_scaler = _ROOT / "models" / "knowledge_scaler_v15.pkl"
    log_path = ensure_logs_dir() / f"knowledge_v15_{args.symbol.upper()}.log"
    train_end = args.train_end or distill_cfg.get("train_end") or "2024-12-31"

    stats = train_knowledge_net_v14_distill(
        x,
        y,
        teacher_probs,
        times,
        train_end=train_end,
        train_stride=int(distill_cfg.get("train_stride", 2)),
        epochs=int(args.epochs or distill_cfg.get("epochs", 80)),
        batch_size=int(distill_cfg.get("batch_size", 512)),
        lr=float(distill_cfg.get("lr", 0.0005)),
        patience=int(distill_cfg.get("patience", 20)),
        hidden_dim=int(distill_cfg.get("hidden_dim", 128)),
        embed_dim=int(distill_cfg.get("embed_dim", 32)),
        out_path=out_model,
        scaler_path=out_scaler,
        log_path=log_path,
        distill_weight=float(distill_cfg.get("distill_weight", 0.95)),
        temperature=float(distill_cfg.get("temperature", 2.0)),
        class_weights=list(distill_cfg.get("class_weights") or [2.0, 1.0, 2.5]),
        device=str(dev_cfg.get("torch", "auto")),
        num_res_blocks=int(distill_cfg.get("num_res_blocks", 3)),
        min_v14_agreement=float(distill_cfg.get("min_v15_agreement", 0.68)),
        min_trade_precision=float(distill_cfg.get("min_trade_precision", 0.58)),
    )

    print("=== V15 蒸馏验收 ===")
    print(f"教师一致率: {stats['v14_agreement']:.2%}")
    print(f"交易一致率: {stats['trade_precision']:.2%}")
    print(f"模型: {stats['model_path']}")
    ok = stats.get("passed_acc", False) and stats.get("passed_size", False)
    print("结果:", "PASS" if ok else "FAIL — 未达 KN 蒸馏指标")
    if ok:
        print("下一步: py -3 scripts/convert_knowledge_net_to_onnx.py --model models/knowledge_net_v15.pth --out models/knowledge_net_v15.onnx --no-benchmark")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
