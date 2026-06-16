#!/usr/bin/env python3
"""KnowledgeNet V14 教师蒸馏训练（68 维特征 + 时间切分）。"""

from __future__ import annotations

import torch  # noqa: F401 — Windows 下须最先加载

import argparse
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
    resolve_symbol_paths,
    signed_to_class,
)
from zhulong.utils.device import print_gpu_status
from zhulong.v14_live import load_v14_bundle


def compute_v14_teacher_probs(features: np.ndarray, bundle: dict, batch_size: int = 8192) -> np.ndarray:
    """批量 V14 predict_proba，输出 KN 格式软标签 (N,3)。"""
    model = bundle["model"]
    feat_cols: list[str] = bundle["columns"]
    n_feat = len(feat_cols)
    if features.shape[1] < n_feat:
        raise ValueError(f"特征维 {features.shape[1]} < V14 需要 {n_feat}")
    x = features[:, :n_feat].astype(np.float32)
    probs_v14 = np.zeros((len(x), 3), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        probs_v14[start:end] = model.predict_proba(x[start:end])
    return v14_proba_to_kn(probs_v14)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD", "USOIL"])
    parser.add_argument("--npz", default=None)
    parser.add_argument("--config", default="config_training.yaml")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--distill-weight", type=float, default=-1.0)
    parser.add_argument("--train-end", default="")
    parser.add_argument("--train-stride", type=int, default=0)
    args = parser.parse_args()

    cfg = load_training_config(_ROOT / args.config)
    print_gpu_status()
    paths = resolve_symbol_paths(args.symbol, cfg)
    npz_path = Path(args.npz) if args.npz else paths["npz"]
    if not npz_path.is_file():
        print(f"缺少训练包 {npz_path}，请先运行 prepare_knowledge_data.py")
        return 1

    data = load_npz(npz_path)
    if "time" not in data:
        print("NPZ 缺少 time 列，请用 prepare_knowledge_data.py 重新生成")
        return 1

    x = data["struct"]
    y = signed_to_class(data["labels"])
    times = data["time"]

    kn_cfg = cfg.get("knowledge_net") or {}
    distill_cfg = kn_cfg.get("v14_distill") or {}
    dev_cfg = cfg.get("device") or {}
    rl_cfg = cfg.get("rl") or {}

    print("加载 V14 教师模型...", flush=True)
    bundle = load_v14_bundle(args.symbol, model_subdir="v14", root=_ROOT)
    cache_path = npz_path.with_name(npz_path.stem + "_v14_teacher.npy")
    if cache_path.is_file():
        print(f"加载缓存软标签: {cache_path}", flush=True)
        teacher_probs = np.load(cache_path)
        if teacher_probs.shape[0] != len(x):
            print("缓存行数不匹配，重新生成...")
            teacher_probs = compute_v14_teacher_probs(x, bundle)
            np.save(cache_path, teacher_probs)
    else:
        print(f"生成 V14 软标签 ({len(x)} 样本)...", flush=True)
        teacher_probs = compute_v14_teacher_probs(x, bundle)
        np.save(cache_path, teacher_probs)
        print(f"软标签已缓存: {cache_path}", flush=True)

    out_model = paths["knowledge_model"]
    out_scaler = paths["knowledge_scaler"]
    log_path = ensure_logs_dir() / f"knowledge_v14_{args.symbol.upper()}.log"

    train_end = args.train_end or distill_cfg.get("train_end") or str(rl_cfg.get("train_through_year", 2024)) + "-12-31"

    stats = train_knowledge_net_v14_distill(
        x,
        y,
        teacher_probs,
        times,
        train_end=train_end,
        train_stride=int(
            args.train_stride if args.train_stride > 0 else distill_cfg.get("train_stride", 4)
        ),
        epochs=int(args.epochs if args.epochs > 0 else distill_cfg.get("epochs", kn_cfg.get("epochs", 80))),
        batch_size=int(distill_cfg.get("batch_size", kn_cfg.get("batch_size", 512))),
        lr=float(args.lr if args.lr > 0 else distill_cfg.get("lr", kn_cfg.get("lr", 0.0005))),
        patience=int(args.patience if args.patience > 0 else distill_cfg.get("patience", kn_cfg.get("patience", 20))),
        hidden_dim=int(
            args.hidden_dim if args.hidden_dim > 0 else distill_cfg.get("hidden_dim", kn_cfg.get("hidden_dim", 128))
        ),
        embed_dim=int(distill_cfg.get("embed_dim", kn_cfg.get("embed_dim", 32))),
        out_path=out_model,
        scaler_path=out_scaler,
        log_path=log_path,
        distill_weight=float(
            args.distill_weight if args.distill_weight >= 0 else distill_cfg.get("distill_weight", 0.7)
        ),
        temperature=float(distill_cfg.get("temperature", 2.0)),
        class_weights=list(distill_cfg.get("class_weights") or kn_cfg.get("class_weights") or [2.0, 1.0, 2.0]),
        device=str(dev_cfg.get("torch", "auto")),
        num_res_blocks=int(distill_cfg.get("num_res_blocks", kn_cfg.get("num_res_blocks", 3))),
        min_v14_agreement=float(distill_cfg.get("min_v14_agreement", 0.70)),
        min_trade_precision=float(distill_cfg.get("min_trade_precision", 0.55)),
    )

    min_agreement = float(distill_cfg.get("min_v14_agreement", 0.70))
    min_trade = float(distill_cfg.get("min_trade_precision", 0.55))
    print("=== V14 蒸馏验收 ===")
    print(f"V14 一致率: {stats['v14_agreement']:.2%} (目标 ≥{min_agreement:.0%})")
    print(f"交易一致率: {stats['trade_precision']:.2%} (目标 ≥{min_trade:.0%})")
    print(f"宏平均 F1: {stats['macro_f1']:.4f}")
    print(f"交易占比: {stats['trade_rate']:.2%}")
    print(f"模型: {stats['model_path']}")
    print(f"Scaler: {stats['scaler_path']}")
    ok = stats["passed_acc"] and stats["passed_size"]
    print("结果:", "PASS" if ok else "FAIL — 未达验收标准，禁止启用智能体")
    if ok:
        print("提示: 运行 py -3 scripts/convert_knowledge_net_to_onnx.py 导出 ONNX 加速推理")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
