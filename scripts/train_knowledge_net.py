#!/usr/bin/env python3
"""训练 KnowledgeNet（监督学习）。"""

from __future__ import annotations

# Windows: torch 必须早于 sklearn/zhulong 导入，否则 c10.dll 可能初始化失败
import torch  # noqa: F401

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.knowledge_net import train_knowledge_net
from zhulong.agent.training_utils import (
    ensure_logs_dir,
    load_npz,
    load_training_config,
    resolve_symbol_paths,
    signed_to_class,
)
from zhulong.utils.device import print_gpu_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD", "USOIL"])
    parser.add_argument("--npz", default=None)
    parser.add_argument("--config", default="config_training.yaml")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--smote-ratio", type=float, default=-1.0)
    parser.add_argument("--val-ratio", type=float, default=-1.0)
    parser.add_argument("--num-res-blocks", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--select-by", choices=["accuracy", "f1"], default="")
    args = parser.parse_args()

    cfg = load_training_config(_ROOT / args.config)
    print_gpu_status()
    paths = resolve_symbol_paths(args.symbol, cfg)
    npz_path = Path(args.npz) if args.npz else paths["npz"]
    if not npz_path.is_file():
        print(f"缺少训练包 {npz_path}，请先运行 prepare_training_data.py")
        return 1

    data = load_npz(npz_path)
    x = data["struct"]
    y = signed_to_class(data["labels"])

    kn_cfg = cfg.get("knowledge_net") or {}
    sym_cfg = (kn_cfg.get("oil") or {}) if args.symbol.upper() == "USOIL" else {}
    dev_cfg = cfg.get("device") or {}
    out_model = paths["knowledge_model"]
    out_scaler = paths["knowledge_scaler"]
    log_path = ensure_logs_dir() / f"knowledge_{args.symbol.upper()}.log"

    stats = train_knowledge_net(
        x,
        y,
        val_ratio=float(
            args.val_ratio if args.val_ratio >= 0 else kn_cfg.get("val_ratio", 0.2)
        ),
        epochs=int(args.epochs if args.epochs > 0 else sym_cfg.get("epochs", kn_cfg.get("epochs", 100))),
        batch_size=int(kn_cfg.get("batch_size", 256)),
        lr=float(args.lr if args.lr > 0 else sym_cfg.get("lr", kn_cfg.get("lr", 0.001))),
        patience=int(args.patience if args.patience > 0 else sym_cfg.get("patience", kn_cfg.get("patience", 10))),
        hidden_dim=int(args.hidden_dim if args.hidden_dim > 0 else sym_cfg.get("hidden_dim", kn_cfg.get("hidden_dim", 64))),
        embed_dim=int(kn_cfg.get("embed_dim", 32)),
        out_path=out_model,
        scaler_path=out_scaler,
        shuffle_train=not args.no_shuffle and bool(sym_cfg.get("shuffle_train", kn_cfg.get("shuffle_train", True))),
        log_path=log_path,
        use_smote=bool(kn_cfg.get("use_smote", True)),
        smote_ratio=float(
            args.smote_ratio if args.smote_ratio >= 0 else kn_cfg.get("smote_ratio", 0.3)
        ),
        class_weights=list(kn_cfg.get("class_weights") or [5.0, 1.0, 5.0]),
        device=str(dev_cfg.get("torch", "auto")),
        num_res_blocks=int(
            args.num_res_blocks if args.num_res_blocks > 0 else sym_cfg.get("num_res_blocks", kn_cfg.get("num_res_blocks", 3))
        ),
        select_by=str(args.select_by or sym_cfg.get("select_by", kn_cfg.get("select_by", "accuracy"))),
    )

    min_f1 = float(kn_cfg.get("min_macro_f1", 0.45))
    min_acc = float(kn_cfg.get("min_val_accuracy", 0.60))
    print("=== 验收 ===")
    print(f"验证准确率: {stats['val_accuracy']:.2%} (目标 ≥{min_acc:.0%})")
    print(f"宏平均 F1: {stats['macro_f1']:.4f} (目标 ≥{min_f1})")
    print(f"模型大小: {stats['model_size_kb']:.1f} KB (目标 <1024 KB)")
    print(f"单样本推理: {stats['infer_ms_single']:.3f} ms (ONNX 目标 <5 ms)")
    print(f"模型: {stats['model_path']}")
    print(f"Scaler: {stats['scaler_path']}")
    ok = (
        stats["val_accuracy"] >= min_acc
        and stats["passed_size"]
        and stats["macro_f1"] >= min_f1
    )
    print("结果:", "PASS" if ok else "FAIL — 未达验收标准，禁止启用智能体")
    if ok:
        print("提示: 运行 py -3 scripts/convert_knowledge_net_to_onnx.py 导出 ONNX 加速推理")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
