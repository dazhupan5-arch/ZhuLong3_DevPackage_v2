#!/usr/bin/env python3
"""
Transformer 15% mask 预训练骨架（G2）。
正式训练需配合 download_m1.py 数据；本脚本仅验证管线可 import/运行。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    diff = (pred - target) ** 2
    return float(diff[mask].mean()) if mask.any() else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Transformer mask 预训练骨架")
    parser.add_argument("--feature-dim", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true", help="仅验证依赖与随机数据")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.error("需要 torch: pip install torch")
        return 1

    from zhulong.inference_engine import TransformerEncoder

    enc = TransformerEncoder(feature_dim=args.feature_dim)
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)

    x = torch.randn(4, args.seq_len, args.feature_dim)
    mask = torch.rand(4, args.seq_len, args.feature_dim) < args.mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0

    for step in range(3 if args.dry_run else 100):
        emb = enc(x_masked)
        # 骨架：用 embedding 范数作伪损失，正式版应接 decoder 重建
        loss = emb.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            logger.info("step=%d loss=%.6f", step, float(loss))

    out = Path("data/training/transformer_pretrain_skeleton.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(enc.state_dict(), out)
    logger.info("骨架 checkpoint -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
