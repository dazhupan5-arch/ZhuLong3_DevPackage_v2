#!/usr/bin/env python3
"""KN2 V16 训练：65 维 struct+horizon 特征，输出 kn2_trader_v16.pth。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from zhulong.utils.win_dll import configure_native_dll_paths

    configure_native_dll_paths()
except Exception:
    pass

import torch  # noqa: F401 — 须在 numpy/pandas 之前

import numpy as np
import pandas as pd

from zhulong.agent.knowledge_net_kn2 import encode_position_state, train_kn2_end_to_end, train_kn2_fast
from zhulong.agent.trading_env_kn2 import _KN2_LABELS_FAST, generate_kn2_training_labels
from zhulong.agent.training_utils import load_npz
from zhulong.strategies.indicators import atr_series
from zhulong.utils.device import print_gpu_status

KN2_V16_MARKET_DIM = 65
# 压低 hold、抬高 long/short，避免全 hold 塌缩（上一版无 class_weights 导致验收失败）
KN2_V16_CLASS_WEIGHTS = [0.85, 2.5, 2.5, 1.0, 1.0, 1.0]


def _parse_class_weights(raw: str) -> list[float]:
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) not in (3, 6):
        raise ValueError("class-weights 需要 3 或 6 个逗号分隔浮点数")
    if len(parts) == 3:
        parts = parts + [1.0, 1.0, 1.0]
    return parts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/kn2_training_v16.npz")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output", default="models/kn2_trader_v16.pth")
    parser.add_argument("--mode", choices=("fast", "e2e"), default="fast",
                        help="fast=batched GRU (GPU-friendly); e2e=slow bar-by-bar")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="sequences per batch (fast mode only; RTX 3050: 32-64)")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument(
        "--class-weights",
        default=",".join(str(x) for x in KN2_V16_CLASS_WEIGHTS),
        help="hold,long,short[,...] 共 3 或 6 维；默认压低 hold",
    )
    args = parser.parse_args()

    try:
        class_weights = _parse_class_weights(args.class_weights)
    except ValueError as ex:
        print(ex)
        return 1

    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}，请先运行 prepare_kn2_v16_data.py")
        return 1

    print_gpu_status()
    data = load_npz(npz_path)
    market_feat = np.asarray(data["market_feat"], dtype=np.float32)
    n = len(market_feat)
    md = int(data.get("market_dim", [KN2_V16_MARKET_DIM])[0])
    print(f"bars={n:,} market_dim={md} feat_shape={market_feat.shape}")

    times = pd.to_datetime(data["time"]) if "time" in data else pd.date_range("2020-01-01", periods=n, freq="5min")
    df = pd.DataFrame(
        {
            "open": data.get("open", np.zeros(n)),
            "high": data.get("high", np.zeros(n)),
            "low": data.get("low", np.zeros(n)),
            "close": data.get("close", np.zeros(n)),
            "volume": data.get("volume", np.zeros(n)),
        },
        index=times[:n],
    )
    atr = np.asarray(data.get("atr", df["close"].values * 0.001), dtype=np.float32)
    if "atr" not in data:
        atr = atr_series(df).bfill().fillna(df["close"] * 0.001).values.astype(np.float32)
    df = df.assign(atr=atr)

    train_mask = df.index.year <= 2024
    val_mask = df.index.year == 2025
    if val_mask.sum() < 1000:
        split = int(n * 0.85)
        train_idx = np.zeros(n, dtype=bool)
        train_idx[:split] = True
        val_idx = ~train_idx
    else:
        train_idx = np.asarray(train_mask, dtype=bool)
        val_idx = np.asarray(val_mask, dtype=bool)

    print(f"train={train_idx.sum():,} val={val_idx.sum():,}")
    print(f"class_weights={class_weights}")

    print(f"Generating KN2 labels (fast={'numba' if _KN2_LABELS_FAST else 'python'})...")
    t_label = time.perf_counter()
    labels = generate_kn2_training_labels(df, market_feat, progress_every=50000 if not _KN2_LABELS_FAST else 0)
    print(
        f"labels done in {time.perf_counter() - t_label:.1f}s | "
        f"should_trade={labels['should_trade'].mean() * 100:.1f}% | "
        f"actions={np.bincount(labels['action'], minlength=6).tolist()}"
    )
    pos_states = np.tile(encode_position_state(), (n, 1)).astype(np.float32)

    out_path = _ROOT / args.output
    backup = _ROOT / "models" / "backups" / "kn2_trader_v16_pretrain"
    if out_path.is_file():
        backup.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(out_path, backup / out_path.name)
        meta = out_path.with_suffix(".meta.json")
        if meta.is_file():
            shutil.copy2(meta, backup / meta.name)

    t0 = time.perf_counter()
    train_kw = dict(
        val_ratio=0.15,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        embed_dim=64,
        num_actions=6,
        out_path=out_path,
        device=args.device,
        sequence_length=64,
        class_weights=class_weights,
    )
    if args.mode == "fast":
        stats = train_kn2_fast(
            market_feat[train_idx],
            pos_states[train_idx],
            {k: v[train_idx] for k, v in labels.items()},
            market_dim=md,
            batch_size=args.batch_size,
            **train_kw,
        )
    else:
        stats = train_kn2_end_to_end(
            market_feat[train_idx],
            pos_states[train_idx],
            {k: v[train_idx] for k, v in labels.items()},
            market_dim=md,
            **train_kw,
        )
    elapsed = time.perf_counter() - t0
    report = {
        "architecture": "kn2_v16",
        "market_dim": md,
        "training_rows": int(train_idx.sum()),
        "val_rows": int(val_idx.sum()),
        "epochs_requested": args.epochs,
        "mode": args.mode,
        "batch_size": args.batch_size if args.mode == "fast" else None,
        "device_requested": args.device,
        "class_weights": class_weights,
        "elapsed_sec": round(elapsed, 1),
        **stats,
    }
    report_path = _ROOT / "data" / "training" / "reports" / "kn2_v16" / "train_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"KN2 V16 model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
