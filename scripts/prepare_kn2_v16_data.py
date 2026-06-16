#!/usr/bin/env python3
"""KN2 V16 训练数据：struct(30) + horizon_probs(3) + horizon_embed(32) = 65 维。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: F401

KN2_V16_MARKET_DIM = 65
STRUCT_DIM = 30
HORIZON_PROB_DIM = 3
HORIZON_EMBED_DIM = 32


def _save_progress(progress_path: Path, done: int, total: int) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(
            {
                "done": done,
                "total": total,
                "pct": round(100.0 * done / max(total, 1), 2),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/clean/training_horizon_v16.npz")
    parser.add_argument("--horizon-onnx", default="models/horizon_v16.onnx")
    parser.add_argument("--horizon-scaler", default="models/horizon_v16_scaler.pkl")
    parser.add_argument("--out", default="data/clean/kn2_training_v16.npz")
    parser.add_argument("--checkpoint-every", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    npz_path = _ROOT / args.npz
    if not npz_path.is_file():
        print(f"缺少 {npz_path}，请先 prepare/enrich horizon_v16 NPZ")
        return 1

    out_path = _ROOT / args.out
    cache_dir = _ROOT / "data" / "training" / "kn2_v16"
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cache_dir / "prepare_progress.json"
    probs_cache = cache_dir / "horizon_probs.npy"
    emb_cache = cache_dir / "horizon_embed.npy"

    raw = np.load(npz_path, allow_pickle=True)
    struct = np.asarray(raw["struct"], dtype=np.float32)
    n = len(struct)
    print(f"struct rows: {n:,} x {struct.shape[1]}")

    if args.rebuild and probs_cache.is_file():
        probs_cache.unlink()
    if args.rebuild and emb_cache.is_file():
        emb_cache.unlink()

    start = 0
    probs = np.zeros((n, HORIZON_PROB_DIM), dtype=np.float32)
    embed = np.zeros((n, HORIZON_EMBED_DIM), dtype=np.float32)
    if progress_path.is_file() and not args.rebuild:
        try:
            prog = json.loads(progress_path.read_text(encoding="utf-8"))
            start = int(prog.get("done", 0))
        except Exception:
            start = 0
    if start >= n and probs_cache.is_file() and emb_cache.is_file():
        probs = np.load(probs_cache)
        embed = np.load(emb_cache)
        print(f"horizon cache complete ({n:,} rows), skip inference")
    elif start > 0 and probs_cache.is_file() and emb_cache.is_file():
        probs = np.load(probs_cache)
        embed = np.load(emb_cache)
        print(f"resume horizon inference from row {start:,} / {n:,}")

    if start < n:
        from zhulong.agent.knowledge_net import KnowledgeNetInference

        onnx = _ROOT / args.horizon_onnx
        scaler = _ROOT / args.horizon_scaler
        if not onnx.is_file():
            print(f"缺少 {onnx}")
            return 1
        kn = KnowledgeNetInference(onnx, scaler_path=scaler if scaler.is_file() else None, allow_pytorch=False)
        if not kn.is_ready:
            print("Horizon ONNX 未就绪")
            return 1

        bs = max(512, args.batch_size)
        ckpt = max(5000, args.checkpoint_every)
        for i in range(start, n, bs):
            j = min(i + bs, n)
            x = struct[i:j]
            p, e = kn.predict(x)
            probs[i:j] = p[:, :HORIZON_PROB_DIM] if p.ndim > 1 else p.reshape(1, -1)[: j - i]
            if e is None or len(e) == 0:
                embed[i:j] = 0.0
            else:
                e_arr = np.asarray(e, dtype=np.float32)
                if e_arr.ndim == 1:
                    e_arr = e_arr.reshape(1, -1)
                embed[i:j, : min(HORIZON_EMBED_DIM, e_arr.shape[1])] = e_arr[:, :HORIZON_EMBED_DIM]
            if (j - start) % ckpt == 0 or j == n:
                np.save(probs_cache, probs)
                np.save(emb_cache, embed)
                _save_progress(progress_path, j, n)
                print(f"horizon infer {j:,} / {n:,} ({100.0 * j / n:.1f}%) [checkpoint]")

    market_feat = np.concatenate([struct[:, :STRUCT_DIM], probs, embed], axis=1).astype(np.float32)
    if market_feat.shape[1] != KN2_V16_MARKET_DIM:
        print(f"WARN: market_feat dim {market_feat.shape[1]} != {KN2_V16_MARKET_DIM}")
    np.nan_to_num(market_feat, nan=0.0, posinf=10.0, neginf=-10.0, copy=False)

    out = {k: raw[k] for k in raw.files}
    out["market_feat"] = market_feat
    out["market_dim"] = np.array([KN2_V16_MARKET_DIM])
    out["feature_layout"] = np.array(["struct30+horizon_prob3+horizon_embed32"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024**2:.1f} MB)")
    print(f"  market_feat: {market_feat.shape}")
    _save_progress(progress_path, n, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
