#!/usr/bin/env python3
"""V16 Horizon 训练数据：StructureAnalyzer 30 维 + 1h 方向标签（断点续跑，防中途被杀）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.structure_analyzer import FEATURE_DIM, StructureAnalyzer
from zhulong.agent.training_utils import load_m5_csv, load_training_config, resolve_symbol_paths


def _meta_key(m5: pd.DataFrame, jobs: int) -> str:
    return f"{len(m5)}|{m5.index[0]}|{m5.index[-1]}|{jobs}"


def _compute_struct_resumable(
    sa: StructureAnalyzer,
    m5: pd.DataFrame,
    struct_cache: Path,
    progress_path: Path,
    meta_key: str,
    *,
    checkpoint_every: int = 10000,
    jobs: int = 1,
) -> np.ndarray:
    """逐 bar 计算并定期落盘；中断后可从 parquet 续跑。"""
    m5 = m5.sort_index()
    n = len(m5)
    start = 0
    rows = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    if struct_cache.is_file() and progress_path.is_file():
        try:
            prog = json.loads(progress_path.read_text(encoding="utf-8"))
            if prog.get("meta_key") == meta_key and int(prog.get("done", 0)) > 0:
                partial = pd.read_parquet(struct_cache)
                start = min(int(prog["done"]), len(partial), n)
                if start > 0:
                    rows[:start] = partial.iloc[:start].values.astype(np.float32)
                    print(f"resume from checkpoint {start} / {n} ({100.0 * start / n:.1f}%)", flush=True)
        except Exception as ex:
            print(f"WARN: checkpoint load failed, restart from 0: {ex}", flush=True)
            start = 0

    if start >= n:
        return rows

    if jobs > 1 and start == 0:
        print("computing structure features (parallel)...", flush=True)
        out = sa.compute_all(m5, progress_every=checkpoint_every, n_jobs=jobs)
        pd.DataFrame(out, index=m5.index[: len(out)]).to_parquet(struct_cache)
        progress_path.write_text(
            json.dumps({"meta_key": meta_key, "done": n, "total": n}), encoding="utf-8"
        )
        return out

    print("computing structure features (single-thread, checkpoint every "
          f"{checkpoint_every})...", flush=True)
    mtf = sa._build_mtf_context(m5)
    for i in range(start, n):
        rows[i] = sa.compute_row(m5, i, mtf=mtf)
        if checkpoint_every and i > 0 and i % checkpoint_every == 0:
            done = i + 1
            pd.DataFrame(rows[:done], index=m5.index[:done]).to_parquet(struct_cache)
            progress_path.write_text(
                json.dumps({"meta_key": meta_key, "done": done, "total": n}), encoding="utf-8"
            )
            print(f"结构特征 {done} / {n} ({100.0 * done / n:.1f}%) [checkpoint saved]", flush=True)

    pd.DataFrame(rows, index=m5.index).to_parquet(struct_cache)
    progress_path.write_text(json.dumps({"meta_key": meta_key, "done": n, "total": n}), encoding="utf-8")
    print(f"cached {struct_cache}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--gain", type=float, default=0.002)
    parser.add_argument("--quick", type=int, default=0, help="仅末 N 根 M5（建议首次 50000）")
    parser.add_argument("--jobs", type=int, default=1, help="结构特征并行度，默认 1 避免 CPU 打满")
    parser.add_argument("--rebuild", action="store_true", help="删除断点，从零重算")
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    args = parser.parse_args()

    cfg = load_training_config(_ROOT / "config_training.yaml")
    paths = resolve_symbol_paths(args.symbol, cfg)
    df = load_m5_csv(paths["csv"], "2016-01-01", args.end)
    if args.quick > 0 and len(df) > args.quick:
        df = df.iloc[-args.quick :].reset_index(drop=True)
    m5 = df.set_index("time")
    print(f"M5 rows={len(m5)} jobs={args.jobs}", flush=True)

    cache_dir = _ROOT / "data" / "training" / "v16" / args.symbol.upper()
    cache_dir.mkdir(parents=True, exist_ok=True)
    struct_cache = cache_dir / "struct_features.parquet"
    progress_path = cache_dir / "struct_progress.json"
    meta_cache = cache_dir / "struct_meta.txt"
    meta_key = _meta_key(m5, max(1, args.jobs))

    if args.rebuild:
        for p in (struct_cache, progress_path, meta_cache):
            if p.is_file():
                p.unlink()
        print("rebuild: cleared struct checkpoint", flush=True)

    sa = StructureAnalyzer(cfg.get("structure_analyzer") or {})
    struct: np.ndarray | None = None

    if (
        struct_cache.is_file()
        and progress_path.is_file()
        and not args.rebuild
    ):
        try:
            prog = json.loads(progress_path.read_text(encoding="utf-8"))
            if prog.get("meta_key") == meta_key and int(prog.get("done", 0)) >= len(m5):
                struct = pd.read_parquet(struct_cache).values.astype(np.float32)
                print(f"loaded complete cache {struct_cache} shape={struct.shape}", flush=True)
        except Exception:
            struct = None

    if struct is None:
        struct = _compute_struct_resumable(
            sa,
            m5,
            struct_cache,
            progress_path,
            meta_key,
            checkpoint_every=max(1000, args.checkpoint_every),
            jobs=max(1, args.jobs),
        )
        meta_cache.write_text(meta_key, encoding="utf-8")

    close = m5["close"].values.astype(np.float64)
    labels = np.zeros(len(close), dtype=np.int8)
    h, g = args.horizon, args.gain
    for i in range(len(close) - h):
        ret = (close[i + h] - close[i]) / max(close[i], 1e-9)
        if ret > g:
            labels[i] = 1
        elif ret < -g:
            labels[i] = -1

    high = m5["high"].values
    low = m5["low"].values
    open_ = m5["open"].values
    volume = m5["volume"].values
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = np.zeros(len(close), dtype=np.float64)
    for i in range(14, len(close)):
        atr[i] = tr[i - 13 : i + 1].mean()
    atr[:14] = atr[14]

    n = min(len(struct), len(labels))
    out = _ROOT / "data" / "training_horizon_v16.npz"
    np.savez(
        out,
        symbol=np.array([args.symbol.upper()]),
        time=m5.index[:n].astype(str).values,
        open=open_[:n],
        high=high[:n],
        low=low[:n],
        close=close[:n],
        volume=volume[:n],
        atr=atr[:n],
        struct=struct[:n].astype(np.float32),
        labels=labels[:n],
        horizon=np.array([h]),
        gain=np.array([g]),
    )
    c = {int(v): int((labels[:n] == v).sum()) for v in (-1, 0, 1)}
    print(f"saved {out} rows={n} struct={struct.shape[1]} labels short={c[-1]} flat={c[0]} long={c[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
