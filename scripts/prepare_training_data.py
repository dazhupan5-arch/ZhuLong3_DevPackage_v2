#!/usr/bin/env python3
"""离线预计算训练数据包（结构特征 + 标签 + ATR）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.structure_analyzer import StructureAnalyzer
from zhulong.agent.training_utils import (
    build_signed_labels,
    compute_atr,
    ensure_logs_dir,
    load_m5_csv,
    load_training_config,
    resolve_symbol_paths,
)
from zhulong.utils.device import print_gpu_status, resolve_structure_n_jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="预计算 KnowledgeNet / PPO 训练 npz")
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD", "USOIL"])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--csv", default=None, help="覆盖 CSV 路径")
    parser.add_argument("--out", default=None, help="覆盖 npz 输出路径")
    parser.add_argument("--config", default="config_training.yaml")
    parser.add_argument("--max-rows", type=int, default=0, help="调试用：仅处理最近 N 根")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="结构特征并行 worker，0=自动多核，1=单线程",
    )
    parser.add_argument("--check-gpu", action="store_true", help="仅检查 GPU/CUDA 后退出")
    args = parser.parse_args()

    if args.check_gpu:
        return print_gpu_status()

    cfg = load_training_config(_ROOT / args.config)
    paths = resolve_symbol_paths(args.symbol, cfg)
    csv_path = Path(args.csv) if args.csv else paths["csv"]
    if not csv_path.is_file():
        print(f"CSV 不存在: {csv_path}")
        print("请先导出: py -3 scripts/export_m5_mt5.py --symbol", args.symbol)
        return 1

    start = args.start or (cfg.get("data") or {}).get("default_start", "2016-01-01")
    end = args.end or (cfg.get("data") or {}).get("default_end", "2025-12-31")
    horizon = int((cfg.get("data") or {}).get("label_horizon", 12))
    thr = float((cfg.get("data") or {}).get("label_threshold", 0.002))
    if args.symbol.upper() == "USOIL":
        oil_cfg = (cfg.get("knowledge_net") or {}).get("oil") or {}
        horizon = int(oil_cfg.get("label_horizon", horizon))
        thr = float(oil_cfg.get("label_gain", thr))
        print(f"USOIL 标签: horizon={horizon} gain={thr}", flush=True)

    df = load_m5_csv(csv_path, start, end)
    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.iloc[-args.max_rows :].reset_index(drop=True)
    if len(df) < 500:
        print(f"数据过少: {len(df)} 行")
        return 1

    sa_cfg = cfg.get("structure_analyzer") or {}
    analyzer = StructureAnalyzer(sa_cfg)
    m5 = df.set_index("time")
    n_jobs = resolve_structure_n_jobs(
        args.n_jobs if args.n_jobs is not None else sa_cfg.get("n_jobs", 0)
    )
    progress_every = int(sa_cfg.get("progress_every", 2000))

    struct_cache = _ROOT / "data" / "training" / "struct" / args.symbol.upper() / "struct_features.npz"
    csv_sig = (
        f"{csv_path.resolve()}|{csv_path.stat().st_mtime}|{csv_path.stat().st_size}|"
        f"{start}|{end}|{horizon}|{thr}|{sa_cfg.get('periods')}|"
        f"{sa_cfg.get('zigzag_atr_mult')}|{sa_cfg.get('lookback')}"
    )
    struct = None
    if struct_cache.is_file():
        try:
            cached = np.load(struct_cache, allow_pickle=True)
            if str(cached["csv_sig"].item()) == csv_sig:
                struct = cached["struct"]
                print(f"结构特征缓存命中: {struct.shape} ← {struct_cache}", flush=True)
        except Exception as exc:
            print(f"结构特征缓存读取失败，将重算: {exc}", flush=True)

    print_gpu_status()
    if struct is None:
        print(
            f"结构特征预计算: {len(m5)} 根, n_jobs={'auto' if n_jobs == 0 else n_jobs}, "
            f"progress_every={progress_every}",
            flush=True,
        )
        print("说明: StructureAnalyzer 为 CPU 算法，无法用 GPU；多核可缩短耗时。", flush=True)
        struct = analyzer.compute_all(m5, progress_every=progress_every, n_jobs=n_jobs)
        struct_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(struct_cache, struct=struct.astype(np.float32), csv_sig=np.array([csv_sig]))
        print(f"结构特征已缓存: {struct_cache}", flush=True)
    n = min(len(df), len(struct))
    df = df.iloc[:n].reset_index(drop=True)
    struct = struct[:n]

    labels = build_signed_labels(df["close"].values, horizon=horizon, thr=thr)
    atr = compute_atr(df)

    out = Path(args.out) if args.out else paths["npz"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        symbol=np.array([args.symbol.upper()]),
        time=df["time"].astype(str).values,
        open=df["open"].values,
        high=df["high"].values,
        low=df["low"].values,
        close=df["close"].values,
        volume=df["volume"].values,
        atr=atr[:n],
        struct=struct.astype(np.float32),
        labels=labels,
    )

    log = ensure_logs_dir() / f"prepare_{args.symbol.upper()}.log"
    log.write_text(
        f"symbol={args.symbol}\nrows={n}\nstart={start}\nend={end}\nout={out}\n",
        encoding="utf-8",
    )
    print(f"saved {out} rows={n} ({start} ~ {end})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
