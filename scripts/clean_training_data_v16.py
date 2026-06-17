#!/usr/bin/env python3
"""V16 训练数据清洗：CSV → 干净 NPZ → data/clean/（可 commit + LFS push）。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.training_utils import (
    clean_m5_bars,
    load_m5_csv,
    load_training_config,
    resolve_symbol_paths,
    resolve_v16_paths,
)

CLEAN_DIR = _ROOT / "data" / "clean"
REPORT_PATH = CLEAN_DIR / "cleaning_report.json"


def _load_raw_m5(path: Path, start: str, end: str) -> pd.DataFrame:
    return load_m5_csv(path, start, end, clean=False, filter_low_volume=False)


def _audit_df(df: pd.DataFrame) -> dict:
    times = pd.to_datetime(df["time"], utc=True)
    weekend = int((times.dt.dayofweek >= 5).sum())
    dup = int(df["time"].duplicated().sum())
    nan = int(df[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum())
    ohlc_bad = int(
        (
            (df["high"] < df[["open", "close", "low"]].max(axis=1))
            | (df["low"] > df[["open", "close", "high"]].min(axis=1))
            | (df["high"] < df["low"])
        ).sum()
    )
    return {
        "rows": len(df),
        "weekend_bars": weekend,
        "duplicate_timestamps": dup,
        "nan_rows": nan,
        "ohlc_invalid": ohlc_bad,
        "time_start": str(times.min()),
        "time_end": str(times.max()),
    }


def _save_clean_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def _run_script(script: str, args: list[str]) -> int:
    cmd = [sys.executable, str(_ROOT / "scripts" / script), *args]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(_ROOT))


def _rebuild_horizon_npz(
    cleaned: pd.DataFrame,
    source_npz: Path,
    out_path: Path,
    *,
    horizon: int = 12,
    gain: float = 0.002,
    symbol: str = "XAUUSD",
) -> dict:
    """按 time 与旧 horizon NPZ 对齐，复用 struct，用干净 OHLC 重算 labels/atr。"""
    old = np.load(source_npz, allow_pickle=True)
    old_times = pd.to_datetime(old["time"])
    pos_map = {t: i for i, t in enumerate(old_times)}

    cleaned = cleaned.sort_values("time").reset_index(drop=True)
    pick_old: list[int] = []
    keep_rows: list[int] = []
    for i, t in enumerate(pd.to_datetime(cleaned["time"], utc=True)):
        j = pos_map.get(t)
        if j is not None:
            pick_old.append(j)
            keep_rows.append(i)

    sub = cleaned.iloc[keep_rows].reset_index(drop=True)
    struct = old["struct"][pick_old].astype(np.float32)
    close = sub["close"].values.astype(np.float64)
    labels = np.zeros(len(close), dtype=np.int8)
    for i in range(len(close) - horizon):
        ret = (close[i + horizon] - close[i]) / max(close[i], 1e-9)
        if ret > gain:
            labels[i] = 1
        elif ret < -gain:
            labels[i] = -1

    high = sub["high"].values
    low = sub["low"].values
    open_ = sub["open"].values
    volume = sub["volume"].values
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = np.zeros(len(close), dtype=np.float64)
    for i in range(14, len(close)):
        atr[i] = tr[i - 13 : i + 1].mean()
    atr[:14] = atr[14]

    n = len(sub)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        symbol=np.array([symbol.upper()]),
        time=sub["time"].astype(str).values,
        open=open_[:n],
        high=high[:n],
        low=low[:n],
        close=close[:n],
        volume=volume[:n],
        atr=atr[:n],
        struct=struct[:n],
        labels=labels[:n],
        horizon=np.array([horizon]),
        gain=np.array([gain]),
    )
    dropped = len(cleaned) - n
    c = {int(v): int((labels == v).sum()) for v in (-1, 0, 1)}
    print(
        f"horizon npz {out_path.name}: rows={n} (dropped {dropped} without struct) "
        f"short={c[-1]} flat={c[0]} long={c[1]}"
    )
    return {"rows": n, "dropped_no_struct": dropped, "label_dist": c}


def _rebuild_kn2_npz(horizon_npz: Path, source_kn2: Path, out_path: Path) -> dict:
    """按 time 对齐旧 KN2 NPZ，复用 market_feat。"""
    hz = np.load(horizon_npz, allow_pickle=True)
    kn2 = np.load(source_kn2, allow_pickle=True)
    hz_times = pd.to_datetime(hz["time"])
    kn2_times = pd.to_datetime(kn2["time"])
    pos_map = {t: i for i, t in enumerate(kn2_times)}

    pick_kn2: list[int] = []
    keep_hz: list[int] = []
    for i, t in enumerate(hz_times):
        j = pos_map.get(t)
        if j is not None:
            pick_kn2.append(j)
            keep_hz.append(i)

    out: dict = {k: hz[k] for k in hz.files}
    if "market_feat" in kn2.files:
        out["market_feat"] = kn2["market_feat"][pick_kn2].astype(np.float32)
    for k in ("market_dim", "feature_layout"):
        if k in kn2.files:
            out[k] = kn2[k]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"kn2 npz {out_path.name}: rows={len(pick_kn2)} market_dim={out.get('market_dim')}")
    return {"rows": len(pick_kn2), "market_dim": int(out["market_dim"][0]) if "market_dim" in out else None}

    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    n = len(data.get("close", []))
    labels = data.get("labels")
    label_dist = {}
    if labels is not None:
        for v in (-1, 0, 1):
            label_dist[str(v)] = int((labels == v).sum())
    struct = data.get("struct")
    nan_struct = int(np.isnan(struct).sum()) if struct is not None else 0
    return {
        "path": str(path.relative_to(_ROOT)).replace("\\", "/"),
        "rows": n,
        "label_dist": label_dist,
        "struct_nan": nan_struct,
        "size_mb": round(path.stat().st_size / 1024**2, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="清洗 V16 训练数据并写入 data/clean/")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--csv", default="", help="原始 CSV，默认 config 路径")
    parser.add_argument("--skip-horizon", action="store_true", help="仅清洗 CSV，不重建 horizon NPZ")
    parser.add_argument("--skip-kn2", action="store_true", help="不重建 kn2 NPZ（需 horizon ONNX）")
    parser.add_argument("--full-rebuild", action="store_true", help="从零重算结构特征（慢）")
    parser.add_argument("--source-horizon-npz", default="data/training_horizon_v16.npz")
    parser.add_argument("--source-kn2-npz", default="data/kn2_training_v16.npz")
    parser.add_argument("--rebuild-struct", action="store_true", help="强制重算结构特征")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    cfg = load_training_config(_ROOT / "config_training.yaml")
    paths = resolve_symbol_paths(args.symbol, cfg)
    v16 = resolve_v16_paths(args.symbol, cfg)
    csv_path = Path(args.csv) if args.csv else paths["csv"]
    clean_csv = Path(v16["clean_csv"])
    horizon_npz = Path(v16["horizon_npz"])
    kn2_npz = Path(v16["kn2_npz"])
    if not csv_path.is_file():
        print(f"CSV 不存在: {csv_path}")
        return 1

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_m5_csv(
        csv_path, args.start, args.end, clean=False, filter_low_volume=False
    )
    before = _audit_df(raw_df)

    clean_report: dict[str, int] = {}
    cleaned = clean_m5_bars(raw_df, report=clean_report)
    # 与训练流水线一致：去掉最低 5% 成交量
    if len(cleaned) > 100:
        q = cleaned["volume"].quantile(0.05)
        n_low = int((cleaned["volume"] < q).sum())
        if n_low:
            clean_report["low_volume_bottom5pct"] = n_low
            cleaned = cleaned[cleaned["volume"] >= q].reset_index(drop=True)
    after = _audit_df(cleaned)

    _save_clean_csv(cleaned, clean_csv)

    report = {
        "version": "v16_clean_1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol.upper(),
        "source_csv": str(csv_path.relative_to(_ROOT)).replace("\\", "/"),
        "clean_csv": str(clean_csv.relative_to(_ROOT)).replace("\\", "/"),
        "date_range": {"start": args.start, "end": args.end},
        "before": before,
        "after": after,
        "clean_steps": clean_report,
        "artifacts": {},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    if args.skip_horizon:
        print(f"跳过 horizon NPZ；已写入 {clean_csv} 与 {REPORT_PATH}")
        return 0

    source_hz = _ROOT / args.source_horizon_npz
    if args.full_rebuild or not source_hz.is_file():
        horizon_args = [
            "--symbol",
            args.symbol,
            "--end",
            args.end,
            "--horizon",
            str(v16["horizon"]),
            "--gain",
            str(v16["gain"]),
            "--csv",
            str(clean_csv),
            "--out",
            str(horizon_npz),
            "--jobs",
            str(max(1, args.jobs)),
        ]
        if args.rebuild_struct or args.full_rebuild:
            horizon_args.append("--rebuild")
        rc = _run_script("prepare_horizon_v16_data.py", horizon_args)
        if rc != 0:
            return rc
    else:
        meta = _rebuild_horizon_npz(
            cleaned,
            source_hz,
            horizon_npz,
            horizon=int(v16["horizon"]),
            gain=float(v16["gain"]),
            symbol=args.symbol,
        )
        report["horizon_rebuild"] = meta
    report["artifacts"]["horizon_npz"] = _verify_npz(horizon_npz)

    if args.skip_kn2:
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("跳过 kn2 NPZ")
        return 0

    source_kn2 = _ROOT / args.source_kn2_npz
    if args.full_rebuild or not source_kn2.is_file():
        kn2_args = [
            "--npz",
            str(horizon_npz),
            "--out",
            str(kn2_npz),
        ]
        if v16["horizon_onnx"].is_file():
            kn2_args.extend(["--horizon-onnx", str(v16["horizon_onnx"])])
            kn2_args.extend(["--horizon-scaler", str(v16["horizon_scaler"])])
        rc = _run_script("prepare_kn2_v16_data.py", kn2_args)
        if rc != 0:
            return rc
    else:
        meta = _rebuild_kn2_npz(horizon_npz, source_kn2, kn2_npz)
        report["kn2_rebuild"] = meta
    report["artifacts"]["kn2_npz"] = _verify_npz(kn2_npz)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"完成。干净数据: {horizon_npz}, {kn2_npz}")
    return 0


def _verify_npz(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    n = len(data.get("close", []))
    labels = data.get("labels")
    label_dist = {}
    if labels is not None:
        for v in (-1, 0, 1):
            label_dist[str(v)] = int((labels == v).sum())
    struct = data.get("struct")
    nan_struct = int(np.isnan(struct).sum()) if struct is not None else 0
    return {
        "path": str(path.relative_to(_ROOT)).replace("\\", "/"),
        "rows": n,
        "label_dist": label_dist,
        "struct_nan": nan_struct,
        "size_mb": round(path.stat().st_size / 1024**2, 2),
    }


if __name__ == "__main__":
    raise SystemExit(main())
