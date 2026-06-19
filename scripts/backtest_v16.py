#!/usr/bin/env python3
"""V16 回测：Structure → Horizon(对称) → 1h 交易（预计算结构特征）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: F401 — Windows: 须在 numpy/pandas 之前加载，避免 c10.dll 冲突

import numpy as np
import pandas as pd

from zhulong.agent.horizon_predictor import HorizonPredictor
from zhulong.agent.structure_analyzer import StructureAnalyzer
from zhulong.agent.structure_service import StructureService
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.v10.backtest import backtest_both


def _load_config(root: Path) -> dict:
    return json.loads((root / "config" / "config_agent.json").read_text(encoding="utf-8-sig"))


def _struct_matrix(m5: pd.DataFrame, cfg: dict, jobs: int = 1) -> np.ndarray:
    cache = _ROOT / "data" / "training" / "v16" / "XAUUSD" / "struct_features.parquet"
    if cache.is_file():
        cached = pd.read_parquet(cache)
        common = m5.index.intersection(cached.index)
        if len(common) >= len(m5) * 0.9:
            return cached.reindex(m5.index).fillna(0).values.astype(np.float32)
    sa = StructureAnalyzer(cfg.get("structure_analyzer") or {})
    struct = sa.compute_all(m5, progress_every=20000, n_jobs=jobs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(struct, index=m5.index[: len(struct)]).to_parquet(cache)
    return struct


def _dirs_from_struct(
    m5: pd.DataFrame,
    index: pd.DatetimeIndex,
    struct: np.ndarray,
    predictor: HorizonPredictor,
    struct_service: StructureService,
) -> tuple[np.ndarray, dict]:
    ix_map = {t: i for i, t in enumerate(m5.index)}
    dirs = np.zeros(len(index), dtype=np.int8)
    stats = {"pred_long": 0, "pred_short": 0, "pred_flat": 0, "trade_long": 0, "trade_short": 0}
    for j, ts in enumerate(index):
        i = ix_map.get(ts)
        if i is None or i < 200:
            continue
        snap = struct_service._row_to_snapshot(struct[i])
        fc = predictor.predict(snap)
        stats[f"pred_{fc.direction}"] = stats.get(f"pred_{fc.direction}", 0) + 1
        if fc.direction == "long":
            dirs[j] = 1
            stats["trade_long"] += 1
        elif fc.direction == "short":
            dirs[j] = -1
            stats["trade_short"] += 1
    return dirs, stats


def run_period(
    m5: pd.DataFrame,
    struct: np.ndarray,
    predictor: HorizonPredictor,
    start: str,
    end: str,
    cfg: dict,
    *,
    backtest_params: dict | None = None,
) -> dict:
    sub_ix = m5.loc[start:end].index
    sub_ix = sub_ix[200:] if len(sub_ix) > 200 else sub_ix
    if len(sub_ix) < 10:
        return {"error": "too_few_bars"}
    ss = StructureService(cfg.get("structure_analyzer"))
    dirs, ps = _dirs_from_struct(m5, sub_ix, struct, predictor, ss)
    bt_kwargs = dict(backtest_params or {})
    bt = backtest_both(
        m5,
        sub_ix,
        dirs,
        max_hold=12,
        cooldown_bars=3,
        max_daily_signals=8,
        **bt_kwargs,
    )
    return {"forecast": ps, "backtest": bt}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    root = _ROOT
    cfg = _load_config(root)
    m5_all = load_vendor_csv(root / "data" / "training" / "lgb" / "XAUUSD" / "XAUUSD_M5.csv")
    pad = pd.Timestamp(args.start) - pd.Timedelta(days=5)
    m5 = m5_all.loc[pad : args.end]
    print(f"M5 slice {len(m5)} bars, computing/using struct...", flush=True)
    struct = _struct_matrix(m5, cfg, jobs=args.jobs)
    predictor = HorizonPredictor(root, cfg)
    print(f"Horizon ready={predictor.is_ready}", flush=True)

    main = run_period(m5, struct, predictor, args.start, args.end, cfg)
    print("=" * 72)
    print(f"  V16 BACKTEST {args.start} ~ {args.end}")
    print(f"  Forecast: {main.get('forecast')}")
    print(f"  Backtest: {json.dumps(main.get('backtest', {}), indent=2, default=str)}")
    print("=" * 72)

    for label, s, e in [("March 10-18", "2026-03-10", "2026-03-18"), ("June 10", "2026-06-10", "2026-06-10")]:
        if pd.Timestamp(s) < m5.index.min() or pd.Timestamp(e) > m5_all.index.max():
            continue
        ext = m5_all.loc[pd.Timestamp(s) - pd.Timedelta(days=5) : e]
        st = _struct_matrix(ext, cfg, jobs=1)
        r = run_period(ext, st, predictor, s, e, cfg)
        b = r.get("backtest", {})
        print(f"\n  [{label}] forecast={r.get('forecast')}")
        print(f"    trades={b.get('n_trades')} long={b.get('n_long')} short={b.get('n_short')} "
              f"win={b.get('win_rate', 0):.1%} pnl_r={b.get('total_pnl_r', 0):.1f}")

    out = root / "data" / "training" / "reports" / "v16" / "backtest_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(main, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
