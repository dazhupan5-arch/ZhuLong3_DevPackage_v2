#!/usr/bin/env python3
"""V17 全链路含成本 OOS 回测。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from zhulong.agent.structure_service import StructureService
from zhulong.agent.v17.backtest import backtest_direction_signals
from zhulong.agent.v17.direction_scorer import DirectionScorer
from zhulong.agent.v17.execution_composer_v17 import ExecutionComposerV17
from zhulong.agent.v17.location_gate import LocationGate
from zhulong.agent.kn2_location_labels import compute_pos_in_range
from zhulong.training.lgb.data_io import load_vendor_csv


def _load_config(root: Path) -> dict:
    return json.loads((root / "config" / "config_agent.json").read_text(encoding="utf-8-sig"))


def _struct_matrix(m5: pd.DataFrame, cfg: dict) -> np.ndarray:
    from zhulong.agent.structure_analyzer import StructureAnalyzer

    sa = StructureAnalyzer(cfg.get("structure_analyzer") or {})
    return sa.compute_all(m5, progress_every=50000, n_jobs=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--oos-start", default="2025-01-01")
    parser.add_argument("--oos-end", default="2025-12-31")
    parser.add_argument("--with-cost", action="store_true", default=True)
    parser.add_argument("--output", default="data/training/reports/v17/oos_backtest.json")
    args = parser.parse_args()

    root = _ROOT
    cfg = _load_config(root)
    cfg.setdefault("architecture", {})["version"] = "v17"
    m5_all = load_vendor_csv(root / "data" / "training" / "lgb" / "XAUUSD" / "XAUUSD_M5.csv")
    pad = pd.Timestamp(args.oos_start) - pd.Timedelta(days=5)
    m5 = m5_all.loc[pad : args.oos_end]
    sub_ix = m5.loc[args.oos_start : args.oos_end].index[200:]

    struct = _struct_matrix(m5, cfg)
    ds = DirectionScorer(root, cfg)
    lg = LocationGate(root, cfg)
    composer = ExecutionComposerV17(cfg)
    ss = StructureService(cfg.get("structure_analyzer"))
    pos_all = compute_pos_in_range(m5["close"].to_numpy(dtype=np.float32), window=48)

    dirs = np.zeros(len(sub_ix), dtype=np.int8)
    modes: list[str] = []
    stats = {"signals": 0, "filtered": 0}

    ix_map = {t: i for i, t in enumerate(m5.index)}
    for j, ts in enumerate(sub_ix):
        i = ix_map.get(ts)
        if i is None or i < 200:
            continue
        snap = ss._row_to_snapshot(struct[i])
        score = ds.predict(snap)
        loc_q = lg.predict(snap, pos_in_range=float(pos_all[i]), direction_score=score)
        close = float(m5["close"].iloc[i])
        atr = float(m5.get("atr", pd.Series(index=m5.index)).iloc[i]) if "atr" in m5 else close * 0.001
        if "atr" not in m5.columns:
            from zhulong.strategies.indicators import atr_series

            atr = float(atr_series(m5).iloc[i])
        plan = composer.compose_v17(
            direction_score=score,
            location_quality=loc_q,
            snapshot=snap,
            close=close,
            atr=atr,
            pos_in_range=float(pos_all[i]),
        )
        if not plan.should_trade:
            stats["filtered"] += 1
            continue
        stats["signals"] += 1
        d = 1 if plan.direction == "long" else -1
        dirs[j] = d
        modes.append(str(plan.entry_mode))

    bt = backtest_direction_signals(
        m5,
        sub_ix,
        dirs,
        np.array(modes) if modes else None,
        symbol=args.symbol,
        with_cost=args.with_cost,
    )
    result = {"forecast_stats": stats, "backtest": bt}
    print(json.dumps(result, indent=2))

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
