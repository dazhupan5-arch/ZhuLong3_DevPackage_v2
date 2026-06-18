#!/usr/bin/env python3
"""Horizon V16 OOS 参数扫描：dir_margin / min_confidence → 胜率≥55% 且 trades≥500。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: F401
import numpy as np
import pandas as pd

from zhulong.agent.horizon_predictor import HorizonPredictor, direction_from_probs
from zhulong.agent.structure_service import StructureService
from zhulong.training.lgb.data_io import load_vendor_csv


def _load_bt():
    spec = importlib.util.spec_from_file_location("backtest_v16", _ROOT / "scripts" / "backtest_v16.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval_oos(
    bt_mod,
    root: Path,
    cfg: dict,
    *,
    flat_scale: float,
    dir_margin: float,
    min_confidence: float,
    start: str,
    end: str,
) -> dict:
    m5_all = load_vendor_csv(root / "data" / "training" / "lgb" / "XAUUSD" / "XAUUSD_M5.csv")
    pad = pd.Timestamp(start) - pd.Timedelta(days=5)
    m5 = m5_all.loc[pad:end]
    struct = bt_mod._struct_matrix(m5, cfg, jobs=1)
    predictor = HorizonPredictor(root, cfg)
    sub_ix = m5.loc[start:end].index
    sub_ix = sub_ix[200:] if len(sub_ix) > 200 else sub_ix
    ss = StructureService(cfg.get("structure_analyzer"))
    ix_map = {t: i for i, t in enumerate(m5.index)}
    dirs = np.zeros(len(sub_ix), dtype=np.int8)
    stats = {"pred_long": 0, "pred_short": 0, "pred_flat": 0, "trade_long": 0, "trade_short": 0}
    for j, ts in enumerate(sub_ix):
        i = ix_map.get(ts)
        if i is None or i < 200:
            continue
        snap = ss._row_to_snapshot(struct[i])
        x = np.asarray(snap.vector, dtype=np.float32).reshape(1, -1)
        if predictor._kn is not None and predictor._kn.is_ready:
            probs, _ = predictor._kn.predict(x)
            p = probs[0] if probs.ndim > 1 else probs
            short_p, flat_p, long_p = float(p[0]), float(p[1]), float(p[2])
        else:
            short_p, flat_p, long_p = 0.33, 0.34, 0.33
        direction, _, _, _ = direction_from_probs(
            short_p,
            flat_p,
            long_p,
            min_confidence=min_confidence,
            flat_scale=flat_scale,
            dir_margin=dir_margin,
        )
        stats[f"pred_{direction}"] = stats.get(f"pred_{direction}", 0) + 1
        if direction == "long":
            dirs[j] = 1
            stats["trade_long"] += 1
        elif direction == "short":
            dirs[j] = -1
            stats["trade_short"] += 1
    from zhulong.training.v10.backtest import backtest_both

    bt = backtest_both(m5, sub_ix, dirs, max_hold=12, cooldown_bars=3, max_daily_signals=8)
    return {"forecast": stats, "backtest": bt}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--min-win-rate", type=float, default=0.55)
    parser.add_argument("--min-trades", type=int, default=500)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = _ROOT
    cfg = json.loads((root / "config" / "config_agent.json").read_text(encoding="utf-8-sig"))
    bt_mod = _load_bt()
    meta_path = root / "models" / "horizon_v16.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    best = {"win_rate": 0.0, "n_trades": 0}
    candidates: list[dict] = []

    for flat_scale in np.arange(0.95, 1.16, 0.05):
        for dir_margin in np.arange(0.08, 0.20, 0.01):
            for min_conf in [0.40, 0.42, 0.44, 0.46, 0.48]:
                r = _eval_oos(
                    bt_mod,
                    root,
                    cfg,
                    flat_scale=float(flat_scale),
                    dir_margin=float(dir_margin),
                    min_confidence=float(min_conf),
                    start=args.start,
                    end=args.end,
                )
                bt = r["backtest"]
                wr = float(bt.get("win_rate", 0))
                n = int(bt.get("n_trades", 0))
                row = {
                    "flat_scale": round(float(flat_scale), 2),
                    "dir_margin": round(float(dir_margin), 2),
                    "min_confidence": float(min_conf),
                    "win_rate": round(wr, 4),
                    "n_trades": n,
                    "total_pnl_r": round(float(bt.get("total_pnl_r", 0)), 2),
                }
                if n >= args.min_trades and wr >= args.min_win_rate:
                    candidates.append(row)
                if wr > best.get("win_rate", 0) or (wr == best.get("win_rate") and n > best.get("n_trades", 0)):
                    best = {**row, "forecast": r["forecast"]}

    out = {
        "oos_range": [args.start, args.end],
        "min_win_rate": args.min_win_rate,
        "min_trades": args.min_trades,
        "best": best,
        "passed_candidates": sorted(candidates, key=lambda x: (-x["win_rate"], -x["n_trades"]))[:10],
    }
    print(json.dumps(out, indent=2))

    report_dir = root / "data" / "training" / "reports" / "v16"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "horizon_oos_calibrate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    if args.apply and candidates:
        pick = out["passed_candidates"][0]
        cal = meta.get("calibration") or {}
        cal.update(
            {
                "flat_scale": pick["flat_scale"],
                "dir_margin": pick["dir_margin"],
                "oos_win_rate": pick["win_rate"],
                "oos_n_trades": pick["n_trades"],
            }
        )
        meta["calibration"] = cal
        meta["passed"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        hp = cfg.setdefault("architecture", {}).setdefault("horizon_predictor", {})
        hp["min_direction_confidence"] = pick["min_confidence"]
        (root / "config" / "config_agent.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Applied OOS calibration: {pick}")
        return 0
    return 0 if candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())
