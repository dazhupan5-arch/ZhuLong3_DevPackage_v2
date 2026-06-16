#!/usr/bin/env python3
"""拟合因果图线性系数（Phase 4 离线）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.causal_inference import load_causal_graph, save_causal_coef
from zhulong.agent.training_utils import load_npz, resolve_symbol_paths


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 30:
        return 0.0, 0.0, 0.0
    xm, ym = x.mean(), y.mean()
    var = ((x - xm) ** 2).sum()
    if var < 1e-12:
        return float(ym), 0.0, 0.0
    beta = ((x - xm) * (y - ym)).sum() / var
    alpha = ym - beta * xm
    pred = alpha + beta * x
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(alpha), float(beta), float(r2)


def _build_series_from_npz(data: dict, horizon: int) -> pd.DataFrame:
    close = np.asarray(data["close"], dtype=np.float64)
    ret = np.zeros_like(close)
    ret[:-horizon] = (close[horizon:] - close[:-horizon]) / np.maximum(close[:-horizon], 1e-9) * 100.0
    vol = pd.Series(close).pct_change().rolling(48, min_periods=12).std().fillna(0).to_numpy()
    trend = pd.Series(close).pct_change(12).fillna(0).to_numpy()
    macro_shock = np.clip(0.5 * trend + 0.5 * vol * np.sign(trend), -3, 3)
    risk = vol * 100.0
    dollar = -pd.Series(close).pct_change(24).fillna(0).to_numpy() * 100.0
    demand = pd.Series(ret).rolling(24, min_periods=6).mean().fillna(0).to_numpy()
    return pd.DataFrame(
        {
            "macro_shock": macro_shock,
            "risk_aversion": risk,
            "dollar_index": dollar,
            "demand": demand,
            "price_change": ret,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()


def fit_symbol(symbol: str, npz_path: Path, horizon: int) -> dict:
    data = load_npz(npz_path)
    df = _build_series_from_npz(data, horizon)
    demand_key = "gold_demand" if symbol == "XAUUSD" else "oil_demand"

    a0, a1, r2_ra = _ols(df["risk_aversion"].to_numpy(), df["macro_shock"].to_numpy())
    b0, b1, r2_dx = _ols(df["dollar_index"].to_numpy(), df["macro_shock"].to_numpy())
    c0, c1, _ = _ols(df["demand"].to_numpy(), df["risk_aversion"].to_numpy())
    _, c2, _ = _ols(df["demand"].to_numpy(), df["dollar_index"].to_numpy())
    d0, d1, r2_pc = _ols(df["price_change"].to_numpy(), df["demand"].to_numpy())

    coef = {
        "symbol": symbol,
        "horizon_bars": horizon,
        "risk_aversion": {"intercept": a0, "macro_shock": a1, "r2": r2_ra},
        "dollar_index": {"intercept": b0, "macro_shock": b1, "r2": r2_dx},
        demand_key: {
            "intercept": c0,
            "risk_aversion": c1,
            "dollar_index": c2,
        },
        "price_change": {"intercept": d0, demand_key: d1, "r2": r2_pc},
        "fit_stats": {"n": int(len(df)), "horizon": horizon},
    }
    return coef


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ALL", help="XAUUSD | USOIL | ALL")
    parser.add_argument("--out", default="models/causal_coef.pkl")
    parser.add_argument("--graph", default="config/causal_graph.yaml")
    parser.add_argument("--npz-xau", default="")
    parser.add_argument("--npz-oil", default="")
    args = parser.parse_args()

    graph = load_causal_graph(_ROOT / args.graph)
    symbols_cfg = (graph.get("symbols") or {})
    out: dict = {}
    targets = ["XAUUSD", "USOIL"] if args.symbol.upper() == "ALL" else [args.symbol.upper()]

    for sym in targets:
        paths = resolve_symbol_paths(sym)
        npz = Path(args.npz_xau if sym == "XAUUSD" and args.npz_xau else args.npz_oil if sym == "USOIL" and args.npz_oil else paths["npz"])
        if not npz.is_file():
            print(f"跳过 {sym}: 缺少 {npz}")
            continue
        horizon = int((symbols_cfg.get(sym) or {}).get("horizon_bars", 6))
        out[sym] = fit_symbol(sym, npz, horizon)
        print(f"[{sym}] fit ok n={out[sym]['fit_stats']['n']} price_change_r2={out[sym]['price_change']['r2']:.4f}")

    if not out:
        print("无可用数据，未写入系数")
        return 1

    save_causal_coef(out, _ROOT / args.out)
    print(f"已保存 → {_ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
