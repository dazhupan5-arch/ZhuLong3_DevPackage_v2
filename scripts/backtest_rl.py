#!/usr/bin/env python3
"""RL 智能体样本外回测（默认 2025 年）。"""

from __future__ import annotations

import torch  # noqa: F401 — Windows 下须最先加载

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.knowledge_net import KnowledgeNetInference
from zhulong.agent.rl_agent import RlAgent
from zhulong.agent.trading_env import TradingEnv
from zhulong.agent.training_utils import (
    ensure_logs_dir,
    filter_npz_by_year,
    load_npz,
    load_training_config,
    resolve_symbol_paths,
)


def _metrics(trades: list[dict], initial: float, equities: list[float]) -> dict:
    if not trades:
        return {"winrate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "total_return": 0.0, "trades": 0}
    pnls = [t["pnl_r"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]
    winrate = len(wins) / len(pnls)
    gross_win = sum(wins) if wins else 0.0
    gross_loss = sum(losses) if losses else 1e-9
    pf = gross_win / gross_loss
    peak = initial
    max_dd = 0.0
    for eq in equities:
        peak = max(peak, eq)
        dd = (peak - eq) / max(peak, 1e-9)
        max_dd = max(max_dd, dd)
    total_ret = (equities[-1] - initial) / initial if equities else 0.0
    return {
        "winrate": winrate,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "total_return": total_ret,
        "trades": len(pnls),
    }


def run_backtest(
    data: dict,
    model: RlAgent,
    struct,
    emb,
    env_cfg: dict,
    symbol: str,
    knowledge_probs=None,
) -> dict:
    df = pd.DataFrame(
        {
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
            "atr": data["atr"],
            "time": pd.to_datetime(data["time"]),
        }
    )
    scaler = env_cfg.get("_scaler_path")
    env = TradingEnv(
        df,
        struct,
        emb,
        env_cfg,
        symbol=symbol,
        scaler_path=scaler,
        knowledge_probs=knowledge_probs,
    )
    obs, _ = env.reset()
    equities = [env.equity]
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        equities.append(env.equity)
    return _metrics(env.trades, env.initial_balance, equities)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD", "USOIL"])
    parser.add_argument("--year", type=int, default=0)
    parser.add_argument("--npz", default=None)
    parser.add_argument("--config", default="config_training.yaml")
    args = parser.parse_args()

    cfg = load_training_config(_ROOT / args.config)
    paths = resolve_symbol_paths(args.symbol, cfg)
    npz_path = Path(args.npz) if args.npz else paths["npz"]
    if not npz_path.is_file():
        print(f"缺少 {npz_path}")
        return 1

    eval_year = args.year or int((cfg.get("backtest") or {}).get("eval_year", 2025))
    data = filter_npz_by_year(load_npz(npz_path), eval_year)
    if len(data["close"]) < 100:
        print(f"{eval_year} 年样本不足: {len(data['close'])} 行")
        return 1

    kn = KnowledgeNetInference(paths["knowledge_model"], scaler_path=paths["knowledge_scaler"])
    struct = data["struct"]
    kn_probs, emb = kn.predict(struct)
    scaler_path = _ROOT / "data" / f"agent_state_scaler_{args.symbol.lower()}.json"
    rl = RlAgent(paths["rl_model"], symbol=args.symbol)
    if not rl.is_ready:
        print(f"RL 模型未就绪: {paths['rl_model']}")
        return 1

    env_cfg = dict(cfg.get("env") or {})
    env_cfg["point_cost"] = {args.symbol.upper(): paths["point_cost"]}
    env_cfg["cost_scale"] = 1.0
    cog_align = dict((cfg.get("rl") or {}).get("cognition_align") or {})
    if cog_align:
        env_cfg["cognition_align"] = cog_align
    if scaler_path.is_file():
        env_cfg["_scaler_path"] = str(scaler_path)
    agent_cfg_path = _ROOT / "config" / "config_agent.json"
    if agent_cfg_path.is_file():
        agent_cfg = json.loads(agent_cfg_path.read_text(encoding="utf-8-sig"))
        te = agent_cfg.get("trading_env") or {}
        env_cfg["counterfactual"] = agent_cfg.get("counterfactual") or te.get("counterfactual") or {"enabled": False}
        env_cfg["causal"] = agent_cfg.get("causal") or te.get("causal") or {}
    stats = run_backtest(data, rl, struct, emb, env_cfg, args.symbol.upper(), kn_probs)

    bt_cfg = cfg.get("backtest") or {}
    min_trades = int(bt_cfg.get("min_trades", 10))
    passed = (
        stats["trades"] >= min_trades
        and stats["winrate"] >= float(bt_cfg.get("min_winrate", 0.5))
        and stats["profit_factor"] >= float(bt_cfg.get("min_profit_factor", 1.3))
        and stats["max_drawdown"] <= float(bt_cfg.get("max_drawdown", 0.25))
    )

    report = {"symbol": args.symbol.upper(), "year": eval_year, **stats, "passed": passed}
    out = ensure_logs_dir() / f"backtest_{args.symbol.upper()}_{eval_year}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== {args.symbol} {eval_year} 回测 ===")
    print(f"交易次数: {stats['trades']}")
    print(f"胜率: {stats['winrate']:.2%}")
    print(f"盈亏比: {stats['profit_factor']:.2f}")
    print(f"最大回撤: {stats['max_drawdown']:.2%}")
    print(f"总收益: {stats['total_return']:.2%}")
    print(f"验收: {'PASS' if passed else 'FAIL'}")
    print(f"报告: {out}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
