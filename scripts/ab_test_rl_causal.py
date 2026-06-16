#!/usr/bin/env python3
"""RL 基线 vs 因果奖励 A/B 回测。"""

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

sys.path.insert(0, str(_ROOT / "scripts"))
from backtest_rl import run_backtest  # noqa: E402
from zhulong.agent.knowledge_net import KnowledgeNetInference
from zhulong.agent.rl_agent import RlAgent
from zhulong.agent.training_utils import filter_npz_by_year, load_npz, load_training_config, resolve_symbol_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", default="logs/rl/ab_test_causal.json")
    args = parser.parse_args()

    cfg = load_training_config(_ROOT / "config_training.yaml")
    paths = resolve_symbol_paths(args.symbol, cfg)
    data = filter_npz_by_year(load_npz(paths["npz"]), args.year)
    if len(data.get("close", [])) < 100:
        print("样本不足")
        return 1

    kn = KnowledgeNetInference(paths["knowledge_model"], scaler_path=paths["knowledge_scaler"])
    struct = data["struct"]
    _, emb = kn.predict(struct)
    model = RlAgent(paths["rl_model"], symbol=args.symbol.upper())
    if not model.is_ready:
        print("RL 模型未就绪")
        return 1

    base_env = {
        "initial_balance": 10000,
        "hold_penalty": 0.001,
        "point_cost": {args.symbol.upper(): paths["point_cost"]},
        "counterfactual": {"enabled": False},
    }
    causal_env = dict(base_env)
    causal_env["counterfactual"] = {"enabled": True}
    causal_env["causal"] = {"coef_path": "models/causal_coef.pkl"}

    baseline = run_backtest(data, model, struct, emb, base_env, args.symbol.upper())
    # 因果奖励仅影响训练；回测用同一策略比较决策路径
    causal = run_backtest(data, model, struct, emb, causal_env, args.symbol.upper())

    report = {
        "symbol": args.symbol,
        "year": args.year,
        "baseline": baseline,
        "causal_env_metrics": causal,
        "note": "causal_env_metrics 使用因果奖励环境重放同一策略",
    }
    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
