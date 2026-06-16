#!/usr/bin/env python3
"""在线元学习：每周微调 PPO 策略（Phase 6）。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.knowledge_net import KnowledgeNetInference
from zhulong.agent.training_utils import load_npz, load_training_config, resolve_symbol_paths
from zhulong.agent.trading_env import TradingEnv


def _build_env_df(data: dict) -> pd.DataFrame:
    return pd.DataFrame(
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD", "USOIL"])
    parser.add_argument("--config", default="config/config_agent.json")
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--lr-factor", type=float, default=0.1, help="学习率缩放（防遗忘）")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as ex:
        print("需要 stable_baselines3:", ex)
        return 1

    cfg = json.loads((_ROOT / args.config).read_text(encoding="utf-8-sig"))
    train_cfg = load_training_config(_ROOT / "config_training.yaml")
    paths = resolve_symbol_paths(args.symbol, train_cfg)
    model_path = paths["rl_model"]
    zip_path = Path(str(model_path) + ".zip")
    if not zip_path.is_file() and not model_path.is_file():
        print(f"缺少 RL 模型 {model_path}")
        return 1

    npz_path = paths["npz"]
    if not npz_path.is_file():
        print(f"缺少训练数据 {npz_path}")
        return 1

    data = load_npz(npz_path)
    tail_n = min(5000, len(data["close"]))
    for k in list(data.keys()):
        if isinstance(data[k], np.ndarray) and len(data[k]) > tail_n:
            data[k] = data[k][-tail_n:]

    kn = KnowledgeNetInference(paths["knowledge_model"], scaler_path=paths["knowledge_scaler"])
    struct = data["struct"]
    _, emb = kn.predict(struct)

    env_cfg = dict(cfg.get("trading_env") or {})
    env_cfg["point_cost"] = {args.symbol: paths["point_cost"]}
    env_cfg["counterfactual"] = cfg.get("counterfactual") or {"enabled": True}
    env_cfg["causal"] = (cfg.get("causal") or {}) | {"coef_path": "models/causal_coef.pkl"}
    env_cfg["meta_learning"] = {"enabled": False}

    scaler_path = _ROOT / "data" / f"agent_state_scaler_{args.symbol.lower()}.json"
    df = _build_env_df(data)

    def make_env():
        return TradingEnv(df, struct, emb, env_cfg, symbol=args.symbol.upper(), scaler_path=scaler_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    snap_dir = _ROOT / "models" / "snapshots" / args.symbol.lower()
    snap_dir.mkdir(parents=True, exist_ok=True)
    if zip_path.is_file():
        shutil.copy2(zip_path, snap_dir / f"rl_agent_{stamp}_pre.zip")

    model = PPO.load(str(model_path), env=DummyVecEnv([make_env]), device="cpu")
    base_lr = float(model.learning_rate if not callable(model.learning_rate) else model.learning_rate(1))
    model.learning_rate = base_lr * args.lr_factor
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)
    model.save(str(model_path))

    if zip_path.is_file():
        shutil.copy2(zip_path, snap_dir / f"rl_agent_{stamp}_post.zip")

    summary = {
        "symbol": args.symbol,
        "timesteps": args.timesteps,
        "lr": model.learning_rate,
        "snapshot_dir": str(snap_dir),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    log_path = _ROOT / "logs" / "meta" / f"weekly_finetune_{args.symbol}_{stamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"周度微调完成 → {model_path}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
