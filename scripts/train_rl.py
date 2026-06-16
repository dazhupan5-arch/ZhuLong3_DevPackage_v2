#!/usr/bin/env python3
"""PPO 强化学习训练。"""

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

from zhulong.agent.state_builder import StateBuilder
from zhulong.agent.trading_env import TradingEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/rl_features.npz")
    parser.add_argument("--config", default="config/config_agent.json")
    parser.add_argument("--out", default="models/rl_agent")
    parser.add_argument("--timesteps", type=int, default=0)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as ex:
        print("需要 stable-baselines3:", ex)
        return 1

    cfg = json.loads((_ROOT / args.config).read_text(encoding="utf-8-sig"))
    env_cfg = cfg.get("trading_env") or {}
    rl_cfg = cfg.get("rl") or {}
    train_cfg = rl_cfg.get("training") or {}
    steps = args.timesteps or int(train_cfg.get("total_timesteps", 500000))

    data = np.load(_ROOT / args.npz, allow_pickle=True)
    n = len(data["close"])
    df = pd.DataFrame(
        {
            "open": data["close"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": np.ones(n),
            "atr": data["atr"],
            "time": pd.to_datetime(data["time"]),
        }
    )
    struct = data["struct"]
    emb = data["emb"]

    scaler_path = _ROOT / "data/agent_state_scaler.json"
    # StateBuilder.build() uses struct[:30] + emb[:32] + account(12) = 74 dims
    raw = np.concatenate([struct[:, :30], emb[:, :32], np.zeros((n, 12), dtype=np.float32)], axis=1)
    sb = StateBuilder()
    sb.save_scaler(raw[: min(2000, n)], scaler_path)

    def make_env():
        return TradingEnv(df, struct, emb, env_cfg, scaler_path=str(scaler_path))

    env = DummyVecEnv([make_env])
    policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        n_steps=int(train_cfg.get("n_steps", 2048)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        n_epochs=int(train_cfg.get("n_epochs", 10)),
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(_ROOT / "logs"),
    )
    model.learn(total_timesteps=steps)
    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
