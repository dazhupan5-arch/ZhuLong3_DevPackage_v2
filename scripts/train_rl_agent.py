#!/usr/bin/env python3
"""PPO 强化学习智能体训练。"""

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
from zhulong.agent.state_builder import STATE_DIM, StateBuilder
from zhulong.agent.trading_env import ForcedOpenExplorationWrapper, TradingEnv
from zhulong.agent.training_utils import (
    ensure_logs_dir,
    filter_npz_by_year,
    load_npz,
    load_training_config,
    resolve_symbol_paths,
    resolve_v16_paths,
)
from zhulong.utils.device import print_gpu_status, resolve_sb3_device


def _build_env_df(data: dict) -> pd.DataFrame:
    n = len(data["close"])
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


def _mask_npz(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    n = len(mask)
    out: dict[str, np.ndarray] = {}
    for k, v in data.items():
        arr = np.asarray(v)
        if k == "symbol" or arr.shape[0] != n:
            out[k] = arr
        else:
            out[k] = arr[mask]
    return out


def _train_eval_slices(
    data: dict[str, np.ndarray],
    *,
    train_through_year: int,
    eval_bars: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    times = pd.to_datetime(data["time"])
    train_mask = np.asarray(times.year <= train_through_year)
    train = _mask_npz(data, train_mask)
    eval_source = filter_npz_by_year(data, train_through_year)
    if len(eval_source.get("close", [])) > eval_bars:
        eval_source = {k: v[-eval_bars:] if np.asarray(v).shape[0] == len(eval_source["close"]) else v for k, v in eval_source.items()}
    if len(eval_source.get("close", [])) < 500:
        tail = min(eval_bars, len(train["close"]))
        eval_source = {k: v[-tail:] if np.asarray(v).shape[0] == len(train["close"]) else v for k, v in train.items()}
    print(
        f"RL train rows={len(train['close'])} (≤{train_through_year}), "
        f"eval rows={len(eval_source['close'])}"
    )
    return train, eval_source


def _make_wrapped_env(
    df: pd.DataFrame,
    struct: np.ndarray,
    emb: np.ndarray,
    shocks: np.ndarray,
    env_cfg: dict,
    symbol: str,
    scaler_path: Path,
    *,
    max_episode_steps: int,
    force_explore: bool,
    expl_cfg: dict,
    knowledge_probs: np.ndarray | None = None,
):
    from gymnasium.wrappers import TimeLimit

    env = TradingEnv(
        df,
        struct,
        emb,
        env_cfg,
        symbol=symbol,
        scaler_path=str(scaler_path),
        exogenous_shocks=shocks,
        knowledge_probs=knowledge_probs,
    )
    if force_explore and expl_cfg.get("enabled", True):
        env = ForcedOpenExplorationWrapper(
            env,
            explore_steps=int(expl_cfg.get("steps", 20000)),
            open_prob=float(expl_cfg.get("prob", 0.3)),
        )
    return TimeLimit(env, max_episode_steps=max_episode_steps)


def _unwrap_trading_env(env) -> TradingEnv:
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD", "USOIL"])
    parser.add_argument("--npz", default=None)
    parser.add_argument("--config", default="config_training.yaml")
    parser.add_argument("--timesteps", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="快速冒烟（5000 步）")
    parser.add_argument("--v16", action="store_true", help="使用 horizon_v16 NPZ + ONNX（V16 正式 PPO）")
    parser.add_argument(
        "--device",
        default="",
        help="覆盖 config_training.yaml device.rl：auto|cuda|cpu",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as ex:
        print("需要 stable-baselines3:", ex)
        return 1

    cfg = load_training_config(_ROOT / args.config)
    print_gpu_status()
    paths = resolve_symbol_paths(args.symbol, cfg)
    dev_cfg = cfg.get("device") or {}
    rl_device_pref = str(args.device or dev_cfg.get("rl", "auto"))
    rl_device = resolve_sb3_device(rl_device_pref)
    print(f"PPO device: {rl_device} (pref={rl_device_pref})", flush=True)

    if args.v16:
        v16 = resolve_v16_paths(args.symbol, cfg)
        npz_path = Path(v16["horizon_npz"])
        kn_path = Path(v16["horizon_onnx"])
        kn_scaler = Path(v16["horizon_scaler"])
        if not kn_path.is_file():
            kn_path = Path(v16["horizon_pth"])
        rl_out = Path(v16["rl_model"])
        paths = {**paths, "npz": npz_path, "knowledge_model": kn_path, "knowledge_scaler": kn_scaler, "rl_model": rl_out}
        print(f"V16 PPO: {args.symbol.upper()} horizon NPZ + horizon model", flush=True)
    else:
        npz_path = Path(args.npz) if args.npz else paths["npz"]

    if not npz_path.is_file():
        print(f"缺少 {npz_path}")
        return 1

    kn_path = paths["knowledge_model"]
    kn_scaler = paths["knowledge_scaler"]
    if not kn_path.is_file():
        hint = "请先 train_horizon_v16.py + convert ONNX" if args.v16 else "请先 train_knowledge_net.py"
        print(f"缺少模型 {kn_path}，{hint}")
        return 1

    data = load_npz(npz_path)
    if args.v16 and "open" not in data:
        print("V16 NPZ 缺少 OHLCV，请先运行 scripts/enrich_horizon_v16_npz.py")
        return 1
    rl_cfg = cfg.get("rl") or {}
    sym = args.symbol.upper()
    if sym == "USOIL":
        sym_rl = rl_cfg.get("oil") or {}
    elif sym == "XAUUSD":
        sym_rl = rl_cfg.get("xau") or {}
    else:
        sym_rl = {}
    bt_cfg = cfg.get("backtest") or {}
    train_through = int(rl_cfg.get("train_through_year", bt_cfg.get("eval_year", 2025) - 1))
    eval_bars = int(rl_cfg.get("eval_bars", 15000))
    max_episode_steps = int(rl_cfg.get("max_episode_steps", 5000))
    train_data, eval_data = _train_eval_slices(
        data,
        train_through_year=train_through,
        eval_bars=eval_bars,
    )

    struct = train_data["struct"]
    kn = KnowledgeNetInference(kn_path, scaler_path=kn_scaler)
    num_train = len(struct)
    kn_probs_list = []
    emb_list = []
    chunk = 50000
    for start in range(0, num_train, chunk):
        end = min(start + chunk, num_train)
        p, e = kn.predict(struct[start:end])
        kn_probs_list.append(p)
        emb_list.append(e)
    kn_probs = np.concatenate(kn_probs_list, axis=0)
    emb = np.concatenate(emb_list, axis=0)

    df = _build_env_df(train_data)
    eval_struct = eval_data["struct"]
    num_eval = len(eval_struct)
    eval_probs_list = []
    eval_emb_list = []
    for start in range(0, num_eval, chunk):
        end = min(start + chunk, num_eval)
        p, e = kn.predict(eval_struct[start:end])
        eval_probs_list.append(p)
        eval_emb_list.append(e)
    eval_probs = np.concatenate(eval_probs_list, axis=0)
    eval_emb = np.concatenate(eval_emb_list, axis=0)
    eval_df = _build_env_df(eval_data)
    env_yaml = dict(cfg.get("env") or {})
    env_cfg = dict(env_yaml)
    env_cfg["point_cost"] = {args.symbol.upper(): paths["point_cost"]}
    env_cfg["trader_memory"] = {"max_len": 20}
    if sym_rl.get("open_reward_bonus") is not None:
        env_cfg["open_reward_bonus"] = float(sym_rl["open_reward_bonus"])
    cog_align = dict(rl_cfg.get("cognition_align") or {})
    cog_align.update(sym_rl.get("cognition_align") or {})
    if cog_align:
        env_cfg["cognition_align"] = cog_align
    agent_cfg_path = _ROOT / "config" / "config_agent.json"
    if agent_cfg_path.is_file():
        agent_cfg = json.loads(agent_cfg_path.read_text(encoding="utf-8-sig"))
        te = agent_cfg.get("trading_env") or {}
        if "counterfactual" not in env_yaml:
            env_cfg["counterfactual"] = agent_cfg.get("counterfactual") or te.get("counterfactual") or {"enabled": False}
        if "causal" not in env_yaml:
            env_cfg["causal"] = agent_cfg.get("causal") or te.get("causal") or {}
        if "meta_learning" not in env_yaml:
            env_cfg["meta_learning"] = agent_cfg.get("meta_learning") or te.get("meta_learning") or {"enabled": False}
        if args.v16 and "execution_parity" not in env_yaml:
            env_cfg["execution_parity"] = te.get("execution_parity") or {
                "enabled": True,
                "entry_quality_bonus": 0.05,
                "pending_expire_bars": 48,
                "pending_expire_penalty": 0.01,
            }
    shocks = np.clip(struct[:, 0] if struct.shape[1] else np.zeros(len(struct)), -3, 3).astype(np.float32)
    eval_shocks = np.clip(
        eval_struct[:, 0] if eval_struct.shape[1] else np.zeros(len(eval_struct)),
        -3,
        3,
    ).astype(np.float32)

    steps = args.timesteps or int(rl_cfg.get("total_timesteps", 500000))
    if args.quick:
        steps = 5000

    scaler_path = _ROOT / "data" / f"agent_state_scaler_{args.symbol.lower()}.json"
    from zhulong.agent.state_builder import encode_cognition_features, infer_regime_from_struct

    cog_tail = []
    for i in range(min(len(struct), 5000)):
        regime = infer_regime_from_struct(struct[i])
        probs = kn_probs[i] if i < len(kn_probs) else None
        conf = float(probs[2]) if probs is not None and len(probs) >= 3 else 0.0
        cog_tail.append(encode_cognition_features(probs, regime, conf, conf >= 0.42))
    cog_mat = np.stack(cog_tail, axis=0) if cog_tail else np.zeros((0, STATE_DIM - 74), dtype=np.float32)
    raw_base = np.concatenate([struct[:, :30], emb[:, :32], np.zeros((len(struct), 12), dtype=np.float32)], axis=1)
    raw = np.concatenate([raw_base[: len(cog_mat)], cog_mat], axis=1)
    sb = StateBuilder()
    sb.save_scaler(raw[: min(5000, len(raw))], scaler_path)

    expl_cfg = {**(rl_cfg.get("forced_explore") or {}), **(sym_rl.get("forced_explore") or {})}

    def make_train_env():
        return _make_wrapped_env(
            df,
            struct,
            emb,
            shocks,
            env_cfg,
            args.symbol.upper(),
            scaler_path,
            max_episode_steps=max_episode_steps,
            force_explore=True,
            expl_cfg=expl_cfg,
            knowledge_probs=kn_probs,
        )

    def make_eval_env():
        return _make_wrapped_env(
            eval_df,
            eval_struct,
            eval_emb,
            eval_shocks,
            env_cfg,
            args.symbol.upper(),
            scaler_path,
            max_episode_steps=max_episode_steps,
            force_explore=False,
            expl_cfg=expl_cfg,
            knowledge_probs=eval_probs,
        )

    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])

    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    log_dir = _ROOT / "logs" / "rl" / args.symbol.lower()
    log_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device=rl_device,
        learning_rate=float(rl_cfg.get("learning_rate", 3e-4)),
        n_steps=int(rl_cfg.get("n_steps", 2048)),
        batch_size=int(rl_cfg.get("batch_size", 64)),
        n_epochs=int(rl_cfg.get("n_epochs", 10)),
        gamma=float(rl_cfg.get("gamma", 0.99)),
        gae_lambda=float(rl_cfg.get("gae_lambda", 0.95)),
        clip_range=float(rl_cfg.get("clip_range", 0.2)),
        ent_coef=float(rl_cfg.get("ent_coef", 0.01)),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
    )

    eval_freq = max(int(rl_cfg.get("eval_freq", 10000)), steps // 10 if args.quick else 10000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(paths["rl_model"].parent),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    ent_start = float(sym_rl.get("ent_coef_start", rl_cfg.get("ent_coef_start", rl_cfg.get("ent_coef_high", rl_cfg.get("ent_coef", 0.1)))))
    ent_end = float(sym_rl.get("ent_coef_end", rl_cfg.get("ent_coef_end", rl_cfg.get("ent_coef_low", 0.01))))

    try:
        from stable_baselines3.common.callbacks import BaseCallback

        class EntCoefLinearCallback(BaseCallback):
            """ent_coef 从 ent_start 线性衰减到 ent_end。"""

            def __init__(self, total_steps: int, start: float, end: float):
                super().__init__()
                self.total_steps = max(total_steps, 1)
                self.start = start
                self.end = end

            def _on_step(self) -> bool:
                progress_remaining = 1.0 - min(self.num_timesteps / self.total_steps, 1.0)
                model.ent_coef = self.end + (self.start - self.end) * progress_remaining
                return True

        ent_callback = EntCoefLinearCallback(steps, ent_start, ent_end)

        metrics_freq = int(rl_cfg.get("metrics_freq", eval_freq))
        metrics_log = ensure_logs_dir() / f"rl_metrics_{args.symbol.upper()}.jsonl"

        class TradingMetricsCallback(BaseCallback):
            def __init__(self, vec_eval_env, frequency: int, log_file: Path):
                super().__init__()
                self.vec_eval_env = vec_eval_env
                self.frequency = frequency
                self.log_file = log_file

            def _on_step(self) -> bool:
                if self.num_timesteps == 0 or self.num_timesteps % self.frequency != 0:
                    return True
                self._log_metrics()
                return True

            def _log_metrics(self) -> None:
                trades_all: list[dict] = []
                trades_last_ep = 0
                max_dd = 0.0
                for _ in range(3):
                    obs = self.vec_eval_env.reset()
                    done = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, _, dones, _ = self.vec_eval_env.step(action)
                        done = bool(dones[0])
                    inner = _unwrap_trading_env(self.vec_eval_env.envs[0])
                    trades_all.extend(inner.trades)
                    trades_last_ep = len(inner.trades)
                    max_dd = max(max_dd, float(inner._max_dd_since_reward))

                recent = trades_all[-100:]
                wins = [t for t in recent if float(t.get("pnl_r", 0)) > 0]
                win_rate = len(wins) / len(recent) if recent else 0.0
                holds = [
                    max(int(t.get("exit_step", 0)) - int(t.get("entry_step", 0)), 1)
                    for t in recent
                    if "entry_step" in t
                ]
                avg_hold = sum(holds) / len(holds) if holds else 0.0
                record = {
                    "timesteps": int(self.num_timesteps),
                    "win_rate_recent": round(win_rate, 4),
                    "avg_hold_bars": round(avg_hold, 2),
                    "max_drawdown": round(max_dd, 4),
                    "trades_this_episode": trades_last_ep,
                    "trades_sampled": len(recent),
                }
                with self.log_file.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[metrics @ {self.num_timesteps}] {record}")

                if win_rate < 0.35 and len(recent) >= 20:
                    print(
                        f"WARN: win_rate {win_rate:.2%} < 35% — "
                        "consider raising ent_coef to 0.05"
                    )

        metrics_callback = TradingMetricsCallback(eval_env, metrics_freq, metrics_log)
        callbacks = [eval_callback, ent_callback, metrics_callback]
    except ImportError:
        callbacks = [eval_callback]

    model.learn(total_timesteps=steps, callback=callbacks)
    out = paths["rl_model"]
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))

    summary = {
        "symbol": args.symbol.upper(),
        "timesteps": steps,
        "model": str(out) + ".zip",
        "state_scaler": str(scaler_path),
        "knowledge_model": str(kn_path),
        "architecture": "v16" if args.v16 else "legacy",
    }
    summary_path = ensure_logs_dir() / f"rl_{args.symbol.upper()}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.v16:
        v16 = resolve_v16_paths(args.symbol, cfg)
        v16_dir = Path(v16["rl_meta"]).parent
        v16_dir.mkdir(parents=True, exist_ok=True)
        Path(v16["rl_meta"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"RL 训练完成 → {out}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
