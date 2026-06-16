#!/usr/bin/env python3
"""KN 2.0 端到端训练 —— GRU 认知叙事网络 + PPO 策略优化。

训练流程：
  1. 加载 M5 历史数据
  2. 构建 V14(68维) + struct(30维) = 98维市场特征
  3. 生成 Triple Barrier 训练标签
  4. 在 TradingEnvKN2 中端到端训练 GRU 策略网络
  5. 导出模型用于推理

训练目标：
  - 阶段 1：监督预训练（用 Triple Barrier 标签训练 GRU 的多任务头）
  - 阶段 2：PPO 端到端优化（最大化交易 ROI，最小化最大回撤）

安全机制：
  - 只在历史数据上训练（2024 年及以前）
  - 验证集：2025 年数据
  - 回退：模型不覆盖现有 kn2_trader.pth，而是输出到 kn2_trader_{timestamp}.pth
"""

from __future__ import annotations

import torch  # noqa: F401 — 必须在 sklearn 之前导入

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.knowledge_net_kn2 import (
    KN2Inference,
    _build_trader_gru_class,
    _ensure_torch,
    build_triple_barrier_labels,
    train_kn2_end_to_end,
)
from zhulong.agent.trading_env_kn2 import (
    TradingEnvKN2,
    generate_kn2_training_labels,
    train_kn2_with_ppo,
)
from zhulong.agent.training_utils import (
    ensure_logs_dir,
    load_npz,
    load_training_config,
    resolve_symbol_paths,
)
from zhulong.strategies.indicators import atr_series
from zhulong.training.lgb.features import FEATURE_COLUMNS_LGB_V13, compute_features
from zhulong.utils.device import print_gpu_status


def build_market_features(
    data: pd.DataFrame,
    struct_features: np.ndarray,
) -> np.ndarray:
    """构建 98 维市场特征：V14(68) + struct(30)。

    Args:
        data: M5 K线数据 (columns: open, high, low, close, volume, time)
        struct_features: 结构特征 (n_bars, 30+)

    Returns:
        (n_bars, 98) 市场特征数组
    """
    n = len(data)
    market_feat = np.zeros((n, 98), dtype=np.float32)

    # 分批计算 V14 特征（避免内存峰值）
    chunk_size = 5000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_data = data.iloc[:end].copy()
        try:
            feats = compute_features(chunk_data, include_mtf=True, include_reversal=True)
            cols = [c for c in FEATURE_COLUMNS_LGB_V13 if c in feats.columns]
            chunk_arr = feats[cols].iloc[start:end].to_numpy(dtype=np.float32)

            # 补齐缺失列
            if chunk_arr.shape[1] < 68:
                padded = np.zeros((chunk_arr.shape[0], 68), dtype=np.float32)
                padded[:, :chunk_arr.shape[1]] = chunk_arr
                chunk_arr = padded
            market_feat[start:end, :68] = chunk_arr[:, :68]
        except Exception as ex:
            print(f"  WARN: V14特征计算失败 [{start}:{end}]: {ex}")
            # 零填充

    # 拼接结构特征
    struct = np.asarray(struct_features, dtype=np.float32)[:n, :30]
    if struct.shape[1] < 30:
        padded = np.zeros((struct.shape[0], 30), dtype=np.float32)
        padded[:, :struct.shape[1]] = struct
        struct = padded
    market_feat[:, 68:98] = struct

    # 填充 NaN
    np.nan_to_num(market_feat, nan=0.0, posinf=10.0, neginf=-10.0, copy=False)

    return market_feat


def build_position_states(data: pd.DataFrame, market_features: np.ndarray) -> np.ndarray:
    """为历史数据构建伪持仓状态（全零，无真实持仓）。"""
    n = len(data)
    from zhulong.agent.knowledge_net_kn2 import encode_position_state

    return np.tile(encode_position_state(), (n, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="KN 2.0 端到端训练")
    parser.add_argument("--symbol", default="XAUUSD", choices=["XAUUSD", "USOIL"])
    parser.add_argument("--npz", default=None, help="训练数据 npz 路径")
    parser.add_argument("--config", default="config_training.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--ppo-steps", type=int, default=50000)
    parser.add_argument("--phase", default="both",
                        choices=["supervised", "ppo", "both"],
                        help="训练阶段: supervised(监督预训练) / ppo(PPO优化) / both(全部)")
    parser.add_argument("--output", default="models/kn2_trader.pth")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print("=" * 60)
    print("KN 2.0 端到端训练")
    print(f"  品种: {args.symbol}")
    print(f"  阶段: {args.phase}")
    print(f"  GRU: hidden={args.hidden_dim} layers={args.num_layers} embed={args.embed_dim}")
    print(f"  设备: {args.device}")
    print("=" * 60)

    print_gpu_status()

    # ---- 1. 加载数据 ----
    print("\n[1/6] 加载训练数据...")
    cfg = load_training_config(_ROOT / args.config)
    paths = resolve_symbol_paths(args.symbol, cfg)
    npz_path = Path(args.npz) if args.npz else paths.get("npz")

    if not npz_path or not npz_path.is_file():
        print(f"错误: 训练数据不存在 {npz_path}")
        print("请先运行 prepare_training_data.py 或指定 --npz 路径")
        return 1

    data = load_npz(npz_path)
    print(f"  数据加载: {list(data.keys())}")
    n_bars = len(data.get("time", data.get("close", [])))
    print(f"  总 bar 数: {n_bars:,}")

    # 构建 DataFrame
    df_cols = {}
    for col in ["open", "high", "low", "close", "volume", "time"]:
        if col in data:
            df_cols[col] = data[col]
    if "time" not in df_cols:
        df_cols["time"] = pd.date_range("2020-01-01", periods=n_bars, freq="5min")
    df = pd.DataFrame(df_cols)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()

    # 分年切片
    train_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    val_year = 2025

    train_mask = df.index.year.isin(train_years)
    val_mask = df.index.year == val_year

    print(f"  训练集: {train_mask.sum():,} bars ({', '.join(map(str, train_years))})")
    print(f"  验证集: {val_mask.sum():,} bars ({val_year})")

    if train_mask.sum() < 1000:
        print("错误: 训练数据不足，需要至少 1000 条 bar")
        return 1

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    # ---- 2. 构建市场特征 ----
    print("\n[2/6] 构建 98 维市场特征...")
    t0 = time.perf_counter()

    struct_feat = data.get("struct", np.zeros((n_bars, 30), dtype=np.float32))
    struct_train = struct_feat[train_mask.values] if hasattr(train_mask, 'values') else struct_feat[train_mask.to_numpy()]
    struct_val = struct_feat[val_mask.values] if hasattr(val_mask, 'values') else struct_feat[val_mask.to_numpy()]

    train_market = build_market_features(train_df, struct_train)
    val_market = build_market_features(val_df, struct_val)

    print(f"  训练特征: {train_market.shape}")
    print(f"  验证特征: {val_market.shape}")
    print(f"  耗时: {time.perf_counter() - t0:.1f}s")

    # ---- 3. 生成训练标签 ----
    print("\n[3/6] 生成 Triple Barrier 训练标签...")
    t0 = time.perf_counter()

    # 计算 ATR
    atr_train = atr_series(train_df).bfill().fillna(train_df["close"] * 0.001).values
    atr_val = atr_series(val_df).bfill().fillna(val_df["close"] * 0.001).values

    train_labels = generate_kn2_training_labels(
        train_df.assign(atr=atr_train),
        train_market,
        tp_atr_mult=2.0,
        sl_atr_mult=1.5,
        max_hold_bars=48,
    )
    val_labels = generate_kn2_training_labels(
        val_df.assign(atr=atr_val),
        val_market,
        tp_atr_mult=2.0,
        sl_atr_mult=1.5,
        max_hold_bars=48,
    )

    pos_train = build_position_states(train_df, train_market)
    pos_val = build_position_states(val_df, val_market)

    print(f"  标签分布: {np.bincount(train_labels['action'], minlength=6)}")
    should_trade_pct = train_labels["should_trade"].mean() * 100
    print(f"  should_trade 比例: {should_trade_pct:.1f}%")
    print(f"  耗时: {time.perf_counter() - t0:.1f}s")

    # ---- 4. 监督预训练 ----
    if args.phase in ("supervised", "both"):
        print("\n[4/6] 监督预训练（多任务 Triple Barrier）...")
        print(f"  训练参数: epochs={args.epochs} lr={args.lr} batch={args.batch_size}")

        out_path = Path(args.output)
        result = train_kn2_end_to_end(
            market_features=train_market,
            position_states=pos_train,
            targets={
                "action": train_labels["action"],
                "position_size": train_labels["position_size"],
                "sl_atr_mult": train_labels["sl_atr_mult"],
                "tp_atr_mult": train_labels["tp_atr_mult"],
                "should_trade": train_labels["should_trade"],
            },
            val_ratio=0.1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            embed_dim=args.embed_dim,
            out_path=out_path,
            device=args.device,
            sequence_length=args.seq_len,
        )
        print(f"  监督训练完成: val_loss={result['val_loss']:.4f}")

    # ---- 5. PPO 端到端优化 ----
    if args.phase in ("ppo", "both"):
        print("\n[5/6] PPO 端到端优化...")

        torch_mod, _ = _ensure_torch()
        KnCls, _ = _build_trader_gru_class(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            embed_dim=args.embed_dim,
        )

        model = KnCls()
        pth_path = Path(args.output)
        if args.phase == "ppo" and pth_path.is_file():
            state_dict = torch_mod.load(pth_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print(f"  加载预训练权重: {pth_path}")
        elif args.phase == "both":
            # 刚完成监督训练，权重已保存
            state_dict = torch_mod.load(pth_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print(f"  加载监督预训练权重: {pth_path}")

        device_obj = torch_mod.device(
            args.device if args.device != "auto" else
            ("cuda" if torch_mod.cuda.is_available() else "cpu")
        )
        model = model.to(device_obj)
        model.train()

        # 构建训练环境
        env_config = {
            "initial_balance": 10000,
            "slippage": 0.1,
            "stop_loss_atr_mult": 1.5,
            "take_profit_atr_mult": 2.0,
            "max_hold_bars": 48,
            "hold_penalty": 0.0001,
            "max_trades_per_episode": 20,
            "drawdown_penalty_coef": 0.5,
            "drawdown_penalty_trigger": 0.05,
            "unrealized_reward_scale": 0.01,
            "close_profit_bonus": 0.1,
            "profit_protect_pct": 0.5,
            "trail_stop_atr": 1.0,
            "point_cost": {"XAUUSD": 0.2, "USOIL": 0.03},
            "trader_memory": {"max_len": 20},
        }

        env = TradingEnvKN2(
            train_df.assign(atr=atr_train),
            train_market,
            pos_train,
            config=env_config,
            symbol=args.symbol,
        )

        print(f"  PPO 训练步数: {args.ppo_steps}")
        result = train_kn2_with_ppo(
            env,
            model,
            total_timesteps=args.ppo_steps,
            sequence_length=args.seq_len,
            log_interval=1000,
        )

        # 保存 PPO 微调后的模型
        ppo_out = pth_path.with_name(pth_path.stem + "_ppo.pth")
        torch_mod.save(model.state_dict(), ppo_out)
        print(f"  PPO 优化完成: {result['avg_reward']:.4f}")
        print(f"  模型保存: {ppo_out}")

    # ---- 6. 验证 ----
    print("\n[6/6] 验证模型...")

    out_path = Path(args.output)
    if args.phase == "ppo":
        out_path = out_path.with_name(out_path.stem + "_ppo.pth")

    if out_path.is_file():
        kn2 = KN2Inference(out_path)
        if kn2.is_ready:
            print(f"  模型加载成功: {out_path}")
            print(f"  隐藏维度: {kn2.hidden_dim}")
            print(f"  GRU 层数: {kn2.num_layers}")
            print(f"  嵌入维度: {kn2.embed_dim}")

            # 快速推理测试
            test_mf = val_market[:5]
            for i in range(min(5, len(test_mf))):
                decision = kn2.predict(test_mf[i])
                print(f"  bar {i}: action={decision['action_name']} "
                      f"conf={decision['confidence']:.3f} "
                      f"size={decision['position_size']:.2f} "
                      f"trade={decision['should_trade']}")
        else:
            print(f"  警告: 模型加载失败，回退到启发式模式")
    else:
        print(f"  警告: 模型文件不存在: {out_path}")

    print("\n" + "=" * 60)
    print("KN 2.0 训练完成！")
    print(f"  模型路径: {out_path}")
    print(f"  下一步: 将 config_agent.json 中 kn2.model_path 设为 {out_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
