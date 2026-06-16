# 烛龙交易智能体 · 训练手册

本手册对应 Cursor 训练方案，覆盖 **KnowledgeNet** 与 **PPO** 全流程。

## 前置条件

1. Python 依赖（含 PyTorch、stable-baselines3、gymnasium、PyYAML）：
   ```powershell
   py -3 -m pip install -r requirements.txt
   ```
2. M5 历史 CSV（MT5 导出）：
   ```powershell
   py -3 scripts/export_m5_mt5.py --symbol XAUUSD --timeframe M5
   py -3 scripts/export_m5_mt5.py --symbol USOIL --timeframe M5
   ```
   默认读取：
   - `data/training/XAUUSD_M5.csv`
   - `data/training/lgb/USOIL/USOIL_M5.csv`（无表头：`date,time,OHLCV` 格式自动识别）

> **注意**：USOIL 全量 M5（70 万+ 根）结构特征预计算可能需要 **数小时**。可先用 `--max-rows 50000` 冒烟，或缩小 `--start/--end` 区间。

3. 配置文件：`config_training.yaml`

## 一键流水线

```powershell
# 黄金完整训练（prepare → knowledge → PPO → backtest）
.\scripts\run_agent_training.ps1 -Symbol XAUUSD

# 原油
.\scripts\run_agent_training.ps1 -Symbol USOIL

# 快速冒烟（PPO 仅 5000 步）
.\scripts\run_agent_training.ps1 -Symbol XAUUSD -Quick
```

## 分步执行

### 1. 离线预计算

```powershell
py -3 scripts/prepare_training_data.py --symbol XAUUSD --start 2016-01-01 --end 2025-12-31
py -3 scripts/prepare_training_data.py --symbol USOIL
```

输出：
- `data/training_data.npz`（黄金）
- `data/oil_training_data.npz`（原油）

包含字段：`time, open, high, low, close, volume, atr, struct(30), labels(-1/0/1)`。

### 2. 训练 KnowledgeNet

```powershell
py -3 scripts/train_knowledge_net.py --symbol XAUUSD
py -3 scripts/train_knowledge_net.py --symbol USOIL
```

输出：
| 品种 | 模型 | Scaler |
|------|------|--------|
| XAUUSD | `models/knowledge_net.pth` | `models/knowledge_scaler.pkl` |
| USOIL | `models/knowledge_net_oil.pth` | `models/knowledge_scaler_oil.pkl` |

**验收标准**（脚本末尾打印）：
- 验证集准确率 ≥ 60%
- 模型 < 1 MB
- CPU 单样本推理 < 1 ms

日志：`logs/training/knowledge_XAUUSD.log`

### 3. 训练 PPO

```powershell
py -3 scripts/train_rl_agent.py --symbol XAUUSD
py -3 scripts/train_rl_agent.py --symbol USOIL
```

输出：
- `models/rl_agent_xau.zip`
- `models/rl_agent_oil.zip`
- TensorBoard：`logs/rl/xauusd/`、`logs/rl/usoil/`

默认 `total_timesteps=500000`（约 2–4 小时，视 CPU 而定）。

### 4. 回测评估

```powershell
py -3 scripts/backtest_rl.py --symbol XAUUSD --year 2025
py -3 scripts/backtest_rl.py --symbol USOIL --year 2025
```

**验收标准**（`config_training.yaml` → `backtest`）：
- 胜率 ≥ 50%
- 盈亏比 ≥ 1.3
- 最大回撤 ≤ 25%

报告：`logs/training/backtest_XAUUSD_2025.json`

## 推理集成

训练完成后，在 `config/config_agent.json` 中指向对应模型：

```json
"knowledge_net": {
  "model_path": "models/knowledge_net.pth",
  "scaler_path": "models/knowledge_scaler.pkl"
},
"rl": {
  "model_path_xau": "models/rl_agent_xau",
  "model_path_oil": "models/rl_agent_oil"
}
```

WinUI 设置页启用 **RL 交易智能体** 即可实机 tick。

命令行实机：

```powershell
py -3 scripts/realtime_signal.py --agent --agent-config config/config_agent.json
```

## 模块索引

| 文件 | 说明 |
|------|------|
| `zhulong/agent/structure_analyzer.py` | 30 维结构特征 |
| `zhulong/agent/knowledge_net.py` | 知识网络 + 训练 |
| `zhulong/agent/trading_env.py` | Gym 环境 |
| `zhulong/agent/rl_agent.py` | PPO 推理封装 |
| `zhulong/agent/trading_agent.py` | 实机 tick |
| `scripts/prepare_training_data.py` | 数据包 |
| `scripts/train_knowledge_net.py` | 监督训练 |
| `scripts/train_rl_agent.py` | PPO 训练 |
| `scripts/backtest_rl.py` | 样本外回测 |

## 常见问题

**Q: torch DLL 加载失败？**  
A: 在干净 PowerShell 中单独运行训练脚本；WinUI 与子进程隔离，不影响实机。

**Q: 没有 2016 年数据？**  
A: `prepare_training_data.py` 会使用 CSV 中实际可用区间；建议 MT5 尽量导出更长历史。

**Q: RL 未达标？**  
A: 调整 `config_training.yaml` 中 `env` 止损/止盈、`rl.ent_coef`，或增加 `total_timesteps`。

## 回滚

训练不影响 v1.0.22 基线；实机可随时关闭 RL 智能体开关，或执行 `scripts/restore-baseline.ps1`。
