# 烛龙进化路线：统计模型 → 因果推理交易智能体

## 阶段总览

| Phase | 名称 | 状态 | 模块 |
|-------|------|------|------|
| 0 | 统计预测器 (XGB v12/v1) | ✅ 已验收 | `zhulong/inference/v12`, `oil_v1` |
| 1 | 结构感知 | ✅ 已落地 | `zhulong/agent/structure_analyzer.py` |
| 2 | 知识内化 | 🔄 训练中 | `zhulong/agent/knowledge_net.py` |
| 3 | 行为优化 (PPO) | 🔄 训练中 | `zhulong/agent/rl_agent.py`, `trading_env.py` |
| 4 | 因果推理 | ✅ 已落地 | `causal_inference.py`, `fit_causal_coef.py` |
| 5 | 反事实奖励 | ✅ 已落地 | `trading_env.py` + `CounterfactualPredictor` |
| 6 | 在线元学习 | ✅ 已落地 | `meta_learner.py`, `weekly_finetune.py` |

## Phase 4：因果推理

### 配置

- 因果图：`config/causal_graph.yaml`
- 系数文件：`models/causal_coef.pkl`（运行 `scripts/fit_causal_coef.py` 生成）
- 智能体配置：`config/config_agent.json` → `causal` 段

### 拟合系数

```powershell
cd ZhuLong_3
py -3 scripts/fit_causal_coef.py --symbol ALL
```

### 实盘融合

`TradingAgent.on_bar` 在 KnowledgeNet 输出后调用 `fuse_knowledge_with_causal`（默认 70% 知识网络 + 30% 因果）。

### 验收

- 宏观事件日方向准确率 ≥ 65%（`scripts/ab_test_rl_causal.py` + 事件日切片）
- 2025 样本外胜率提升 ≥ 3pp（训练完成后对比基线）

## Phase 5：反事实奖励

`trading_env.py` 在 `counterfactual.enabled=true` 时：

```
causal_reward = actual_pnl_r - luck_pnl_r
```

`luck_pnl_r` 由 `CounterfactualPredictor` 根据持仓期间外生冲击估计。

重新训练 RL（启用因果奖励）：

```powershell
py -3 scripts/train_rl_agent.py --symbol XAUUSD
```

A/B 对比：

```powershell
py -3 scripts/ab_test_rl_causal.py --symbol XAUUSD --year 2025
```

## Phase 6：在线元学习

### 模块

- `zhulong/agent/meta_learner.py` — 轨迹缓存 + 偏置微调
- `zhulong/agent/adaptation_trigger.py` — 近 20 笔胜率 < 45% 触发
- `zhulong/agent/agent_scheduler.py` — 实盘调度集成
- `scripts/weekly_finetune.py` — 每周 PPO 继续训练（学习率 ×0.1）

### 配置 (`config_agent.json`)

```json
"meta_learning": {
  "enabled": true,
  "meta_learning_rate": 0.0001,
  "meta_batch_size": 10,
  "update_interval_steps": 50
},
"adaptation_trigger": {
  "window": 20,
  "threshold": 0.45
}
```

### 周度微调

```powershell
py -3 scripts/weekly_finetune.py --symbol XAUUSD --timesteps 5000
```

快照保存在 `models/snapshots/{symbol}/`，可回滚。

## 推荐执行顺序（模型训练完成后）

1. 验证 KnowledgeNet / RL 基线指标达标
2. `fit_causal_coef.py` → 检查 `models/causal_coef.pkl`
3. 启用 `config_agent.json` 中 `causal` + `counterfactual`
4. 重新训练 RL（因果奖励）并与基线 A/B
5. 模拟盘开启元学习，收集 1–2 周基线
6. 胜率下滑时自动触发 `AgentScheduler` / 手动 `weekly_finetune.py`

## 回滚

| 层级 | 操作 |
|------|------|
| 因果融合 | `config_agent.json` → `causal.enabled: false` |
| 反事实奖励 | `counterfactual.enabled: false` 后重训 RL |
| 元学习 | `meta_learning.enabled: false` |
| 全量 | `scripts/restore-baseline.ps1` 或 `models/snapshots/` 恢复 |

## 测试

```powershell
py -3 -m pytest tests/test_causal_inference.py tests/test_meta_learner.py tests/test_trading_env_causal.py tests/test_trading_agent.py -q
```
