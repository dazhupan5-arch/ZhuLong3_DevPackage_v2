# 烛龙 RL 交易智能体（v3.0）

## 架构

```
M5 K线 → StructureAnalyzer (30维)
       → KnowledgeNet (概率 + 32维嵌入)
       → StateBuilder (74维状态)
       → PPO / 启发式 → 动作 0-5 → draw_signal (strategy=rl_agent)
```

启用后 **替代** 多策略/SchedulerCore 路径；WinUI 仍通过命名管道下发图表信号，不直接 OrderSend。

## 配置

`config/config_agent.json` — 结构分析、知识网络、RL 模型路径、交易成本、TraderMemory。

用户 `config.json`：

```json
"trading_agent": {
  "enabled": true,
  "config_path": "config/config_agent.json"
}
```

设置页勾选 **RL 交易智能体** 等效于 `enabled: true`。

## 训练

完整训练流程见 **[docs/TRAINING.md](TRAINING.md)**（`prepare_training_data` → `train_knowledge_net` → `train_rl_agent` → `backtest_rl`）。

一键：

```powershell
.\scripts\run_agent_training.ps1 -Symbol XAUUSD
```

未训练时自动 **启发式回退**（结构特征 + 趋势），不会崩溃。

## WinUI 集成

- Python CLI：`agent_tick`
- C#：`PythonInferenceService.AgentTickAsync`
- 运行时：`RunTradingAgentSignalTickAsync`（优先于多策略）

## 动作空间

| 动作 | 含义 |
|------|------|
| 0 | 观望 |
| 1-2 | 开多 50% / 100% |
| 3-4 | 开空 50% / 100% |
| 5 | 平仓 |

信号 comment 元数据含 `RL_Agent`。

## 命令行（开发）

```powershell
py -3 scripts/realtime_signal.py --agent --config config/config_agent.json
```

## 测试

```powershell
cd ZhuLong_3
py -3 -m pytest tests/test_structure_analyzer.py tests/test_trading_agent.py tests/test_trading_env.py -q
```

## 进化（Phase 4–6）

因果推理、反事实奖励、在线元学习见 **[docs/EVOLUTION.md](EVOLUTION.md)**。

## 回滚

仍可用 `scripts/restore-baseline.ps1` 或 git tag `baseline/v1.0.22-20260609` 回到 v1.0.22。
