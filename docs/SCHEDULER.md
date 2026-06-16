# 烛龙自动调度系统

基于 XAUUSD v12（样本外胜率 55.7%）与 USOIL v1（64.7%）验收结果，在 WinUI / Python 实机路径中集成 **SchedulerCore**：动态权重、市场状态切换、回撤与连亏保护，全程无需人工切换策略。

## 架构

```
M5 (XAUUSD + USOIL)
        │
        ▼
SchedulerEngine
  ├── AIModelStrategy (v12 + oil_v1)
  ├── SchedulerStateMachine (TREND / RANGE / VOLATILE / MODEL_DEGRADED)
  ├── WeightAllocator (基础权重 × 滑动胜率 × 置信度)
  ├── SchedulerRiskManager (总回撤 / 日损 / 连亏 R)
  └── 其他策略 (trend_system / spread_hedge / grid_system)
        │
        ▼
draw_signal (strategy=scheduler_ai, meta.weights, market_state)
```

## 配置文件

| 文件 | 说明 |
|------|------|
| `config/config_scheduler.json` | 调度主配置（权重/状态机/风控/品种） |
| `config/config_multi_strategy.json` | WinUI 默认路径；`scheduler.enabled=true` 时自动加载调度器 |
| `data/scheduler_state.json` | 运行时持久化（滑动胜率、PnL R） |

### 关键参数

**weight_allocator**

- `base_weights`：XAUUSD 0.4 / USOIL 0.6（可改）
- `winrate_window`：滑动窗口笔数（默认 50）
- `target_winrate`：XAUUSD 0.55 / USOIL 0.65

**state_machine**

- `model_degradation_winrate`：近期胜率低于 0.45 → `MODEL_DEGRADED` → 回退 `trend_system`
- `atr_ratio_threshold`：1.5 → `VOLATILE` → `spread_hedge`
- `adx_threshold`：25 → `TREND` → `ai_model`（双模型加权）

**risk_manager**

- `max_total_drawdown_r`：0.3R
- `max_daily_loss_r`：0.15R
- `max_consecutive_losses`：5

## 启动方式

### WinUI 实机（推荐）

v1.0.20+ 已走 `multi_strategy_tick`；当 `config_multi_strategy.json` 中 `scheduler.enabled=true` 时 **自动使用 SchedulerEngine**，无需另启脚本。

### Python 独立服务

```powershell
cd D:\trae_projects\ZhuLong_3

# 完整调度（MT5 + 管道）
py -3 scripts/realtime_signal.py --scheduler --once

# 或
py -3 scripts/run_scheduler_sim.py --once

# 无 MT5 干跑（验证权重/投票）
py -3 scripts/run_scheduler_sim.py --dry
```

## 信号字段

绘图 JSON 额外字段：

- `strategy`: `scheduler_ai`（或回退策略 id）
- `market_state`: `TREND` / `RANGE` / `VOLATILE` / `MODEL_DEGRADED`
- `meta.weights`: 归一化后的品种权重
- `meta.risk_weight`: 该信号建议风险占比

## 测试

```powershell
py -3 -m pytest tests/test_scheduler.py -v
```

覆盖：权重计算、状态切换、风控阈值、SchedulerCore 投票合并。

## 调参建议

1. **USOIL 表现更好**：提高 `base_weights.USOIL`（默认已 0.6）
2. **回撤敏感**：降低 `max_daily_loss_r` 至 0.10
3. **模型退化过于频繁**：降低 `degradation_window` 或提高 `model_degradation_winrate` 阈值
4. **关闭调度回退旧多策略**：`config_multi_strategy.json` → `"scheduler": { "enabled": false }`

## 与验收指标关系

| 品种 | 验收胜率 | 配置目标胜率 |
|------|----------|--------------|
| XAUUSD v12 | 55.7% | 0.55 |
| USOIL v1 | 64.7% | 0.65 |

WeightAllocator 用目标胜率作归一化基准；实机平仓后应调用 `SchedulerEngine.record_closed_trade(symbol, pnl_r)` 更新滑动窗口（后续 C# 平仓钩子可接入）。
