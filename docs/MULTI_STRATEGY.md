# 烛龙多策略引擎

XGBoost AI 与三条规则策略（趋势 / 金油对冲 / ATR 网格）通过状态机自动切换，覆盖趋势、高波动、震荡三类行情。

## 架构

```
MT5 M5 ──► MultiStrategyEngine ──► StrategyStateMachine
                    │                      │
                    │          TREND  → ai_model | trend_system
                    │          VOLATILE → spread_hedge
                    │          RANGE  → grid_system
                    ▼
            ZhuLong_Drawing (draw_signal + strategy 字段)
```

## 目录

| 路径 | 说明 |
|------|------|
| `zhulong/strategies/` | 策略基类与四个策略实现 |
| `zhulong/engine/multi_strategy_engine.py` | 总控与 MT5 Runner |
| `config/config_multi_strategy.json` | 统一配置 |
| `scripts/run_multi_strategy.py` | 独立启动 |
| `scripts/realtime_signal.py --multi-strategy` | 兼容原双品种服务 |
| `scripts/backtest_model.py` | v12 离线回测 |
| `scripts/stress_test.py` | 极端窗口压力测试 |
| `tests/test_model_performance.py` | 单元测试 + 验收门槛 |

## 快速启动

```powershell
cd D:\trae_projects\ZhuLong_3

# 多策略引擎（推荐）
py -3 scripts/run_multi_strategy.py --once

# 或接入原 realtime 服务
py -3 scripts/realtime_signal.py --multi-strategy --once
```

## 配置说明

`config/config_multi_strategy.json`：

- **state_machine.primary_symbol**：主调度品种（默认 XAUUSD）
- **state_machine.trend_strategy**：`ai_model` 或 `trend_system`
- **spread_hedge.entry_zscore**：金油比入场 Z 阈值
- **grid_system.low_volatility_atr_percentile**：网格激活的 ATR 分位

## 绘图 JSON

```json
{
  "action": "draw_signal",
  "signal_id": "multi_20260608_1200_ai_model_XAUUSD_buy",
  "symbol": "XAUUSD",
  "direction": "buy",
  "entry": 4450.0,
  "sl": 4445.0,
  "tp": 4460.0,
  "confidence": 0.85,
  "strategy": "ai_model",
  "expiry_minutes": 240
}
```

MT5 指标无需改协议；`strategy` 为新增字段，旧版忽略即可。

## 模型能力测试

```powershell
# 样本外回测
py -3 scripts/backtest_model.py --symbol XAUUSD --start 2024-01-01

# 压力窗口
py -3 scripts/stress_test.py --symbol XAUUSD

# 单元测试
py -3 -m pytest tests/test_model_performance.py -v
```

### 通过标准（自检清单）

| 测试项 | 门槛 |
|--------|------|
| 样本外胜率 | ≥ 52% |
| 盈亏比 (profit factor) | ≥ 1.3 |
| 压力测试 max_dd_r | ≤ 20 |
| 日信号次数 | ≤ 8（由 AI 冷却控制） |

## 与 WinUI 实机的关系

- **WinUI v1.0.21+**：`multi_strategy.enabled` + `scheduler.enabled=true` 时走 **SchedulerEngine**（动态权重 + 回撤保护 + 状态机）。
- 详见 [SCHEDULER.md](SCHEDULER.md)。
- **设置页**：可关闭多策略回退单模型 V12；「模型回测 / 验收摘要」读取 `models/*/manifest.json`。
- **独立 Python 服务**：`run_multi_strategy.py` / `realtime_signal.py --scheduler`。

## 模拟盘测试建议

1. MT5 模拟账户 + `ZhuLongIndicator` 已连管道  
2. `run_multi_strategy.py` 运行 2–4 周  
3. 对比 `logs/multi_strategy.log` 与 `signals` 表  
4. 按 `strategy` 字段统计各策略胜率与相关性  
