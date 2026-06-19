# 烛龙 V17 架构重训方案

> 完整设计见用户文档 `v17_redesign_20260619`  
> 一键入口：`powershell -ExecutionPolicy Bypass -File scripts/retrain_v17.ps1`

## 核心变化

| 组件 | v16 | v17 |
|------|-----|-----|
| 方向 | Horizon 三分类 ResNet | DirectionScorer LightGBM 回归 |
| 位置 | KN2 GRU 六分类 | LocationGate XGBoost 二分类 |
| 回测 | 已加成本 (P0) | 全链路 + limit/market 滑点区分 |
| RL 验收 | 独立 eval (P0) | 同 v16 P0 修复 |

## 执行顺序

```powershell
cd D:\trae_projects\ZhuLong3_DevPackage_v2

# 阶段 0–3
powershell -ExecutionPolicy Bypass -File scripts/retrain_v17.ps1 -InstallDeps

# 阶段 4：模型通过后 RL
py -3 scripts/train_rl_agent.py --v16 --symbol XAUUSD

# 切换实盘配置（AppData 需 merge）
# 将 config/config_agent_v17.json 合并到 %APPDATA%\ZhuLong\config_agent.json
# architecture.version = "v17"
```

## 验收

- `config/v17_acceptance.json`
- `scripts/accept_direction_scorer.py`
- `scripts/accept_location_gate.py`
- `scripts/backtest_v17_full_chain.py --with-cost`

## 回滚

保持 `architecture.version: "v16"` 即可，v16 路径未删除。
