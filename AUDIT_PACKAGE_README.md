# 烛龙 ZhuLong III — 全量架构审计包

> 用途：供 Claude / 外部审计方做**架构与设计缺陷**全量分析  
> 版本：V16 实盘路径 · 导出日期 2026-06-19  
> 仓库：`ZhuLong3_DevPackage_v2`

---

## 1. 审计目标

请从以下维度审查本系统，找出**架构层、设计层、集成层**的缺陷（不仅限于 bug）：

| 维度 | 关注点 |
|------|--------|
| **双栈一致性** | Python Agent 路径 vs C# WinUI 运行时 vs MT5 MQ5 是否语义一致 |
| **虚拟 vs 实盘** | C# `PositionManagerService` 虚拟持仓 vs MT5 真实 ticket 的匹配与漂移 |
| **信号→执行→平仓** | 全链路是否有断点、旁路、重复逻辑、写了未接入 |
| **配置生效** | `config_agent.json` / AppData 镜像 / C# 合并逻辑是否真正生效 |
| **模型栈** | Horizon → KN2 → TraderMind/ExecutionComposer → Cognition → RL 的依赖与降级 |
| **训练-推理对齐** | 特征维度、标签定义、acceptance gate 与线上路径是否一致 |
| **遗留/并行路径** | 多策略 Scheduler、Python `position_manager.py`、旧 V14/V15 代码是否造成混淆 |
| **可运维性** | 部署、热更新、回滚、日志可观测性 |
| **安全与风控** | RiskGuard、membership、异常降级 |

---

## 2. 系统概览

### 2.1 技术栈

| 层级 | 技术 | 路径 |
|------|------|------|
| UI | WinUI 3 + MVVM | `src/ZhuLong.App` |
| 编排/管道/DB | C# .NET 8 | `src/ZhuLong.Core` |
| Python 嵌入 | Python.NET | `src/ZhuLong.App/Services/PythonInferenceService.cs` |
| Agent 推理 | PyTorch / SB3 / ONNX | `zhulong/agent/` |
| Python 入口 | CLI 桥 | `ZhuLong.PythonEngine/` |
| MT5 | 命名管道 + MQ5 | `mql5/` |
| 配置 | JSON + AppData 镜像 | `config/` + `%APPDATA%/ZhuLong/` |

### 2.2 V16 主链路（实盘）

```text
MT5 M1/M5 数据
    → PipeServer / FeatureCacheService (C#)
    → ZhuLongRuntimeService.RunTradingAgentSignalTickAsync
    → PythonInferenceService.AgentTickAsync
        → StructureService → HorizonPredictor → KN2
        → TraderMind / ExecutionComposer (最佳入场位置)
        → Cognition → RL → draw_signal
    → SignalGeneratorService / TryEmitSignalAsync
    → PositionManagerService (C# 虚拟成交/持仓)
    → FastTrailingStopAsync / ApplyAgentM5PositionManagement
    → CloseVirtualAsync → NotifyAgentClosedTradeAsync
    → MT5 modify_sl_tp（按 signal_id 匹配 comment，非 OrderSend 开仓）
```

**关键设计**：C# 侧以**虚拟持仓**管理 Agent 逻辑；MT5 侧通过 comment 中的 `signal_id` 关联真实 ticket，主要做 SL/TP 修改。

### 2.3 关键入口文件（优先阅读）

| 文件 | 作用 |
|------|------|
| `src/ZhuLong.App/Services/ZhuLongRuntimeService.cs` | 运行时编排、M5 tick、Agent 路径 |
| `src/ZhuLong.App/Services/PositionManagerService.cs` | 虚拟持仓、移动止损、平仓 |
| `src/ZhuLong.App/Services/PythonInferenceService.cs` | Python.NET 桥接 |
| `zhulong/agent/trading_agent.py` | Agent tick 主流程 |
| `zhulong/agent/execution_composer.py` | 入场位置评分 V2、immediate/limit 门控 |
| `zhulong/agent/horizon_predictor.py` | Horizon 宏观预测 |
| `zhulong/agent/knowledge_net_kn2.py` | KN2 情景网络 |
| `zhulong/agent/cognition.py` | 认知层决策 |
| `zhulong/agent/rl_agent.py` | PPO 动作 |
| `config/config_agent.json` | Agent 全量配置 |
| `config/v16_acceptance.json` | V16 验收门槛 `v16_strict_3` |

### 2.4 架构决策 ADR

完整 G1–G13 见 `docs/DECISIONS.md`，实现计划见 `docs/IMPLEMENTATION_PLAN.md`，WinUI 架构见 `docs/WINUI_ARCHITECTURE.md`。

---

## 3. 包内审计脚本

本包已含静态/半静态审计脚本；**实机审计**需在 WinUI 运行状态下于 `%APPDATA%/ZhuLong` 有日志和 DB 时执行。

| 脚本 | 用途 |
|------|------|
| `scripts/audit_production_live.py` | **68 项 14 类**实机全链路审计（核心） |
| `scripts/audit_architecture.py` | KN2/因果图/模块导入 |
| `scripts/audit_integration_wiring.py` | 集成接线 |
| `scripts/audit_execution_parity_contract.py` | 执行层契约对齐 |
| `scripts/audit_entry_location_quality.py` | 最佳入场位置 |
| `scripts/audit_position_management_path.py` | 持仓管理路径 |
| `scripts/audit_training_no_leak.py` | 训练泄漏（`--pre` / `--post`） |
| `scripts/verify_v16_full_stack.py` | V16 全栈验证 |
| `scripts/audit_runtime_db.py` | SQLite 运行时 DB |
| `scripts/audit_v16_live_log.py` | 实盘日志 |

`audit_outputs/` 目录含导出时运行的静态审计摘要（若有）。

---

## 4. 已知问题（开发侧已记录，请验证并扩展）

以下在最新实机审计中**降级为 P2**（不阻塞部署，但属架构/design debt）：

1. **Python `position_manager.py` vs C# `PositionManagerService`**  
   两套持仓管理；Python 版主要在 `app.py` 独立模式，与 WinUI 主进程不同进程，无真实竞态，但增加认知负担与维护成本。

2. **`decision_bar_unix` 与 M5 index 不对齐**  
   低频率；微秒/时区边界问题，可能影响入场位置回溯精度。

3. **双系统架构**  
   旧多策略 Scheduler 路径与 V16 Agent 路径并存；部署新模型时应清理 Python 侧冗余路径。

4. **模型质量未达 `v16_strict_3` 真实门槛**  
   Horizon macro F1≈0.48、KN2 近 100% hold、RL 低于目标；当前 gate 曾通过 meta 中 `passed=true` + `temporal_val=true` 人工修复。请审计**验收体系是否可被绕过**。

5. **文档 vs 实现差距**  
   `docs/WINUI_ARCHITECTURE.md` Phase 2 待办（特征维度 22 vs 30、完整持仓规则等）可能与现状不一致，需对照代码。

---

## 5. 建议审计方法

### 5.1 静态（本包即可）

1. 读 `docs/DECISIONS.md` + 上表关键入口  
2. 追踪 `RunTradingAgentSignalTickAsync` → `AgentTick` → `TryEmitSignalAsync` → `PositionManagerService` 调用链  
3. 对照 `config/config_agent.json` 每个关键字段是否有读取方  
4. 搜索 `OrderSend`、旁路 `draw_signal`、未使用的 `enabled` 开关  
5. 比对 Python/C# 同名概念（virtual fill、signal_id、SL/TP 规则）

### 5.2 动态（需实机，可选）

```powershell
cd <解压路径>
py -3 scripts/audit_production_live.py
py -3 scripts/verify_v16_full_stack.py
py -3 scripts/audit_integration_wiring.py
```

### 5.3 输出格式建议

请按 severity 输出：

- **P0** — 可导致错误交易/资金风险/静默失效  
- **P1** — 架构缺陷、维护不可持续、验收可绕过  
- **P2** — 技术债、文档不一致、低频边界  
- **建议** — 重构方向，非必须立即修复

每项请给出：**现象 → 根因 → 影响 → 建议修复 → 涉及文件**

---

## 6. 包内容说明

| 包含 | 排除 |
|------|------|
| `src/` 源码（无 bin/obj/out） | 编译产物、publish、exe/dll |
| `zhulong/` 全量 Python | `.pt` `.onnx` `.pkl` 权重 |
| `config/` `docs/` `scripts/` `tests/` 源码 | `data/` 训练集、`logs/` |
| `models/*.meta.json` 元数据 | 模型二进制 |
| `mql5/` `native/` `ZhuLong.PythonEngine/` | `.git/` `build_cache/` |

---

## 7. 相关文档索引

- `README.md` — 项目入口  
- `TRAINING_PLAN_V16.md` — V16 重训方案  
- `docs/AGENT.md` — RL Agent 架构  
- `docs/DEPLOYMENT_DUAL_LIVE.md` — 双实盘部署  
- `docs/FEATURE_ALIGNMENT.md` — 特征对齐  
- `docs/MODEL_TRAINING_ACCEPTANCE.md` — 训练验收  
- `MIGRATION_README.md` — 迁移说明  

---

*本文件由烛龙开发包导出流程自动生成，供外部全量架构审计使用。*
