# V16 重训方案与架构审计摘要

> **目标**：GPU 训练机拉取此仓库后，按本方案重训全栈模型，确保验收通过后再部署。
> **开发机角色**：代码审计、方案制定、部署链路保障。本机不跑训练。

---

## 一、为什么必须重训

### 1.1 当前模型质量（均不合格）

| 模型 | 关键指标 | 当前值 | 门槛 | 状态 |
|------|----------|--------|------|------|
| **Horizon V16** | macro F1 | 0.483 | > 0.50 | ❌ |
| **Horizon V16** | OOS win_rate | 54.3% | > 60% | ❌ |
| **KN2 V16** | long precision | 0.0 | > 0.30 | ❌ |
| **KN2 V16** | short precision | 0.0 | > 0.30 | ❌ |
| **KN2 V16** | long recall | 0.0 | > 0.30 | ❌ |
| **KN2 V16** | pred hold 率 | 99.98% | 正常应 < 90% | ❌ |
| **RL** | eval win_rate | 54.3%（fallback OOS） | > 60% | ❌ |

**KN2 已完全失效**：预测 99.98% 的 K 线为 "hold"，长/短精确率归零，等于没有过滤层。

### 1.2 不重训的后果

- 部署链路中 **entry_quality 公式依赖 KN2 因子**（`kn2_factor`），KN2 失效 → 因子永远 1.0
- **location_score 只反映 M5 滚动区间位置**，不反映多周期结构支撑/阻力
- **entry_quality 中位置权重仅 45%**，方向好但位置差也能过门禁
- 实盘表现为：入场位置差 → 被动止损出局 → 错过正向利润

---

## 二、训练目标

### 2.1 核心验收门槛（`config/v16_acceptance.json`）

```
min_macro_f1: 0.50          # Horizon 全三分类 F1
min_win_rate: 0.60           # OOS 回测胜率
min_oos_win_rate: 0.60       # 2025 年 OOS 胜率
min_train_macro_f1: 0.50     # 训练集 F1（防过拟合）
min_test_macro_f1: 0.50      # 测试集 F1
max_train_test_f1_gap: 0.10  # 训练/测试 F1 差距 ≤ 0.10
min_class_precision: 0.80    # long/short 精确率
min_class_recall: 0.80       # long/short 召回率
min_lockbox_march_win_rate: 0.60
min_rl_eval_win_rate: 0.60
require_no_data_leak: true
require_no_future_function: true
require_temporal_val_split: true
```

### 2.2 理想目标（超过门槛有安全边际）

| 模型 | 目标 | 备注 |
|------|------|------|
| Horizon | macro F1 ≥ 0.52 | 超过 0.50 门槛留余地 |
| Horizon | OOS win_rate ≥ 62% | 超过 60% 门槛 |
| KN2 | long/short precision ≥ 0.40 | 不再归零，能区分 trade/hold |
| KN2 | long/short recall ≥ 0.35 | 能捕捉一部分 trade 信号 |
| KN2 | pred trade 率 5%-20% | 不要 99.98% hold |
| RL | eval win_rate ≥ 62% | 独立评估，不 fallback OOS |

---

## 三、训练顺序

### Step 1: 重训 Horizon V16

```powershell
# 在 GPU 训练机上
py -3 scripts/train_horizon_v16.py
```

**可能需要的调参方向（如果第一次跑不达标）**：
- 增大 `hidden_dim`：96 → 128 或 160
- 调整 `num_res_blocks`：2 → 3
- 增加训练 epoch 或降低学习率
- 检查类别权重（flat 类通常占比过高，导致 trade 类学不好）
- calibration 时调大 `flat_scale`（当前 1.15 → 1.3），防止过度预测 flat

### Step 2: 重训 KN2 V16（最关键）

```powershell
py -3 scripts/train_kn2_v16.py
```

**KN2 当前失败根因分析**：
- location 标签天然稀疏（只在真正有利位置标 trade）
- 类别极度不均衡：hold 占 90%+，long/short 各 < 5%
- 模型收敛到"永远预测 hold"的局部最优

**必须的调参方向**：
- **class_weight**：对 long/short 类加权（建议 hold:long:short = 1:5:5 或更高）
- 或使用 **focal loss**（`gamma=2.0`，`alpha` 对 trade 类倾斜）
- 降低 `LocationLabelConfig.min_rr`：1.2 → 1.0，放开更多候选
- 增加 `hidden_dim`：256 → 384
- 如果仍然失败，考虑两阶段训练：先训 hold vs trade 二分类，再训 long vs short

### Step 3: 验收

```powershell
# Horizon 验收（必须全部 PASS）
py -3 scripts/accept_horizon_v16.py

# KN2 验收
py -3 scripts/accept_kn2_v16.py

# 全量门禁（Horizon + KN2 + RL 都过）
py -3 scripts/pre_deploy_v16_gate.py --require-kn2-live
```

验收通过后，`models/horizon_v16.meta.json` 和 `models/kn2_trader_v16.meta.json` 中 `passed` 会设为 `true`。

### Step 4: RL（可选，Horizon/KN2 通过后）

RL 训练依赖 KN1（Horizon）输出作为状态编码器。Horizon/KN2 重训通过后再跑 RL 可提升执行质量。

---

## 四、当前架构状态（审计结论）

### 4.1 信号链

```
StructureAnalyzer → Horizon(方向) → KN2(位置过滤) → ExecutionComposer(入场计划)
→ RL(仓位) → C# PositionManager(执行)
```

### 4.2 已知缺陷（模型重训后仍需修复）

| # | 缺陷 | 严重度 | 修复时机 |
|---|------|--------|----------|
| 1 | `location_score` 只用 M5 滚动区间 → **已升级为 v2 多维评分** | ✅ 已修 | 2026-06-19 |
| 2 | `entry_quality` 公式位置权重仅 45% → **已提升至 70%（可配置）** | ✅ 已修 | 2026-06-19 |
| 3 | `structure_entry_target` pull 系数过小（0.15 ATR）→ **已增大至 0.35 ATR** | ✅ 已修 | 2026-06-19 |
| 4 | KN2 模型训练失败（precision 0%） | 严重 | **现在重训** |
| 5 | Horizon F1 < 0.50，胜率 < 60% | 严重 | **现在重训** |
| 6 | C# 层无独立结构位置控制 | 中 | 新模型部署时 |
| 7 | `structure_location_gate` 仅对 immediate 模式生效 | 中 | 新模型部署时 |
| 8 | RL `entry_quality_bonus` 可忽略 → **已提升至 0.15** | ✅ 已修 | 2026-06-19 |
| 9 | 入场门禁阈值过松 → **immediate 0.72→0.78, limit 0.38→0.45** | ✅ 已修 | 2026-06-19 |

### 4.3 新增验收审计

- **`scripts/audit_entry_location_quality.py`**：验证 location_score_v2 多维评分、pull 系数、门禁阈值、config 契约
- **`scripts/audit_execution_parity_contract.py`**：新增 v16_strict_3 契约检查项
- **`config/v16_acceptance.json`**：升级至 `v16_strict_3`，新增 `entry_location` 分布审计标准
- **`config/config_agent.json`**：execution_composer 新增 `entry_quality_position_weight: 0.70`，阈值收紧

### 4.4 已修复（不需要 GPU 机关注）

- P0：deploy 脚本不再精简覆写 config → 改用 `Merge-V16AgentConfig.ps1`
- P0：`AgentConfigSync` C# 同步键已含 `execution_composer`、`trading_env`、`kn2`
- P2：`audit_training_no_leak.py` min_win_rate 检查已修正

---

## 五、禁止事项

- **禁止放宽验收门槛**：不要改 `v16_acceptance.json` 中的阈值来"通过"验收
- **禁止用旧模型**：`models/horizon_v16.meta.json` 中 `passed=true` 是旧的宽松标准，新模型验收后会覆盖
- **禁止跳过 KN2**：KN2 是位置过滤核心层，不能因为训练困难就跳过
- **禁止在开发机训练**：本机无 GPU，所有训练在 GPU 机器上跑
- **训练数据不能有泄露**：验证集严格用 2025 年数据，训练集用 ≤ 2024-12-31

---

## 六、GPU 训练机操作流程

```powershell
# 1. 同步代码
git pull origin main
git lfs pull

# 2. 安装依赖（如需要）
.\scripts\retrain_v16_no_leak.ps1 -InstallDeps

# 3. 执行全栈重训
.\scripts\retrain_v16_no_leak.ps1

# 4. 验收
py -3 scripts/accept_horizon_v16.py
py -3 scripts/accept_kn2_v16.py

# 5. 如果全部 PASS，标记部署
py -3 scripts/pre_deploy_v16_gate.py --require-kn2-live

# 6. 推送新模型权重
git add models/ data/training/reports/
git commit -m "V16 full retrain: Horizon + KN2 pass acceptance"
git push
```

---

*最后更新：2026-06-19 — 架构审计 + P0 修复完成后*
