# 烛龙两跃迁版本 · 100% 工程 + 实机双闭合

> **目标**：用 **Leap 1（V0.9）** + **Leap 2（V1.0）** 完成全部工程与模拟盘实机验收。

---

## 跃迁总览

| 跃迁 | 版本 | 主题 | 闭合类型 |
|------|------|------|----------|
| **Leap 1** | **V0.9** | 工程全链路闭环 | 代码 / 构建 / 单元+集成测试 / 管道 |
| **Leap 2** | **V1.0** | 实机双闭合 | MT5 模拟盘 + 安装包交付 |

**双闭合定义**

1. **工程闭合**：编译 → 测试 → 配置 → 管道 → 特征 → 推理 → 信号 → DB → GUI 全模块连通  
2. **实机闭合**：MT5 挂指标 → ZhuLong.exe 运行 → 信号出图 → 手动下单 → Comment 匹配 → 托管改 SL/TP

---

## Leap 1 · V0.9 工程闭合 ✅

### 交付清单

- [x] WinUI 3 + Core + Python.NET 架构
- [x] 特征 22→30 维 `FeaturePad`（见 `docs/FEATURE_ALIGNMENT.md`）
- [x] 宏观三源：Finnhub/FMP/CSV 日历 + FRED/sentiment JSON + `MacroFeatureBuilder` 8 维
- [x] 会员门禁（与 V14 互通 `OfflineLicenseCodec` + `membership_state.json`）
- [x] 设置页：授权码 / API Key（`UserSecretsStore`，不入 config.json）
- [x] `PositionManagerService`（G5/G8，`PendingSignalStore`）
- [x] `Mt5ApiWrapper` 改单/平仓（Python `mt5_ops.py`）
- [x] 演示模型四件套 `models/XAUUSD/`、`USOIL/`（`scripts/create_demo_models.py`）
- [x] 集成测试 L1-4（信号→SQLite）、L1-5（管道 JSON）
- [x] 一键验收 `scripts/verify_leap1.ps1`
- [x] 管道 smoke 脚本 `scripts/smoke_pipe.py`（GUI 实机补充）
- [x] PyQt5 遗留归档至 `legacy/pyqt5/`

### 验收命令

```powershell
cd d:\trae_projects\ZhuLong_3
.\scripts\verify_leap1.ps1

# 可选实机 smoke（需 GUI 已「开始运行」）：
python scripts/smoke_pipe.py
```

### 门禁

| # | 检查项 | 标准 | 状态 |
|---|--------|------|------|
| L1-1 | 编译 | 0 error | ✅ 自动化 |
| L1-2 | 测试 | 全部 PASS（含集成） | ✅ 自动化 |
| L1-3 | 特征维 | C# 22→30 + macro 8 | ✅ 单元测试 + 文档 |
| L1-4 | 信号链 | Mock 推理 → DB 有记录 | ✅ `SignalPipelineIntegrationTests` |
| L1-5 | 管道 | JSON bar 收发 | ✅ `PipeServerIntegrationTests` + smoke 脚本 |

---

## Leap 2 · V1.0 实机双闭合

### 交付清单

- [x] `scripts/build_release.ps1`（publish + 资源复制）
- [x] 实机验收清单 `docs/ACCEPTANCE.md`
- [x] Inno Setup 脚本 `installer/build_installer.iss`
- [ ] `python_runtime/` 嵌入（`scripts/setup_python_runtime.ps1`）
- [ ] Inno Setup 编译 `ZhuLong_Setup.exe`（需本机 `iscc`）
- [ ] 模拟盘 ≥3 交易日实机签字

### 验收命令

```powershell
.\scripts\setup_python_runtime.ps1   # 首次发布前
.\scripts\build_release.ps1
iscc installer\build_installer.iss     # 需 Inno Setup

# MT5: 编译指标 → M1 挂载 XAUUSD + USOIL
# 运行 publish\win-x64\ZhuLong.exe → 连接 → 开始
# 按 docs/ACCEPTANCE.md 逐项勾选
```

### 门禁

| # | 检查项 | 标准 |
|---|--------|------|
| L2-1 | 安装包 | 无 Python 环境 VM 可启动 GUI |
| L2-2 | 管道 | 指标连接后日志有 M1 bar |
| L2-3 | 信号 | 图表箭头 + WinUI 列表一致 |
| L2-4 | 匹配 | Comment 下单后持仓被托管 |
| L2-5 | 改单 | 移动止损日志有改单事件 |
| L2-6 | 版本 | EXE / 指标 / config 参数一致 |

---

*2026-06-05 — Leap 1 工程闭合 100%（自动化门禁全绿）；Leap 2 待 python_runtime + 实机签字*
