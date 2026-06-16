# 烛龙三代（ZhuLong III）落地方案

> **来源**：基于 DeepSeek 开发方案（`烛龙三代.md`，Stephen.Pan）整理  
> **定位**：全新独立工程，**不对齐** Zhulong_14 / 数字交易员 / KO 真脑体系  
> **产品形态**：黄金/原油 **半自动日内波段** — 模型出信号，用户手动下单，系统托管持仓  
> **交付形态**：**WinUI 3 可安装 EXE** + **MT5 MQ5 指标**（见 [`DELIVERY.md`](./DELIVERY.md)）  
> **技术栈**：**WinUI 3**（C# .NET 8）+ **ZhuLong.Engine**（Python 推理子进程）+ MQL5 命名管道

---

## 1. DeepSeek 方案评审摘要

### 1.1 方案优点（可直接落地）

| 维度 | 评价 |
|------|------|
| 架构边界清晰 | MT5 指标负责 K 线推送与图表绘制；Python 负责特征、推理、持仓、GUI — 职责分离合理 |
| 特征设计完整 | 5 分钟序列（~30 维 × 60 根）+ 1 小时背景 + 宏观因子，符合日内波段场景 |
| 标签定义可操作 | 三分类 + `entry_offset` 回归，阈值可调，便于回测调参 |
| 模型路线务实 | Transformer 做序列编码 + XGBoost 做融合决策，比纯端到端更易调试、归因 |
| 信号过滤链完整 | 置信度、预期收益、盈亏比、冷却、波动率、宏观安全阀 — 多层风控 |
| 持仓管理规则明确 | 移动止损、分批止盈、时间止损、利润回撤、模型辅助出场 — 可参数化 |
| 文件清单与配置 | `config.json` 结构清晰，模块划分可直接映射到包结构 |

### 1.2 架构决策（G1–G10 · 已钉死）

全部定稿见 **[`DECISIONS.md`](./DECISIONS.md)**。开发时不得偏离；变更须更新该文档。

| 编号 | 主题 | 定稿摘要 |
|------|------|----------|
| G1 | `expected_return` | 训练=未来30分钟实际收益率；MVP=`confidence×historical_avg_gain(100)`；正式版=`reg_expected_return` |
| G2 | Transformer | 两阶段：15% mask 自监督预训练 → 固定编码器 → XGBoost（clf + reg_entry + reg_expected_return） |
| G3 | Scaler | 单一 `models/scaler.pkl`；推理 reshape 变换；缺失则拒绝启动 |
| G4 | MT5 双通道 | 管道线程收 M1；API 线程 1Hz 管持仓；独立断线重连 |
| G5 | 持仓匹配 | `magic=hash(signal_id)&0xFFFF`；Comment=`ZhuLong_<signal_id>`；GUI 一键复制 |
| G6 | 初始 SL/TP | ATR(14)×mult（1.2/2.0）；信号生成时固化 |
| G7 | 符号映射 | `symbol_mapping`；内部标准符号，MT5 侧自动映射 |
| G8 | 价格/点数 | 内部绝对价格；展示用 `SYMBOL_POINT`；改单传绝对价格 |
| G9 | SQLite | `signals` / `trades` / `position_events` — 见 `zhulong/db/schema.sql` |
| G10 | 线程 | 5 线程：GUI、DataReceiver、SignalScheduler(5min)、PositionManager(1Hz)、MacroCalendar(1d) |
| G11 | **交付形态** | **EXE 安装包** + **MQ5 指标**；见 [`DELIVERY.md`](./DELIVERY.md) |
| G12 | **GUI** | **WinUI 3**（`ZhuLong.exe`）+ **Engine**（`ZhuLong.Engine.exe`） |

---

## 2. 目标目录结构（新工程）

```text
ZhuLong_3/
├── ZhuLong.sln
├── src/
│   ├── ZhuLong.App/                # WinUI 3 主程序 → ZhuLong.exe
│   └── ZhuLong.Core/               # 配置、路径、EngineHost、SQLite 读
├── engine.py                       # 引擎入口（开发）
├── ZhuLong.Engine.spec             # PyInstaller → ZhuLong.Engine.exe
├── zhulong/                        # Python 推理引擎（管道/MT5/模型）
├── train.py
├── mql5/ZhuLongIndicator.mq5
├── config.json
├── installer/build_installer.iss
└── docs/
```

---

## 3. 分阶段落地路线图

### Phase 0 — 工程基建（1–2 天）

**目标**：空仓库可运行、可测试、可配置。

- [ ] 初始化 Git 仓库、`requirements.txt`、`.gitignore`（models/、logs/、*.pkl）
- [x] `config.json` 初版（含 `symbol_mapping`、`signal_geometry`、G1–G10 字段）
- [ ] JSON Schema 校验
- [ ] 统一日志（按日滚动）、异常钩子
- [ ] `pyproject.toml` 或 `pytest` 基架

**验收**：`dotnet build src/ZhuLong.App` 通过；`python engine.py` 可启动引擎（需 models）。

---

### Phase 1 — MT5 双通道通信（3–4 天）

**目标**：M1 K 线稳定进 Python；Python 能在图表画信号。

| 任务 | 说明 |
|------|------|
| MQL5 `ZhuLongIndicator.mq5` | 新 M1 bar → JSON 一行写入 `\\.\pipe\ZhuLong_Data`；读 `ZhuLong_Drawing` 画箭头/SL/TP |
| `data_receiver.py` | 阻塞读管道、JSON 解析、断线重连、bar 去重（symbol+time） |
| `drawing_client.py` | 发送 `draw_signal` / `clear_signal` |
| `mt5_bridge.py` | `initialize/shutdown`、账户信息、符号 `symbol_info` |
| 协议测试 | 模拟器脚本向管道写 bar，断言 Python 收到 |

**验收**：MT5 模拟盘挂指标，Python 日志每根 M1 打印；手动发 draw 指令，图表出现标记。

**风险**：Windows 命名管道 buffer、MQL5 `FileOpen` 权限 — 需 MT5「允许 DLL」+ 以管理员测试一次。

---

### Phase 2 — 特征工程与数据管道（4–5 天）

**目标**：实时与离线共用同一套特征代码。

| 任务 | 说明 |
|------|------|
| M1 → M5/M60 重采样 | 仅使用 **已收盘** bar；处理休市 gap |
| `feature_engine.py` | 实现方案 §4.1–4.2 全部特征；输出 `(60, 30)` 序列 + 1h 向量 |
| `macro_calendar.py` | 加载 CSV；`hours_to_next_event` 等；单元测试用固定时间 |
| `scripts/download_m1.py` | 拉 6 个月+ M1 存 parquet |
| Feature manifest | JSON 列名顺序，训练/推理一致 |

**验收**：对固定 CSV 快照，特征 hash 不变；实时缓冲维持 60 根 M5。

---

### Phase 3 — 标签、训练与模型制品（5–7 天）

**目标**：XAUUSD、USOIL 各一套可加载模型。

| 任务 | 说明 |
|------|------|
| `labeler.py` | 方案 §5 三分类 + entry_offset |
| `dataset.py` | 时间序 80/20 划分；**禁止 shuffle** |
| Transformer 阶段一 | 15% mask 特征重建预训练；保存 `transformer.pt`（见 G2） |
| Transformer 阶段二 | 固定编码器 → 提取 32 维 → 拼接全局特征 |
| XGBoost | `xgb_clf` + `reg_entry_offset` + `reg_expected_return`（正式版）；MVP 可暂用 G1 经验公式 |
| Scaler | `joblib.dump` → `models/scaler.pkl`（见 G3） |
| `export_artifacts.py` | scaler、feature manifest、模型版本 `manifest.json` |
| 基线对比 | 纯 XGBoost（无 Transformer）作为 sanity check |

**验收**：

- 验证集 Macro-F1 > 随机基线；回归 MAE 合理  
- `inference_engine.load(symbol)` 单次推理 < 100ms（CPU）

---

### Phase 4 — 信号生成与过滤（3 天）

**目标**：每 5 分钟产出 0–N 条合格信号。

- [ ] `inference_engine.py`：加载制品 → direction / confidence / entry_offset / expected_return
- [ ] `signal_generator.py`：方案 §7 全部过滤 + 冷却状态机
- [ ] SL/TP 公式写入配置
- [ ] `pending_signals` 队列 + 过期清理（`expiry_minutes`）

**验收**：历史 bar 回放，信号时间戳与 M5 对齐；过滤原因可日志追溯。

---

### Phase 5 — 持仓管理（4–5 天）

**目标**：匹配用户单后自动托管。

| 任务 | 说明 |
|------|------|
| 匹配逻辑 | Comment 优先（`ZhuLong_<signal_id>`）→ 降级价格+时间；magic=`hash(signal_id)&0xFFFF`（G5） |
| 移动止损 / 分批止盈 | 方案 §8.2；价格用 bid/ask 区分多空 |
| 时间止损 / 回撤保护 | 每秒扫描 |
| 模型辅助出场 | Phase 5 末开启；与 Phase 4 推理复用 |
| 下单重试 | 最多 3 次；失败写日志 + GUI 告警 |

**验收**：模拟盘手动跟单后，SL 修改、部分平仓在日志可验证；**勿在生产未测前开实盘**。

---

### Phase 6 — GUI 与归因（4–5 天）

**目标**：可操作、可复盘。

- [ ] 主窗口：信号表（含一键复制 Comment）、持仓表、参数面板、日志
- [ ] 参数热更新 vs 需重启 — 文档标明
- [ ] SQLite：`trades`、`signals`、`position_events`
- [ ] `attribution.py`：近 N 笔胜率、盈亏比、参数分箱（matplotlib 嵌入）

**验收**：完整闭环一笔交易写入 DB；归因面板出图。

---

### Phase 7 — 回测、模拟盘验收与 **EXE 交付打包**（5–7 天）

**目标**：产出 **`ZhuLong_Setup_vX.Y.Z.exe`** + **`ZhuLongIndicator.ex5`**，详见 [`DELIVERY.md`](./DELIVERY.md)。

| 任务 | 说明 |
|------|------|
| `backtest/engine.py` | 历史 M5 逐步推理 + 信号过滤 |
| Walk-forward | 滚动 3 个月训练 / 1 个月测 |
| 模拟盘 2 周 | 方案 §15 |
| `ZhuLong.spec` | PyInstaller → `ZhuLong.exe`（`console=False`， bundled models/config） |
| `installer/ZhuLong_Setup.iss` | Inno Setup → 可安装 EXE |
| `scripts/build_release.ps1` | 一键：pytest → pyinstaller → 编译 mq5 → 打安装包 |
| MQL5 编译 | MetaEditor 产出 `ZhuLongIndicator.ex5`，随安装包放入 `mql5/` |
| 安装说明 | `安装说明.pdf`：EXE 安装 + 指标复制到 MT5 + 首次运行 |
| 干净 VM 验收 | **无 Python 环境** 的 Windows 10/11 安装并运行 |

**上线门禁**：

1. 回测与模拟盘通过（§3 Phase 7 原条目）  
2. **`ZhuLong_Setup_*.exe` 在干净 VM 可安装、可启动 GUI**  
3. EXE + `.ex5` 管道连通 smoke test  
4. 版本号：安装包 / EXE / 指标 / `manifest.json` 一致  
5. 用户数据写入 `%APPDATA%\ZhuLong\`，升级安装不丢 DB  

---

## 4. 需补足的设计（DeepSeek 方案未覆盖）

### 4.1 功能增补（建议 MVP 纳入）

| 优先级 | 模块 | 说明 |
|--------|------|------|
| P0 | `mt5_bridge.py` | 统一 MT5 生命周期；避免 GUI/持仓各 init 一次 |
| P0 | 制品版本 `manifest.json` | model_version、训练区间、特征 hash、git commit |
| P0 | Scaler 持久化 | `models/scaler.pkl`，缺失拒绝启动（G3） |
| P0 | magic + comment 匹配 | G5：`hash(signal_id)&0xFFFF` + `ZhuLong_<signal_id>` |
| P0 | 符号映射 `symbol_mapping` | G7 |
| P1 | `risk_guard.py` | 单日最大亏损 %、最大同时持仓数、单品种冷却 |
| P1 | 健康检查 | 管道心跳、最后 bar 时间、MT5 连接状态 — GUI 顶部灯 |
| P1 | 信号/持仓事件审计 | 全量写 SQLite，便于归因 |
| P1 | `scripts/fetch_macro_calendar.py` | 从 Investing/FMP 等拉日历生成 CSV（需 API key 配置） |
| P2 | Telegram/邮件告警 | 连接断开、日损触发、下单失败 |
| P2 | 多账户配置 | 后期扩展 |

### 4.2 SQLite 表结构

已定稿，见 **`zhulong/db/schema.sql`** 与 **[`DECISIONS.md` G9](./DECISIONS.md#g9--sqlite-表结构tradingdb)**。

### 4.3 配置文件

已定稿初版：**`config/config.json`**。关键字段：

- `signal_geometry.initial_stop_loss_atr_mult` / `initial_take_profit_atr_mult`（G6）
- `symbol_mapping`（G7，默认 `{}`）
- `model.use_xgb_expected_return`：`false`=MVP 经验公式，`true`=正式版回归器（G1）
- `model.historical_avg_gain_window`：100（G1）
- `model.transformer_mask_ratio`：0.15（G2）

### 4.4 测试策略

| 层级 | 内容 |
|------|------|
| 单元 | 特征计算、标签、过滤链、宏观特征 |
| 集成 | 管道 mock、MT5 模拟盘 smoke |
| 回归 | 固定 parquet → 固定信号快照 golden file |
| 人工 | 模拟盘 2 周 + K 线锚点复盘 |

### 4.5 运维与非功能

- **时区**：全链路 MT5 服务器时间；macro CSV 统一转 UTC 或 server time 并文档化  
- **安全**：`.env` 存 API key；不进 Git  
- **免责声明**：GUI 关于页 — 非投资建议  
- **备份**：SQLite 日备；models 目录版本管理  

---

## 5. 关键技术决策（建议默认）

| 决策 | 选择 | 理由 |
|------|------|------|
| **用户交付** | **WinUI 3 EXE + Engine + MQ5** | 见 `DELIVERY.md` |
| GUI 框架 | **WinUI 3**（C#） | 弃用 PyQt5 |
| 推理运行时 | **ZhuLong.Engine.exe**（Python 打包） | WinUI 不嵌入 PyTorch |
| 与 Zhulong_14 关系 | 完全独立仓库 | 用户明确要求 |
| K 线来源 | MQL5 管道推送 M1 | 与方案一致；Python 自合成 M5 |
| 下单方式 | 用户手动；系统只改 SL/TP/平仓 | 半自动定位 |
| 模型服务 | 进程内推理 | 低延迟；无额外服务 |
| GUI 框架 | PyQt5 | **WinUI 3**（C# .NET 8） |
| 主 EXE 打包 | PyInstaller | **dotnet publish** |
| 推理 | 同进程 Python | **ZhuLong.Engine.exe** 子进程 |
| 数据存储 | SQLite | 轻量、归因够用 |
| 首次交付品种 | 先 XAUUSD 跑通，再 USOIL | 降低并行复杂度 |

---

## 6. 里程碑与工期估算

| 里程碑 | 内容 | 工期（人日） |
|--------|------|--------------|
| M0 | Phase 0 基建 | 2 |
| M1 | 管道 + MT5 连通 | 4 |
| M2 | 特征 + 训练流水线 | 10 |
| M3 | 信号 + 持仓闭环 | 8 |
| M4 | GUI + 归因 | 5 |
| M5 | 回测 + 模拟盘 + **EXE 安装包交付** | 7 |
| **最终交付物** | `ZhuLong_Setup_vX.Y.Z.exe` + `ZhuLongIndicator.ex5/.mq5` | — |
| **合计** | MVP 可模拟盘运行 | **~36 人日**（约 7–8 周单人兼职） |

---

## 7. 立即执行的下一步

1. ~~钉死 G1–G10 → `docs/DECISIONS.md`~~ ✅  
2. Phase 0：Git、`requirements.txt`、pytest 基架  
3. Phase 1：`ZhuLongIndicator.mq5` + `DataReceiverThread` 管道连通（G4/G10）  
4. 并行：`download_m1.py` 拉 XAUUSD 历史，验证 `feature_engine`  

---

## 8. 与 DeepSeek 方案差异一览

| 项目 | DeepSeek 方案 | 本落地方案 |
|------|---------------|------------|
| 包结构 | 根目录平铺 py 文件 | `zhulong/` 包 + `train/` + `scripts/` |
| expected_return | 未定义 | G1：MVP 经验公式 → 正式版 `reg_expected_return` |
| Transformer 训练 | 简略 | G2：15% mask 预训练 + 三头 XGBoost |
| Scaler | 未提及 | G3：单一 `models/scaler.pkl` |
| 线程 | 未明确 | G10：5 线程模型 |
| 回测 | 未提及 | Phase 7 独立模块 |
| 风控 | 仅信号过滤 | 增补 `risk_guard` 日损/持仓上限 |
| 测试 | 未提及 | pytest + golden snapshot |
| 部署 | `python ZhuLong.py` | **EXE 安装包** + MQ5 指标（`DELIVERY.md`） |
| 持仓识别 | 价格+时间 | + magic + comment |

---

*文档版本：v1.2 · 2026-06-05 · 交付：EXE 安装包 + MQ5 指标*
