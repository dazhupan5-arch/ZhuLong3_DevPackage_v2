# 烛龙三代 · 架构决策记录（ADR）

> **状态**：G1–G10 已全部钉死（2026-06-05）  
> **约束**：开发时务必遵守。任何偏离须更新本文档并通知团队。  
> **关联**：[`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md)

---

## G1 · expected_return 计算公式

### 问题

方案中使用了 `expected_return` 但未给出具体公式。

### 决策

**训练时**

- `expected_return` 直接取 **未来 30 分钟的实际收益率** 作为回归目标。

**推理时（MVP）**

- 若模型未提供可靠的回归输出，使用经验公式：

```text
expected_return = confidence × historical_avg_gain
```

- `historical_avg_gain`：过去 **100 个正例样本**（做多或做空，按当前方向分别统计）的未来 30 分钟平均收益率（取绝对值）。

**推理时（正式版）**

- 使用 XGBoost 回归器 `reg_expected_return` 直接输出预期收益率。
- 与方向分类器共用特征（Transformer 32 维 + 1h 背景 + 宏观），但 **独立训练**。

### 里程碑

| 阶段 | 实现 |
|------|------|
| MVP | 经验公式过渡 |
| 正式版 | **必须** 实现 `reg_expected_return` |

---

## G2 · Transformer 训练流程（两阶段，顺序不可颠倒）

### 阶段一：Transformer 编码器预训练（自监督）

| 项 | 定稿 |
|----|------|
| 任务 | 预测序列中下一时间步的特征（特征重建） |
| 输入 | `(batch, 60, F)` |
| Mask | 随机 mask 掉 **15%** 时间步的特征 |
| 损失 | MSE，**仅计算被 mask 的位置** |
| 产出 | 丢弃预测头；保留编码器 + 输出投影层（**32 维**） |

### 阶段二：特征提取 + XGBoost 训练

1. **固定** Transformer 编码器，对所有样本提取 32 维向量。
2. 拼接 1 小时背景特征（H 维）+ 宏观特征（M 维）→ `(32 + H + M)` 维。
3. 分别训练：
   - `xgb_clf`：3 类分类（做多 / 做空 / 观望）
   - `reg_entry_offset`：仅正例样本（label ±1）
   - `reg_expected_return`：仅正例样本，目标为未来 30 分钟实际收益率

### 禁止

- **禁止** 端到端联合训练（防过拟合）。

---

## G3 · StandardScaler 持久化与加载

### 训练

```python
joblib.dump(scaler, "models/scaler.pkl")
```

- 拟合 `StandardScaler` 后必须保存至 `models/scaler.pkl`。

### 推理

```python
scaler = joblib.load("models/scaler.pkl")
```

- 启动时加载；对实时特征调用 `scaler.transform()`。
- 5 分钟序列 `(60, F)`：**先** `reshape(-1, F)` 变换，**再** 恢复 `(60, F)`。

### 异常

- 找不到 `scaler.pkl` → **程序拒绝启动并报错**。

---

## G4 · GUI「连接 MT5」双通道实现

### 通道一：数据管道（上行）

| 项 | 定稿 |
|----|------|
| 线程 | `DataReceiverThread`（独立线程） |
| 行为 | 阻塞读取 `\\.\pipe\ZhuLong_Data`，接收 M1 K 线 |
| 产出 | 合成 5 分钟 K 线，放入 `queue.Queue` |
| 断线 | Python 端作为管道服务端，`while True` + 异常重试，自动重建管道 |

### 通道二：MT5 API（持仓 / 改单）

| 项 | 定稿 |
|----|------|
| 触发 | 用户点击「连接 MT5」→ `mt5.initialize()` |
| 线程 | `PositionManagerThread`，**1 Hz** 扫描持仓 |
| 操作 | 所有改 SL/TP、部分平仓、全平 → **MT5 API**，不经过数据管道 |
| 断线 | 定时 `mt5.terminal_info()`；失效则重初始化 + GUI 提示 |

### 原则

- 两通道 **独立运行，互不阻塞**。

---

## G5 · 持仓匹配：magic_number + comment 前缀

### magic_number

```python
magic_number = hash(signal_id) & 0xFFFF  # 范围 1–65535
```

- 信号生成时计算，随信号持久化。

### comment 前缀

- 用户手动下单时，在 MT5 **Comment** 字段输入：

```text
ZhuLong_<signal_id>
```

- 示例：`ZhuLong_20260605_1530_XAUUSD_buy`

### 匹配优先级

1. **优先**：`comment` 以 `ZhuLong_` 开头 → 解析 `signal_id`
2. **降级**：价格 + 时间匹配（±5 点、信号后 60 秒内）→ **记录警告日志**

### GUI

- 信号卡片提供 **「一键复制 Comment」** 按钮，用户粘贴至 MT5 下单窗口。
- 不强制填写 comment，但强烈建议。

---

## G6 · 初始止损 / 止盈公式（写入 config）

### 配置项

```json
"initial_stop_loss_atr_mult": 1.2,
"initial_take_profit_atr_mult": 2.0
```

### 计算（信号生成时固化，持仓管理不动态更新初始 SL/TP）

- `ATR(14)`：当前 5 分钟 K 线收盘时，基于 **14 根 5 分钟 K 线** 计算。

**做多**

```text
stop_loss   = entry_price - ATR(14) × initial_stop_loss_atr_mult
take_profit = entry_price + ATR(14) × initial_take_profit_atr_mult
```

**做空**

```text
stop_loss   = entry_price + ATR(14) × initial_stop_loss_atr_mult
take_profit = entry_price - ATR(14) × initial_take_profit_atr_mult
```

---

## G7 · Broker 符号映射

### 配置

```json
"symbol_mapping": {
  "XAUUSD": "XAUUSDm",
  "USOIL": "CL-OIL"
}
```

### 规则

- 系统内部统一使用 **标准符号**（`XAUUSD`、`USOIL`）作为逻辑标识。
- 与 MT5 交互（数据、持仓、下单）时自动映射为 broker 实际符号。
- GUI 可修改映射并 **立即生效**。
- 默认映射为空对象 `{}`（标准符号 = broker 符号），由用户自行填写。

---

## G8 · 价格与点值换算

### 内部计算

- **全部** 使用百分比或 **绝对价格**（`entry_price`、`stop_loss`、`take_profit`）。
- **禁止** 在内部逻辑中使用点数计算。

### 展示

- GUI 同时显示 **绝对价格** 与 **点数**：

```text
points = abs(price_diff) / SYMBOL_POINT
```

- `SYMBOL_POINT` 来自 `SymbolInfoDouble(symbol, SYMBOL_POINT)`（MT5 合约规格）。

### 下单 / 改单

- `mt5.order_send` 的 `sl`、`tp` 使用 **绝对价格**。

### 移动止损步长

- `trailing_step_pips` 等配置为 **百分比**（如 0.10%），运行时换算为绝对价格步长。

---

## G9 · SQLite 表结构（trading.db）

数据库文件：`data/trading.db`（或 `config` 可配置路径）。

### signals

| 字段 | 类型 | 说明 |
|------|------|------|
| signal_id | TEXT PK | 唯一标识 |
| timestamp | INTEGER | Unix 时间戳 |
| symbol | TEXT | 标准符号 |
| direction | TEXT | `buy` / `sell` |
| entry_price | REAL | |
| stop_loss | REAL | |
| take_profit | REAL | |
| confidence | REAL | |
| expected_return | REAL | |
| magic_number | INTEGER | `hash(signal_id) & 0xFFFF` |
| comment_hint | TEXT | 建议 Comment 字符串 |
| status | TEXT | `pending` / `matched` / `expired` / `rejected` |
| params_snapshot | TEXT | JSON，生成时参数快照 |
| created_at | INTEGER | |

### trades

| 字段 | 类型 | 说明 |
|------|------|------|
| trade_id | INTEGER PK AUTOINCREMENT | |
| signal_id | TEXT | 外键 → signals |
| open_time | INTEGER | |
| open_price | REAL | |
| close_time | INTEGER | |
| close_price | REAL | |
| pnl_points | REAL | 盈亏点数（正为盈利） |
| pnl_percent | REAL | 盈亏百分比 |
| is_win | INTEGER | 0 / 1 |
| close_reason | TEXT | `tp` / `sl` / `time_stop` / `trailing` / `model_exit` / `manual` |

### position_events

| 字段 | 类型 | 说明 |
|------|------|------|
| event_id | INTEGER PK AUTOINCREMENT | |
| signal_id | TEXT | |
| event_time | INTEGER | |
| event_type | TEXT | `partial_close` / `move_sl` / `move_tp` / `full_close` |
| price | REAL | |
| volume | REAL | 部分平仓量 |
| old_sl | REAL | |
| new_sl | REAL | |

### 归因

- 主要查询 `trades`，联结 `signals.params_snapshot`。

---

## G10 · 线程模型（5 线程）

| 线程 | 周期 / 触发 | 职责 | 通信 |
|------|-------------|------|------|
| **GUI 主线程** | 事件驱动 | PyQt5 界面、参数修改 | `pyqtSignal` 接收其他线程消息 |
| **DataReceiverThread** | 阻塞读管道 | M1 → 合成 M5 → `queue.Queue` | 向主线程发新 K 线信号 |
| **SignalSchedulerThread** | 每 5 分钟 | 取特征 → 推理 → 过滤 → 绘图指令 | 写 `pending_signals`（加锁）→ 通知 GUI |
| **PositionManagerThread** | 1 Hz | `positions_get`、匹配、移动止损等 | MT5 API；持仓变更信号 → GUI |
| **MacroCalendarThread** | 每天一次 | 重载 `macro_events.csv` | 全局只读缓存供特征计算 |

### 同步

- 所有 `pending_signals` 访问须 `threading.Lock`。
- 持仓线程禁止长时间阻塞；订单修改应快速完成。
- MVP 阶段 MT5 API 保持 **同步** 调用。

---

## G11 · 交付形态：EXE 可安装程序 + MQ5 指标

### 决策

- **用户交付物**：
  1. **`ZhuLong_Setup.exe`** — 压缩包内安装程序（亦可单独分发）
  2. **`ZhuLongIndicator.mq5`** — 安装后位于 `{app}\indicators\`，用户复制至 MT5 编译

- **用户不接触**：Python、pip、`train.py`、虚拟环境、命令行启动

- **开发/发布分离**：
  - 开发：`python ZhuLong.py`
  - 发布：`PyInstaller` → `ZhuLong.exe` → `Inno Setup` → 安装包

- **路径约定**：
  - 程序只读资源：`{InstallDir}\`（models、默认 config、mql5）
  - 用户可写数据：`%APPDATA%\ZhuLong\`（config 覆盖、trading.db、logs）

- **版本锁定**：安装包 semver = EXE 文件版本 = 指标 `#property version` = `models/*/manifest.json`

完整规格见 **[`DELIVERY.md`](./DELIVERY.md)**。

---

## G13 · GUI 与 Python 集成：WinUI 3 + Python.NET 单进程

### 决策

- **WinUI 3**（`src/ZhuLong.App`）为唯一用户 EXE，**MVVM** + **DI**
- **C# 服务层**负责：命名管道、特征缓存、信号过滤、EF Core、运行时编排
- **Python.NET** 同进程加载 `ZhuLong.PythonEngine/inference.py`：
  - 模型推理（Transformer + XGBoost）
  - MT5 API（`MetaTrader5` 包）
- **弃用**：PyQt5、独立 `ZhuLong.Engine.exe` 子进程
- **管道**：`System.IO.Pipes.NamedPipeServerStream`（C# 服务端）

完整架构见 **[`WINUI_ARCHITECTURE.md`](./WINUI_ARCHITECTURE.md)**。

---

## 变更日志

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.3 | 2026-06-05 | G13：WinUI 3 + Python.NET 单进程完整方案 |
| v1.2 | 2026-06-05 | G12：WinUI 3 壳（已由 G13  supersede 子进程方案） |
| v1.1 | 2026-06-05 | G11：交付形态定为 EXE 安装包 + MQ5 指标 |
| v1.0 | 2026-06-05 | G1–G10 初始定稿 |
