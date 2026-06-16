# 烛龙 ZhuLong · 用户手册

> 开发者：Stephen.Pan · 版本 1.0.0

## 1. 安装

1. 解压压缩包，双击 **`ZhuLong_Setup.exe`**
2. 按向导完成安装（默认路径 `C:\Program Files\ZhuLong\`）
3. 从安装目录 `indicators\` 复制 **`ZhuLongIndicator.mq5`** 到 MT5：
   - MT5 → **文件 → 打开数据文件夹** → `MQL5\Indicators\` → 粘贴
   - 导航器刷新，双击编译（F7）

## 2. MT5 设置

- **工具 → 选项 → 专家顾问**
  - ☑ 允许算法交易
  - ☑ 允许 DLL 导入
- 将指标拖到 **XAUUSD、USOIL 的 M1 图表**（各一张）

## 3. 启动烛龙

1. **先启动** 桌面「烛龙系统」（**WinUI 3** · `ZhuLong.exe`）
2. 点击 **「连接 MT5」**（加载配置）
3. 点击 **「开始运行」**（启动管道、推理与持仓扫描）
4. 若提示缺少模型：确认 `models\XAUUSD\` 等目录含 `scaler.pkl` 等文件

## 4. 信号与下单

- 信号出现在 **WinUI 信号列表** 与 MT5 图表（箭头 + SL/TP 线）
- **点击信号行** 复制 Comment 到剪贴板（格式 `ZhuLong_<signal_id>`）
- **手动** 按建议价入场；系统自动托管移动止损、分批止盈

## 5. 参数调节

编辑安装目录或 `%APPDATA%\ZhuLong\config.json`：

| 区块 | 作用 |
|------|------|
| `signal_filters` | 置信度、预期收益、冷却 |
| `signal_geometry` | 初始 SL/TP 的 ATR 倍数 |
| `position_management` | 移动止损、分批止盈、时间止损 |
| `symbol_mapping` | 如 `"XAUUSD": "XAUUSDm"` |

## 6. 技术图形说明（MT5 指标）

指标 `ZhuLongIndicator` v1.01 在图表上**独立绘制**以下线条（不依赖 Python 信号，随 K 线实时刷新）：

| 线条 | 颜色 | 计算方式 |
|------|------|----------|
| **ATR 上轨** | 道奇蓝 | EMA(快) + 倍数 × ATR(14) |
| **ATR 下轨** | 道奇蓝 | EMA(快) − 倍数 × ATR(14) |
| **EMA 快线** | 金色 | 默认 EMA30 |
| **EMA 慢线** | 橙色 | 默认 EMA60 |

**信号对象**（箭头、SL/TP 水平线）仍由命名管道指令绘制，与上述指标线互不冲突。

### 指标输入参数（须与 config.json 同步）

在 MT5 指标「输入」中可调：

| 参数 | 默认值 | config.json 对应项 |
|------|--------|-------------------|
| ATR 周期 | 14 | `atr_channel.period` |
| ATR 通道倍数 | 3.0 | `atr_channel.multiplier` |
| 快 EMA 周期 | 30 | `atr_channel.ema_fast` |
| 慢 EMA 周期 | 60 | `atr_channel.ema_slow` |
| 显示 ATR 通道 | 是 | — |
| 显示 EMA 线 | 是 | — |

**重要**：若修改 MT5 指标中的 ATR/EMA 参数，请同步修改 `config.json` 的 `atr_channel` 区块，否则图表所见与模型内部特征不一致。

## 7. 常见问题

| 问题 | 处理 |
|------|------|
| 管道连接失败 | 先开 ZhuLong.exe，再挂指标；必要时管理员运行 MT5 |
| 缺少 scaler.pkl | 重新安装或联系获取预训练 models |
| 持仓未匹配 | 下单时粘贴 Comment；检查方向与价格 |
| 宏观屏蔽 | 编辑 `data\macro_events.csv` |

## 8. 免责声明

本软件仅供学习研究，不构成投资建议。交易有风险，请自行负责。
