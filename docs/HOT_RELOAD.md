# 烛龙 ZhuLong — 热更新 vs 需重启

调参页保存后，`ZhuLongRuntimeService.ReloadSettingsFromDisk()` 会立即生效的项如下。

| 配置段 | 字段 | 热更新 | 说明 |
|--------|------|--------|------|
| `signal_filters` | 全部 | ✅ | 下一根 M5 信号周期生效 |
| `signal_geometry` | SL/TP ATR 倍数 | ✅ | 仅影响**新**信号 |
| `position_management` | 全部 | ✅ | 下一持仓扫描周期生效 |
| `risk_guard` | 全部 | ✅ | 立即参与风控判断 |
| `macro` | 静默窗口等 | ✅ | `MacroCalendarService` 重载 |
| `mt5` | `deviation` | ✅ | 改单/平仓滑点 |
| `symbol_mapping` | 全部 | ✅ | 映射即时更新 |
| `atr_channel` | period/mult/EMA | ⚠️ 部分 | 须同步 MT5 指标输入参数 |
| `model` | seq_len、symbols 等 | ❌ | 需停止后重新连接 |
| `pipes` | 管道名 | ❌ | 需重启应用 |
| API 密钥 | secrets | ❌ | 设置页保存后点「刷新宏观」 |

**会员授权**：激活后立即刷新，无需重启。

**Python 模型文件**：替换 `models/` 下 pkl 后需断开 MT5 并重新「连接 MT5」以重载推理引擎。
