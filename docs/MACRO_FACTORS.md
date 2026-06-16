# 烛龙宏观因子模块

宏观特征由 **三类数据源** 合成，最终输出 **8 维向量** 供模型推理。

## 数据来源

| 特征维度 | 来源 | 运行方式 |
|---------|------|----------|
| 0–4 日历事件 | Finnhub / FMP REST API | C# 启动 + 每 24h 刷新 → SQLite `macro_events` |
| 5 FRED 综合 | `fred_latest.json` | 离线 Python `fetch_fred.py` |
| 6–7 情绪 | `sentiment.json` | 离线 Python `fetch_sentiment.py`（LLM） |

API 失败时自动降级：**Investing.com 备用 → `macro_events.csv` → SQLite 缓存**。

## 8 维特征说明

| 索引 | 名称 | 说明 |
|-----|------|------|
| 0 | hours_to_next_norm | 距下一事件小时数 / 48（上限 1） |
| 1 | next_event_impact | high=1, medium=0.5, low=0.25 |
| 2 | hours_since_last_norm | 距上一事件小时数 / 48 |
| 3 | event_just_happened | 1 小时内刚发生则为 1 |
| 4 | next_event_type | FOMC/NFP/CPI 等关键词编码 0~1 |
| 5 | fred_composite | 失业率/利率/通胀预期综合 0~1 |
| 6 | gold_silver_norm | 金银比归一化（60~100） |
| 7 | llm_sentiment | LLM 整体情绪 0~1 |

## API 密钥（烛龙设置页）

**不在 config.json 明文保存。** 请在 WinUI「设置」页填写，或写入：

- `%APPDATA%\ZhuLong\secrets\fred_api_key.txt`
- `%APPDATA%\ZhuLong\secrets\finnhub_api_key.txt`
- `%APPDATA%\ZhuLong\secrets\fmp_api_key.txt`
- `%APPDATA%\ZhuLong\secrets\llm_api_key.txt`

优先级：**环境变量 > LocalSettings > secrets 文件**。

## config.json 配置

仅保留宏观行为参数（provider、series 等），不含密钥。

## 命令

```powershell
# 离线拉取 FRED + LLM 情绪
.\scripts\fetch_macro_offline.ps1

# 启动 ZhuLong 后自动：拉取经济日历 → 读 JSON → 合成特征
```

## 静默窗口

`force_silence: true` 时，在 `force_silence_events`（如 FOMC、Nonfarm Payrolls）前后静默期内跳过信号调度。
