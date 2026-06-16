# 烛龙 V1.0 实机验收清单

> 模拟盘 ≥3 交易日，或连续 8 小时联调通过。

## 前置

- [ ] 已安装 `ZhuLong_Setup.exe` 或 `publish/win-x64`
- [ ] MT5 模拟账户已登录
- [ ] `ZhuLongIndicator.mq5` 已编译并挂到 XAUUSD M1
- [ ] `config.json` 中 `symbol_mapping` 与券商符号一致（如有后缀）

## L2-1 安装与启动

- [ ] 无独立 Python 环境下双击 `ZhuLong.exe` 可启动
- [ ] 日志无 Python 初始化致命错误

## L2-2 管道

- [ ] 点击「开始运行」后管道状态变为「等待 MT5 连接」或「已接收 M1」
- [ ] MT5 Experts 日志无 pipe 连接错误
- [ ] `python scripts/smoke_pipe.py` 可发送 bar（可选）

## L2-3 信号

- [ ] 5 分钟调度后 WinUI 信号列表有新条目
- [ ] MT5 图表出现箭头/SL/TP 对象
- [ ] 信号 Comment 可复制（点击列表行）

## L2-4 匹配托管

- [ ] 按信号方向手动下单，Comment = `ZhuLong_<signal_id>`
- [ ] 持仓面板出现「已匹配」
- [ ] SQLite `%APPDATA%\ZhuLong\trading.db` 有 `signals` 记录

## L2-5 改单

- [ ] 浮盈达到 `trailing_activation_pct` 后 SL 上移（Experts/日志有改单）
- [ ] `position_events` 表有 `partial_close` 或 `full_close`（如触发）

## L2-6 版本一致

- [ ] EXE 标题/版本与 `config.json` `app.version` 一致
- [ ] 指标 ATR period/mult、EMA 与 `atr_channel` 一致

## 记录

| 日期 | 品种 | 信号数 | 匹配数 | 改单数 | 备注 |
|------|------|--------|--------|--------|------|
|      |      |        |        |        |      |
