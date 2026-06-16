# 烛龙系统双品种实盘部署指南

适用于 **XAUUSD（黄金 v12）** 与 **WTI/原油（USOIL v1）** 部署到 MT5 实盘。

## 一、部署前检查

- Windows 10/11 或 VPS（建议 4 核 8G+）
- MT5 已登录（模拟或真实）
- Python 3.10+（推荐 3.10.11）
- 确认经纪商原油符号（`USOIL` / `XTIUSD` / `CL-OIL` 等）

## 二、获取代码与模型

```powershell
cd d:\trae_projects\ZhuLong_3
pip install -r requirements-inference.txt
```

确认模型存在：

```
models/XAUUSD/v11/xgb_triple.json
models/USOIL/v1/xgb_triple_oil.json
```

## 三、MT5 指标

1. 复制 `indicators/ZhuLongIndicator.mq5` → MT5 数据目录 `MQL5/Indicators/`
2. MetaEditor（F4）编译
3. 将指标拖到 **XAUUSD M1** 图表
4. 将指标拖到 **原油品种 M1** 图表（与 `broker_symbol` 一致）
5. 指标属性 → 勾选 **允许导入 DLL**

指标创建管道：`\\.\pipe\ZhuLong_Data`（上行）、`\\.\pipe\ZhuLong_Drawing`（下行）。

v1.12+ 指标已支持按 `symbol` 字段过滤绘图，双图表不会互相干扰。

## 四、配置文件

| 文件 | 说明 |
|------|------|
| `config/config_xau_v12.json` | 黄金 v12 阈值/冷却/止损 |
| `config/config_oil_v1.json` | 原油 v1，**务必修改 `broker_symbol`** |

原油配置示例：

```json
{
  "training_symbol": "USOIL",
  "broker_symbol": "XTIUSD",
  ...
}
```

若 MT5 市场报价显示 `CL-OIL`，则 `"broker_symbol": "CL-OIL"`。

## 五、一键部署与启动

```powershell
# 部署模型 + 启动双品种服务
.\scripts\start_trading_dual.ps1

# 指定经纪商原油符号
.\scripts\start_trading_dual.ps1 -BrokerOil XTIUSD

# 仅测试一轮
.\scripts\start_trading_dual.ps1 -Once

# 仅部署不启动
.\scripts\start_trading_dual.ps1 -DeployOnly
```

或手动：

```powershell
py -3 scripts/deploy_dual_production.py
py -3 scripts/realtime_signal.py --config-xau config/config_xau_v12.json --config-oil config/config_oil_v1.json
```

## 六、验证

- 日志：`logs/trading.log`
- 每 30 秒轮询，新 M5 收盘时推理
- 有信号时 MT5 图表出现箭头 + 止损/止盈线
- 冷却、日限额、EIA 屏蔽按品种独立状态文件：
  - `data/realtime_state_xau.json`
  - `data/realtime_state_oil.json`

## 七、常见问题

| 问题 | 解决 |
|------|------|
| 无箭头 | 先挂指标再启 Python；检查 DLL 权限 |
| Pipe not found | 指标未运行或未编译 |
| 原油无数据 | 修改 `config_oil_v1.json` 的 `broker_symbol` |
| 黄金/原油信号画错图 | 升级指标（含 SymbolMatchesChart） |
| 双品种冷却冲突 | 已按品种分 state 文件 |

## 八、安全提醒

- 单笔风险 ≤ 1%，日亏损上限建议 2%
- 定期备份 `models/`、`config/`、`data/realtime_state_*.json`
- 模拟盘验证 2 周后再上实盘
