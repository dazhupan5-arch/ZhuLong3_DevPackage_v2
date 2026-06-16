# 烛龙（ZhuLong）交付方案 · WinUI 3 + Python.NET

> **主程序**：`ZhuLong.exe`（WinUI 3 · 单进程）  
> **指标**：`ZhuLongIndicator.mq5`

---

## 交付物

| 文件 | 说明 |
|------|------|
| `ZhuLong_Setup.exe` | Inno Setup 安装程序 |
| `ZhuLongIndicator.mq5` | MT5 指标 |

安装目录含：

- `ZhuLong.exe` — WinUI 3 主程序
- `ZhuLong.PythonEngine/` — `inference.py`
- `requirements.txt` + `install_python_deps.ps1` — **目标机需安装 Python 3.10+**，运行脚本安装依赖（**不捆绑 python_runtime**）
- `models/` — 预训练模型
- `config.json`、`data/macro_events.csv`

---

## 技术栈

- **UI**：WinUI 3 + MVVM（CommunityToolkit.Mvvm）
- **服务层**：C# .NET 8（管道、特征、信号、EF Core）
- **推理 / MT5**：Python.NET 调用 `inference.py` + `MetaTrader5`
- **打包**：`dotnet publish` + Inno Setup（MSIX 可选）

详见 [`WINUI_ARCHITECTURE.md`](./WINUI_ARCHITECTURE.md)。

---

## 用户流程

1. 安装 `ZhuLong_Setup.exe`
2. 安装 **Python 3.10+**（勾选 Add to PATH），在烛龙目录运行 `install_python_deps.ps1`
3. 复制 `indicators\ZhuLongIndicator.mq5` → MT5 `MQL5\Indicators\`
4. 启动 **烛龙系统** → **连接 MT5** → **开始运行**

---

*v3.1 · WinUI 3 + Python.NET 单进程*
