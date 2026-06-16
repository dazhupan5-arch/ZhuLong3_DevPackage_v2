# 烛龙 WinUI 3 完整实现方案

> **状态**：MVP 骨架已实现 · 构建通过  
> **开发者**：Stephen.Pan · v1.0.0

---

## 一、技术栈

| 层级 | 技术 | 项目/路径 |
|------|------|-----------|
| UI | **WinUI 3** + MVVM | `src/ZhuLong.App` → `ZhuLong.exe` |
| 本地服务 | **C# .NET 8** | `src/ZhuLong.Core` |
| 跨语言 | **Python.NET** | `Services/PythonInferenceService.cs` |
| 模型 | PyTorch + XGBoost | `zhulong/` + `ZhuLong.PythonEngine/inference.py` |
| MT5 API | MetaTrader5 via Python.NET | `Services/Mt5ApiWrapper.cs` |
| 管道 | **System.IO.Pipes** | `Core/Pipes/PipeServer.cs` |
| 数据库 | **SQLite + EF Core** | `Core/Data/ZhuLongDbContext.cs` |
| 依赖注入 | Microsoft.Extensions.DependencyInjection | `App.xaml.cs` |
| 打包 | dotnet publish + MSIX（可选）+ Inno Setup | `installer/build_installer.iss` |

---

## 二、架构图

```text
MT5 + ZhuLongIndicator.mq5
        │ 命名管道 M1 / 绘图 JSON
        ▼
┌───────────────────────────────────────────┐
│  ZhuLong.exe (WinUI 3)                    │
│  ├─ MainViewModel / NavigationView        │
│  ├─ PipeServer (C#)                       │
│  ├─ FeatureCacheService (C#)              │
│  ├─ SignalGeneratorService (C#)           │
│  ├─ ZhuLongRuntimeService (编排 G10)      │
│  ├─ DatabaseService (EF Core)             │
│  └─ Python.NET ──────────────────────┐    │
└──────────────────────────────────────│────┘
                                       ▼
                            inference.py
                            Transformer + XGBoost
                            MetaTrader5 (mt5)
```

**与旧方案差异**：不再使用独立 `ZhuLong.Engine.exe` 子进程；Python 嵌入主进程（Python.NET）。

---

## 三、项目结构

```text
ZhuLong_3/
├── src/
│   ├── ZhuLong.App/              # WinUI 3（对外名 ZhuLong.WinUI）
│   │   ├── Services/             # PythonInference, Mt5, Runtime
│   │   ├── ViewModels/           # MVVM
│   │   └── Views/                # MainPage + 面板（待扩展）
│   └── ZhuLong.Core/             # 管道、特征、信号、EF、配置
├── ZhuLong.PythonEngine/
│   └── inference.py              # predict() 入口
├── zhulong/                      # Python 模型与训练逻辑
├── mql5/ZhuLongIndicator.mq5
├── config.json
└── models/{symbol}/              # 预训练制品
```

---

## 四、已实现模块

| 模块 | 文件 | 状态 |
|------|------|------|
| 命名管道服务端 | `PipeServer.cs` | ✅ |
| M1→M5 特征缓存 | `FeatureCacheService.cs` | ✅ MVP |
| 信号过滤 | `SignalGeneratorService.cs` | ✅ |
| Python 推理 | `PythonInferenceService.cs` | ✅ |
| MT5 连接/持仓读 | `Mt5ApiWrapper.cs` | ✅ 读持仓 |
| 运行时编排 | `ZhuLongRuntimeService.cs` | ✅ |
| EF SQLite | `ZhuLongDbContext.cs` | ✅ |
| WinUI 主界面 | `MainPage.xaml` | ✅ |
| MVVM | `MainViewModel.cs` | ✅ |

---

## 五、待完善（Phase 2）

- [ ] 持仓管理完整规则（移动止损、分批止盈、G5 匹配）
- [ ] `Mt5ApiWrapper.ModifySlTp` / 平仓
- [ ] C# 特征维度对齐 Python 30 维（当前 22 维，推理前 padding）
- [ ] 参数面板 / 归因面板 / ScottPlot 图表
- [ ] 嵌入 `python_runtime/` 到安装包
- [ ] MSIX 打包项目 `ZhuLong.Package`
- [ ] ONNX 升级路径（脱离 Python 运行时）

---

## 六、开发与发布

```powershell
# 开发
dotnet run --project src/ZhuLong.App -p:Platform=x64

# 发布
dotnet publish src/ZhuLong.App -c Release -r win-x64 -p:Platform=x64 --self-contained -o publish/win-x64
iscc installer\build_installer.iss
```

### 运行前提

1. 已安装 **Python 3.10+** 与 `requirements.txt`（开发机），或安装包内 `python_runtime/`
2. `models/XAUUSD/scaler.pkl` 等模型文件存在
3. MT5 已登录，指标已挂载

---

## 七、关键决策（G1–G12 + G13）

| ID | 内容 |
|----|------|
| G13 | **单进程 Python.NET** — 弃用 Engine 子进程；C# 主程序嵌入 Python |

其余见 [`DECISIONS.md`](./DECISIONS.md)。

---

*2026-06-05*
