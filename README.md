# 烛龙 ZhuLong III

**WinUI 3 + Python.NET 单进程** · 可安装 EXE + MQ5 指标

## 快速开始

```powershell
pip install -r requirements.txt
dotnet run --project src/ZhuLong.App -p:Platform=x64
```

## 文档

- [WinUI 完整架构](./docs/WINUI_ARCHITECTURE.md)
- [交付规格](./docs/DELIVERY.md)
- [架构决策 G1–G13](./docs/DECISIONS.md)

## 结构

| 项目 | 说明 |
|------|------|
| `src/ZhuLong.App` | WinUI 3 主程序 → `ZhuLong.exe` |
| `src/ZhuLong.Core` | 管道、特征、信号、EF Core |
| `ZhuLong.PythonEngine/` | Python.NET 推理入口 |
| `zhulong/` | PyTorch/XGBoost 训练与推理实现 |
| `train.py` | 离线训练 |

开发者：Stephen.Pan
