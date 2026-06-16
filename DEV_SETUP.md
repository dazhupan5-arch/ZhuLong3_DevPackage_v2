# 烛龙三 · 开发环境迁移说明

## 1. 获取代码

在本机（当前电脑）已初始化 Git 仓库，目录：

```
D:\ZhuLong3_Migration_20260609.zip
```

### 方式 A：U 盘 / 网盘拷贝整个目录

直接复制整个项目文件夹到另一台电脑，例如 `D:\ZhuLong3`。

### 方式 B：推送到 GitHub / Gitee 后 clone

```powershell
# 本机（首次）
cd D:\ZhuLong3_Migration_20260609.zip
git remote add origin <你的仓库地址>
git push -u origin main

# 另一台电脑
git clone <你的仓库地址> D:\ZhuLong3
cd D:\ZhuLong3
```

## 2. 环境依赖

### Python（训练 / 推理引擎）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### .NET（WinUI 壳程序，可选）

- Visual Studio 2022 + .NET 8 + Windows App SDK
- 打开 `ZhuLong.sln` 编译

## 3. 训练数据（未入库，需单独拷贝）

Git 仓库不含大体积 CSV/Parquet/NPZ，请从本机复制以下文件到另一台电脑相同路径：

| 文件 | 用途 |
|------|------|
| `data/training/lgb/USOIL/USOIL_M5.csv` | 原油 M5 原始数据 |
| `data/training/v14/USOIL/features.parquet` | V14 68 维特征缓存 |
| `data/training/lgb/XAUUSD/XAUUSD_M5.csv` | 黄金 M5（如训练 XAU） |

快速生成 USOIL npz（有 parquet 后 14 秒）：

```powershell
python -u scripts/prepare_knowledge_data.py --symbol USOIL
```

struct30 特征（CPU 密集，大核机器更快）：

```powershell
python -u scripts/prepare_training_data.py --symbol USOIL --n-jobs 4
```

## 4. 运行 USOIL 智能体训练

```powershell
cd D:\ZhuLong3
python -u scripts/train_usoil_agent_until_pass.py
```

脚本会自动循环：数据准备 → KnowledgeNet → ONNX 导出 → PPO → 回测，直到验收 PASS。

## 5. 关键配置

| 文件 | 说明 |
|------|------|
| `config_training.yaml` | KN / PPO 超参、USOIL 专用 oil 段 |
| `config/config_agent.json` | 智能体开关、模型路径 |
| `config/config_oil_v14.json` | V14 原油生产模型配置 |
| `config_v14.json` | V14 全局配置 |

## 6. 目录结构速览

```
zhulong/              Python 核心引擎 (agent, training, strategies)
ZhuLong.PythonEngine/ MT5 推理桥
scripts/              训练 / 回测 / 部署脚本
config/               运行时 JSON/YAML 配置
src/                  WinUI 3 桌面壳 (C#)
tests/                单元测试
mql5/                 MT5 指标源码
```
