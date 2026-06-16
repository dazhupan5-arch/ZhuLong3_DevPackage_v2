烛龙三 · 开发迁移包
==================

解压到目标电脑，例如 D:\ZhuLong3

## 快速开始

```powershell
cd D:\ZhuLong3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 运行 USOIL 智能体训练

```powershell
python -u scripts/train_usoil_agent_until_pass.py
```

## 本包包含

- 完整源码：zhulong/、src/、ZhuLong.PythonEngine/、scripts/
- 全部配置：config/、config_training.yaml、config_agent.json
- 模型：models/（含 USOIL V14、XAUUSD 等）
- 训练数据：
  - data/training/lgb/USOIL/USOIL_M5.csv
  - data/training/v14/USOIL/features.parquet
  - data/training/lgb/XAUUSD/（黄金）

## 未包含（可忽略或从 GitHub 补）

- bin/obj/publish 构建产物（目标机重新编译）
- logs/ 日志
- 超大冗余 CSV（train_balanced_*.csv 等，可重新生成）
- .git 历史（GitHub: https://github.com/dazhupan5-arch/ZhuLong3）

## 详细说明

见 DEV_SETUP.md

打包时间：2026-06-12
