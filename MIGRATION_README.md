# 烛龙3 移机包 (V14 GPU 训练)

## 环境
```powershell
py -3 -m pip install -r requirements.txt
# GPU PyTorch (CUDA 12.4):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## V14 训练方案（唯一生产方案）

### 黄金 XAUUSD
```powershell
# 全量训练 (400 trees, GPU auto-detect)
py -3 scripts/train_v14.py --symbol XAUUSD

# 部署到生产目录
py -3 scripts/deploy_v14_production.py
```

### 原油 USOIL
```powershell
# 全量训练 (400 trees, horizon=18, gain=0.3%)
py -3 scripts/train_v14.py --symbol USOIL --horizon 18 --gain 0.003

# USOIL 无需单独 deploy 脚本，模型直接写入 models/USOIL/v14/
```

### 快速冒烟
```powershell
py -3 scripts/train_v14.py --symbol XAUUSD --quick
```

## V14 模型说明

| 品种 | 树数 | 阈值 | 特征 | 验收标准 |
|------|------|------|------|----------|
| XAUUSD | 400 | L=0.70 S=0.70 | 68维 v13 tabular | WR>=50%, RR>=1.3, DD<=25% |
| USOIL | 400 | L=0.50 S=0.54 | 68维 v13 tabular | 同上, horizon=18, gain=0.3% |

## 已废弃方案 (不再使用)
- V13 direction / v13_triple — 未通过验收
- V12 — 被 V14 取代
- oil_v1 — 被 V14 取代
- KnowledgeNet + PPO — 基础模型准确率过低

## 已有模型 (预训练)
- models/XAUUSD/v14/ — V14 黄金 (400 trees, WR=71.0%, 验收通过)
- models/USOIL/v14/ — V14 原油 (400 trees, WR=95.3%, n_trades=64)

## 数据
- data/training/lgb/ — M5 CSV (XAUUSD 740K bars, USOIL)
- data/training/v14/ — 68维特征缓存 (parquet)

配置: config_training.yaml, config/config_scheduler.json
