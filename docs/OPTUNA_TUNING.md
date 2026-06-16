# Optuna 超参数调优指南

## 快速开始

```powershell
cd d:\trae_projects\ZhuLong_3
pip install optuna
py -3 scripts/tune_hyperparams.py --symbol XAUUSD --trials 100
py -3 scripts/tune_hyperparams.py --symbol XAUUSD --trials 100 --enhanced
```

输出：`data/training/reports/optuna/XAUUSD/best_params.json`

## 方法

- **目标**：TimeSeriesSplit(5) 平均 multi_logloss（越小越好）
- **剪枝**：MedianPruner，无潜力 trial 提前结束
- **数据**：`train_balanced_v3.csv` + 三重屏障标签

## 搜索空间

| 参数 | 范围 |
|------|------|
| max_depth | 3–8 |
| learning_rate | 0.01–0.1（对数） |
| n_estimators | 200–1000 |
| subsample | 0.6–1.0 |
| colsample_bytree | 0.6–1.0 |
| reg_lambda / reg_alpha | 0–5 |
| min_child_weight | 1–10 |

## 应用最佳参数

将 `best_params.json` 中的参数写入 `scripts/train_v13_triple.py` 的 `V13_TRIPLE_XGB`，或：

```powershell
py -3 scripts/train_v13_triple.py  # 手动合并 params 后重训
```

## 注意

- 必须使用**时间序列**拆分，禁止随机 shuffle
- Optuna 优化 logloss；实盘验收仍看精确率/胜率/OOS 回测
