# 烛龙（ZhuLong）模型训练验收标准

用途：Cursor 完成模型训练后，项目方（Stephen.Pan）依据本标准验收，判断模型是否达到上线要求。

验收前提：训练数据为 5 万根 5 分钟 K 线（约 6 个月），验证集为时间上最近的 20% 数据（约 1.2 万根，产生约 8000 个样本）。另预留最近 2 周 M5 为完全样本外。

## 自动化实现

| 脚本 | 说明 |
|------|------|
| `train.py` | 单次训练 + 验收；**未通过不写入 `models/`** |
| `scripts/train_until_accepted.py` | 网格搜索循环重训直至全部达标 |
| `zhulong/training/acceptance.py` | 指标计算 |
| `zhulong/training/pipeline.py` | 数据划分、训练、报告产出 |

## App 门禁

`manifest.json` 必须 `acceptance_passed: true` 且 `kind: production` 才会被 `InferenceEngine` 加载。  
`kind: demo` 仅用于开发演示；`kind: rejected` 存放在 `data/training/reports/rejected/`。

## 交付物

训练通过后 `data/training/reports/{symbol}/{timestamp}/` 含：

- `acceptance_report.json`
- `confusion_matrix.png`
- `feature_importance.png`
- `MODEL_README.md`
- `oos_equity.png`（样本外资金曲线）

## 一、验收指标（必须全部满足）

详见项目方原文档。代码中阈值见 `zhulong/training/acceptance.py` → `AcceptanceThresholds`。

## 五、不通过处理

运行 `py -3 scripts/train_until_accepted.py` 自动调整标签阈值与 XGBoost 超参重训。  
**不合格模型禁止放入 `models/` 与 app 运行目录。**
