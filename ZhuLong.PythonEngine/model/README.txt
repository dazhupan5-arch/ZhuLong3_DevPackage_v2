模型制品目录说明
================

推理加载路径为安装目录下的 models/{SYMBOL}/（非本文件夹）。

本目录为 Python 引擎占位说明；四件套文件名：
  transformer_encoder.pth
  xgb_classifier.json
  xgb_regressor.json
  scaler.pkl

演示模型：scripts/create_demo_models.py
正式训练：train.py --symbol XAUUSD --m1-csv data/sample_xauusd_m1.csv
