# 特征维度对齐说明

## 概览

| 阶段 | 维度 | 实现 |
|------|------|------|
| C# `FeatureCalculator` | 22 | M5 OHLCV + 技术指标 |
| C# `FeaturePad.ToModelDim` | 22 + MTF(6) → **30** | 2 维保留填零 |
| Python `feature_engine` | **22** 基础 + **6** MTF | 推理序列 30 维 |
| Python 推理输入 | seq(60×30) + hourly(10) + macro(8) | `InferenceEngine.predict` |

## 运行时路径

1. MT5 M1 bar → `PipeServer` → `FeatureCacheService` 聚合 M5
2. `FeatureCalculator.ComputeM5Features` → `(N, 22)`
3. 取末 60 行 → `FeatureCalculator` + `FeatureMtfCalculator` → `FeaturePad.ToModelDim` → `(60, 30)`
4. `MacroFeatureBuilder.Build` → 8 维宏观向量（FRED + sentiment + 日历）
5. `PythonInferenceService.Predict` → Python `predict(symbol, seq, hourly, macro)`

## 对齐策略（V0.9）

- **价格/指标维（0–21）**：C# 与 Python 公式对齐（RSI/MACD/成交量等）；golden 测试 `tests/fixtures/golden_m5.csv`
- **MTF 维（22–27）**：C# `FeatureMtfCalculator` 对齐 Python `compute_mtf_trend_features`
- **保留维（28–29）**：填零
- **宏观维（独立 8 维）**：由 `MacroFeatureBuilder` 在 C# 侧计算，经 Python.NET 传入推理，与 Python `macro` 参数一致。

列定义见 `zhulong/feature_schema.json`。
